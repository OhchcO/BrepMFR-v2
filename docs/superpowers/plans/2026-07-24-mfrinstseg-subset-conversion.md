# MFRInstSeg Subset and BrepMFR Conversion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build reproducible class-balanced MFRInstSeg manifests and convert only the selected STEP models into BrepMFR-compatible DGL `.bin` files while preserving semantic and instance supervision.

**Architecture:** Keep two thin command-line entry points and put reusable logic in a focused `tools/mfrinstseg` package. The subset pipeline parses and validates labels, performs deterministic multi-label balancing, and writes manifests/reports; the conversion pipeline vendors the proven OCC/occwl extractor, maps its output into the existing BrepMFR graph contract, validates every result, and writes atomically.

**Tech Stack:** Python 3.8, NumPy 1.22, SciPy 1.10, PyTorch 1.13, DGL 1.1, pythonocc-core 7.5.1, occwl 3.0, standard-library `unittest`, Conda environment `aagmspenv`.

## Global Constraints

- Official splits stay isolated: training samples only from `train.txt`, small validation only from `val.txt`, and `test.txt` never participates in selection or tuning.
- `train_1pct.txt` contains `floor(43,745 × 0.01) = 437` unique IDs; `val_small.txt` contains 437 unique IDs.
- Default seed is exactly `2026`; identical inputs and seed must produce byte-identical manifests and hashes.
- Class balancing is model-level, multi-label, rare-class-aware, and deterministic; face-level counts are reported but are not claimed to be exactly equal.
- The first conversion pass processes only `train_1pct` and `val_small`; `val_full` and `test` are converted later with the same cache.
- A sample referenced by multiple manifests is geometrically extracted only once.
- DGL metadata values are tensors only. Preserve `instance[N,N]` and `bottom[N]`; derive the sample ID from `<sample_id>.bin`.
- Existing BrepMFR fields and shapes remain compatible: face grid `[N,5,5,7]`, edge grid `[E,5,12]` in the extractor and `[E,5,7]` in the saved graph, `edges_path[N,N,6]`, and D2/A3 frequency histograms `[N,N,64]`.
- Path edge IDs use the current BrepMFR convention: real edges are zero-based and unused path positions are `-1`.
- Do not load the approximately 20 GB `aag/graphs.json`.
- Do not modify the network, attention layers, instance loss, clustering, or training code in this plan.
- Use `apply_patch` for repository file changes and preserve the user's untracked `tmp/` directory.

---

## File Structure

Create or modify only the following files:

```text
tools/
  __init__.py
  build_mfrinstseg_subset.py          # public subset CLI
  convert_mfrinstseg_to_brepmfr.py   # public conversion CLI
  mfrinstseg/
    __init__.py
    labels.py                         # official split/label parsing and validation
    sampling.py                       # deterministic multi-label balancing
    reports.py                        # statistics, hashes, atomic manifest/report writes
    aag_extractor.py                  # vendored OCC/occwl STEP feature extractor
    conversion.py                     # extractor output -> validated DGL graph
    mfrinstseg_aag_schema.json        # exact extraction attribute schema
tests/
  __init__.py
  test_mfrinstseg_labels.py
  test_mfrinstseg_sampling.py
  test_mfrinstseg_subset_cli.py
  test_mfrinstseg_aag_extractor.py
  test_mfrinstseg_conversion.py
  test_mfrinstseg_conversion_cli.py
docs/
  mfrinstseg_data_preparation.md      # commands and output explanation
```

No loader change is included. Compatibility is verified by calling the existing `CADSynth.load_one_graph()` on a generated file. Manifest-aware dataset loading is a later network/training task.

---

### Task 1: Parse and validate official labels and split files

**Files:**
- Create: `tools/__init__.py`
- Create: `tools/mfrinstseg/__init__.py`
- Create: `tools/mfrinstseg/labels.py`
- Create: `tests/__init__.py`
- Create: `tests/test_mfrinstseg_labels.py`

**Interfaces:**
- Produces: `LabelRecord`, `read_split_ids(path)`, and `load_label_record(path, expected_id=None)`.
- `LabelRecord.semantic` is `np.ndarray[int64] [N]`.
- `LabelRecord.instance` is `np.ndarray[uint8] [N,N]`.
- `LabelRecord.bottom` is `np.ndarray[uint8] [N]`.
- `LabelRecord.class_presence` and `face_counts` feed Task 2.

- [ ] **Step 1: Write failing parser tests**

Create a real temporary JSON fixture in `tests/test_mfrinstseg_labels.py`; do not mock file I/O:

```python
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tools.mfrinstseg.labels import (
    LabelFormatError,
    load_label_record,
    read_split_ids,
)


class LabelParsingTests(unittest.TestCase):
    def test_loads_official_wrapped_label_format(self):
        payload = [["part_7", {
            "seg": {"0": 2, "1": 4, "2": 4},
            "inst": [[1, 0, 0], [0, 1, 1], [0, 1, 1]],
            "bottom": {"0": 0, "1": 1, "2": 1},
        }]]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "part_7.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            record = load_label_record(path, expected_id="part_7")

        self.assertEqual(record.sample_id, "part_7")
        np.testing.assert_array_equal(record.semantic, [2, 4, 4])
        self.assertEqual(record.class_presence, frozenset({2, 4}))
        self.assertEqual(record.face_counts, {2: 1, 4: 2})
        self.assertEqual(record.instance_count, 2)

    def test_rejects_noncontiguous_face_keys(self):
        payload = [["part_7", {
            "seg": {"0": 2, "2": 4},
            "inst": [[1, 0], [0, 1]],
            "bottom": {"0": 0, "1": 0},
        }]]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "part_7.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(LabelFormatError, "contiguous"):
                load_label_record(path)

    def test_rejects_asymmetric_instance_matrix(self):
        payload = [["part_7", {
            "seg": {"0": 2, "1": 4},
            "inst": [[1, 1], [0, 1]],
            "bottom": {"0": 0, "1": 0},
        }]]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "part_7.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(LabelFormatError, "symmetric"):
                load_label_record(path)

    def test_split_reader_rejects_duplicate_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "train.txt"
            path.write_text("a\nb\na\n", encoding="utf-8")
            with self.assertRaisesRegex(LabelFormatError, "duplicate"):
                read_split_ids(path)
```

- [ ] **Step 2: Run the tests and verify RED**

Run:

```powershell
conda run -n aagmspenv python -m unittest tests.test_mfrinstseg_labels -v
```

Expected: import failure for `tools.mfrinstseg.labels`.

- [ ] **Step 3: Implement the validated data model**

Implement these exact public objects in `tools/mfrinstseg/labels.py`:

```python
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, FrozenSet, Optional, Tuple
import json

import numpy as np


class LabelFormatError(ValueError):
    pass


@dataclass(frozen=True)
class LabelRecord:
    sample_id: str
    semantic: np.ndarray
    instance: np.ndarray
    bottom: np.ndarray

    @property
    def class_presence(self) -> FrozenSet[int]:
        return frozenset(int(value) for value in np.unique(self.semantic))

    @property
    def face_counts(self) -> Dict[int, int]:
        values, counts = np.unique(self.semantic, return_counts=True)
        return {int(value): int(count) for value, count in zip(values, counts)}

    @property
    def instance_count(self) -> int:
        return int(np.unique(self.instance, axis=0).shape[0])
```

`read_split_ids()` must strip blank lines, preserve official order, reject missing/empty files, and reject duplicates. `load_label_record()` must:

1. require exactly one root entry of `[sample_id, payload]`;
2. require `seg`, `inst`, and `bottom`;
3. convert string face keys to integer indices;
4. require keys exactly `0..N-1`;
5. require an `N×N` binary, symmetric instance matrix with all diagonal values equal to 1;
6. require `bottom` length `N` and binary values;
7. require integer semantic labels in the BrepMFR/MFRInstSeg range `0..24`;
8. reject an ID mismatch against `expected_id`.

- [ ] **Step 4: Run parser tests and verify GREEN**

Run:

```powershell
conda run -n aagmspenv python -m unittest tests.test_mfrinstseg_labels -v
```

Expected: 4 tests pass.

- [ ] **Step 5: Commit Task 1**

```powershell
git add tools/__init__.py tools/mfrinstseg/__init__.py tools/mfrinstseg/labels.py tests/__init__.py tests/test_mfrinstseg_labels.py
git commit -m "feat: validate MFRInstSeg labels"
```

---

### Task 2: Select deterministic multi-label balanced subsets

**Files:**
- Create: `tools/mfrinstseg/sampling.py`
- Create: `tests/test_mfrinstseg_sampling.py`

**Interfaces:**
- Consumes: `Sequence[LabelRecord]` from Task 1.
- Produces: `SelectionResult(sample_ids, class_ids, occurrence_counts, objective_before_swaps, objective_after_swaps)` and `imbalance_objective(counts)`.
- Public function: `select_balanced(records, size, seed=2026, max_swap_attempts=5000)`.

- [ ] **Step 1: Write failing deterministic balancing tests**

Use in-memory `LabelRecord` objects with complete arrays:

```python
import unittest
import numpy as np

from tools.mfrinstseg.labels import LabelRecord
from tools.mfrinstseg.sampling import imbalance_objective, select_balanced


def record(sample_id, labels):
    semantic = np.asarray(labels, dtype=np.int64)
    instance = np.eye(len(labels), dtype=np.uint8)
    bottom = np.zeros(len(labels), dtype=np.uint8)
    return LabelRecord(sample_id, semantic, instance, bottom)


class BalancedSelectionTests(unittest.TestCase):
    def setUp(self):
        self.records = [
            record("common_0", [0]),
            record("common_1", [0]),
            record("common_2", [0]),
            record("common_3", [0]),
            record("rare_1", [0, 1]),
            record("rare_2", [0, 2]),
            record("rare_12", [1, 2]),
        ]

    def test_covers_rare_classes_when_capacity_allows(self):
        result = select_balanced(self.records, size=3, seed=2026)
        selected = {r.sample_id: r for r in self.records}
        covered = set()
        for sample_id in result.sample_ids:
            covered.update(selected[sample_id].class_presence)
        self.assertEqual(covered, {0, 1, 2})

    def test_same_seed_is_byte_order_deterministic(self):
        first = select_balanced(self.records, size=4, seed=2026)
        second = select_balanced(self.records, size=4, seed=2026)
        self.assertEqual(first.sample_ids, second.sample_ids)
        self.assertEqual(first.occurrence_counts, second.occurrence_counts)

    def test_swap_stage_never_worsens_objective(self):
        result = select_balanced(
            self.records, size=4, seed=2026, max_swap_attempts=100
        )
        self.assertLessEqual(
            result.objective_after_swaps,
            result.objective_before_swaps,
        )

    def test_balanced_objective_is_not_worse_than_seeded_random(self):
        result = select_balanced(self.records, size=4, seed=2026)
        rng = np.random.default_rng(2026)
        random_indices = rng.choice(len(self.records), size=4, replace=False)
        random_records = [self.records[int(index)] for index in random_indices]
        class_ids = sorted({c for item in self.records for c in item.class_presence})
        random_counts = np.asarray([
            sum(c in item.class_presence for item in random_records)
            for c in class_ids
        ])
        self.assertLessEqual(
            result.objective_after_swaps,
            imbalance_objective(random_counts),
        )

    def test_rejects_invalid_size(self):
        with self.assertRaisesRegex(ValueError, "size"):
            select_balanced(self.records, size=8, seed=2026)
```

- [ ] **Step 2: Run tests and verify RED**

Run:

```powershell
conda run -n aagmspenv python -m unittest tests.test_mfrinstseg_sampling -v
```

Expected: import failure for `tools.mfrinstseg.sampling`.

- [ ] **Step 3: Implement vectorized greedy selection**

Build a boolean presence matrix `[num_models, num_classes]` from the union of observed class IDs. At each iteration evaluate adding every unselected row using this lexicographic objective:

```python
def imbalance_objective(counts: np.ndarray) -> tuple:
    uncovered = int(np.count_nonzero(counts == 0))
    mean = float(counts.mean())
    coefficient_of_variation = (
        float(counts.std() / mean) if mean > 0.0 else float("inf")
    )
    spread = int(counts.max() - counts.min())
    return uncovered, coefficient_of_variation, spread
```

Choose candidates by:

1. minimum uncovered class count;
2. minimum coefficient of variation;
3. minimum max-minus-min spread;
4. maximum sum of `1 / full_split_frequency[c]` over the candidate's classes;
5. minimum seeded tie rank from one `np.random.default_rng(seed).permutation(M)`.

After greedy selection, perform exactly `max_swap_attempts` deterministic proposed swaps from the same RNG. Accept a swap only when `imbalance_objective(new_counts) < imbalance_objective(current_counts)`. Sort final IDs lexically before returning so manifest bytes do not depend on internal selection order.

Do not optimize face counts in the main objective. Include face counts only in reports.

- [ ] **Step 4: Run tests and verify GREEN**

Run:

```powershell
conda run -n aagmspenv python -m unittest tests.test_mfrinstseg_sampling -v
```

Expected: 5 tests pass.

- [ ] **Step 5: Commit Task 2**

```powershell
git add tools/mfrinstseg/sampling.py tests/test_mfrinstseg_sampling.py
git commit -m "feat: add balanced MFRInstSeg sampling"
```

---

### Task 3: Generate manifests, hashes, and statistics from the public CLI

**Files:**
- Create: `tools/mfrinstseg/reports.py`
- Create: `tools/build_mfrinstseg_subset.py`
- Create: `tests/test_mfrinstseg_subset_cli.py`

**Interfaces:**
- Consumes: Task 1 parsing and Task 2 selection.
- Produces:
  - `manifests/train_1pct.txt`
  - `manifests/val_small.txt`
  - `manifests/val_full.txt`
  - `manifests/test.txt`
  - `reports/subset_statistics.json`
  - `reports/subset_statistics.csv`
- Public orchestration function:

```python
build_subset(
    dataset_root: Path,
    output_root: Path,
    train_ratio: float = 0.01,
    val_small_size: Optional[int] = None,
    seed: int = 2026,
    dry_run: bool = False,
) -> dict
```

- [ ] **Step 1: Write a failing end-to-end CLI test**

Create a temporary miniature official-layout dataset with six training, four validation, and two test labels. Invoke the actual script with `subprocess.run`, then assert:

```python
self.assertEqual(result.returncode, 0, result.stderr)
self.assertEqual(len(train_ids), 3)       # --train-ratio 0.5
self.assertEqual(len(val_small_ids), 3)   # --val-small-size 3
self.assertEqual(val_full_ids, official_val_ids)
self.assertEqual(test_ids, official_test_ids)
self.assertTrue(set(train_ids).isdisjoint(val_full_ids))
self.assertTrue(set(train_ids).isdisjoint(test_ids))
self.assertIn("manifest_sha256", statistics)
self.assertIn("class_model_occurrences", statistics["train_1pct"])
```

Run the command twice into two output directories and compare every generated file byte-for-byte. Add a second invocation with `--dry-run` and assert the output directory is not created.

- [ ] **Step 2: Run the test and verify RED**

Run:

```powershell
conda run -n aagmspenv python -m unittest tests.test_mfrinstseg_subset_cli -v
```

Expected: failure because `tools/build_mfrinstseg_subset.py` does not exist.

- [ ] **Step 3: Implement atomic writers and statistics**

In `reports.py`, provide:

```python
def atomic_write_text(path: Path, text: str) -> None
def sha256_file(path: Path) -> str
def summarize_records(records: Sequence[LabelRecord]) -> dict
def write_statistics_csv(path: Path, summaries: Mapping[str, dict]) -> None
```

Use `tempfile.NamedTemporaryFile(delete=False, dir=path.parent)` followed by `os.replace()`. JSON must use UTF-8, `ensure_ascii=False`, `indent=2`, and `sort_keys=True`; manifests must end with exactly one newline.

Statistics for each subset must include:

- sample count;
- sorted class IDs;
- model occurrence count per class;
- face count per class;
- min/max/mean/median face count per model;
- min/max/mean/median instance count per model;
- class coverage;
- occurrence max-minus-min;
- occurrence coefficient of variation.

The top-level JSON must include arguments, official split sizes, all split-intersection sizes, and SHA-256 values for all four manifests.
If an observed class cannot be covered, or the balanced subset objective is worse than the same-seed random subset objective, include a machine-readable warning and print it to stderr; the official 437/437 run cannot be accepted while either warning is present.

- [ ] **Step 4: Implement the CLI**

`tools/build_mfrinstseg_subset.py` must resolve repository imports when called as a file:

```python
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
```

CLI arguments:

```text
--dataset-root       required Path
--output-root        required Path
--train-ratio        float, default 0.01, range (0,1]
--val-small-size     optional positive int; default actual train subset size
--seed               int, default 2026
--max-swap-attempts  int, default 5000
--dry-run            flag
```

Compute training size with `math.floor()`. Read only `labels/<id>.json`; never open `aag/graphs.json`. Validate that official train/val/test IDs are pairwise disjoint before selection. A missing label or malformed record is fatal and must identify the sample ID.

- [ ] **Step 5: Run all subset tests**

Run:

```powershell
conda run -n aagmspenv python -m unittest `
  tests.test_mfrinstseg_labels `
  tests.test_mfrinstseg_sampling `
  tests.test_mfrinstseg_subset_cli -v
```

Expected: all tests pass.

- [ ] **Step 6: Run a real-data dry run**

Run:

```powershell
conda run -n aagmspenv python tools/build_mfrinstseg_subset.py `
  --dataset-root D:\Code\Datasets\MFRInstSeg `
  --output-root D:\Code\Datasets\MFRInstSeg_BrepMFR `
  --train-ratio 0.01 `
  --val-small-size 437 `
  --seed 2026 `
  --dry-run
```

Expected:

```text
official train=43745 val=9375 test=9375
selected train_1pct=437 val_small=437
split intersections=0
dry-run: no files written
```

- [ ] **Step 7: Commit Task 3**

```powershell
git add tools/mfrinstseg/reports.py tools/build_mfrinstseg_subset.py tests/test_mfrinstseg_subset_cli.py
git commit -m "feat: generate MFRInstSeg experiment manifests"
```

---

### Task 4: Vendor and constrain the OCC/occwl AAG extractor

**Files:**
- Create: `tools/mfrinstseg/aag_extractor.py`
- Create: `tools/mfrinstseg/mfrinstseg_aag_schema.json`
- Create: `tests/test_mfrinstseg_aag_extractor.py`

**Interfaces:**
- Produces: `AAGExtractor(step_file, attribute_schema, scale_body=True).process()`.
- Output keys:
  - `graph: {"edges": (src, dst), "num_nodes": N}`
  - `graph_face_attr: list[N][10]`
  - `graph_face_grid: list[N][7][5][5]`
  - `graph_face_loop_count: list[N]`
  - `graph_face_point_pool: list[N]` of arrays shaped `[K_i,3]`
  - `bounding_box_diagonal: positive float`
  - `graph_edge_attr: list[E][9]`
  - `graph_edge_grid: list[E][12][5]`
- This module may import OCC and occwl; Tasks 1–3 must remain importable without importing it.

- [ ] **Step 1: Write a failing conditional integration test**

Use `MFRINSTSEG_ROOT` when set, otherwise fall back to the known local path and skip only when the STEP file is absent:

```python
ROOT = Path(os.environ.get(
    "MFRINSTSEG_ROOT",
    r"D:\Code\Datasets\MFRInstSeg",
))
SAMPLE_ID = "20221121_154647_0"

@unittest.skipUnless(
    (ROOT / "steps" / f"{SAMPLE_ID}.step").is_file(),
    "real MFRInstSeg sample is unavailable",
)
def test_extracted_shapes_match_official_label(self):
    schema = load_schema()
    output = AAGExtractor(
        ROOT / "steps" / f"{SAMPLE_ID}.step",
        schema,
        scale_body=True,
    ).process()
    labels = load_label_record(
        ROOT / "labels" / f"{SAMPLE_ID}.json",
        expected_id=SAMPLE_ID,
    )
    n = len(labels.semantic)
    e = len(output["graph"]["edges"][0])
    self.assertEqual(output["graph"]["num_nodes"], n)
    self.assertEqual(np.asarray(output["graph_face_grid"]).shape, (n, 7, 5, 5))
    self.assertEqual(len(output["graph_face_loop_count"]), n)
    self.assertEqual(len(output["graph_face_point_pool"]), n)
    self.assertTrue(all(len(points) > 0 for points in output["graph_face_point_pool"]))
    self.assertGreater(output["bounding_box_diagonal"], 0.0)
    self.assertEqual(np.asarray(output["graph_edge_attr"]).shape, (e, 9))
    self.assertEqual(np.asarray(output["graph_edge_grid"]).shape, (e, 12, 5))
```

- [ ] **Step 2: Run the integration test and verify RED**

Run:

```powershell
$env:MFRINSTSEG_ROOT='D:\Code\Datasets\MFRInstSeg'
conda run -n aagmspenv python -m unittest tests.test_mfrinstseg_aag_extractor -v
```

Expected: import failure for `tools.mfrinstseg.aag_extractor`.

- [ ] **Step 3: Add the exact extraction schema**

Create `mfrinstseg_aag_schema.json`:

```json
{
  "face_attributes": [
    "Plane",
    "Cylinder",
    "Cone",
    "SphereFaceAttribute",
    "TorusFaceAttribute",
    "FaceAreaAttribute",
    "RationalNurbsFaceAttribute",
    "FaceCentroidAttribute"
  ],
  "edge_attributes": [
    "Concave edge",
    "Convex edge",
    "Smooth",
    "EdgeLengthAttribute",
    "CircularEdgeAttribute",
    "EllipticalEdgeAttribute",
    "NonRationalBSplineEdgeAttribute",
    "RationalBSplineEdgeAttribute",
    "StraightEdgeAttribute"
  ],
  "UV-grid": {
    "num_srf_u": 5,
    "num_srf_v": 5,
    "num_crv_u": 5
  }
}
```

- [ ] **Step 4: Vendor the proven extractor and remove unrelated CLI code**

Use the existing user-owned implementation at:

```text
tools\AAGExtractor.py
```

Port only `scale_solid_to_unit_box`, `TopologyChecker`, and `AAGExtractor`. Do not port the dataset-statistics CLI, multiprocessing entry point, the old UV-clamping point helpers, or hard-coded paths.

Apply these required corrections during the port:

1. check `STEPControl_Reader.ReadFile()` and `TransferRoots()` status instead of silently accepting a failed STEP read;
2. retain `scale_body=True`;
3. create the edge list only from edges that also produced attributes and grids, so graph edge count always equals feature count;
4. append `graph_face_loop_count` using `TopologyUtils.TopologyExplorer(face.topods_shape(), ignore_orientation=True).number_of_wires()`;
5. build `graph_face_point_pool` from a `32×32` UV point grid filtered by the corresponding `inside` trimming mask; retain all valid XYZ points and fail if a face yields none;
6. compute `bounding_box_diagonal` from the scaled body's axis-aligned bounding box;
7. never print from library code; raise `StepExtractionError(sample_path, reason)`;
8. return arrays/data only and do not return the extractor itself.

Add:

```python
def load_schema() -> dict:
    path = Path(__file__).with_name("mfrinstseg_aag_schema.json")
    return json.loads(path.read_text(encoding="utf-8"))
```

- [ ] **Step 5: Run the real extraction test and verify GREEN**

Run:

```powershell
$env:MFRINSTSEG_ROOT='D:\Code\Datasets\MFRInstSeg'
conda run -n aagmspenv python -m unittest tests.test_mfrinstseg_aag_extractor -v
```

Expected: 1 test passes, node count equals 41 for the fixed sample, and no test is skipped locally.

- [ ] **Step 6: Commit Task 4**

```powershell
git add tools/mfrinstseg/aag_extractor.py tools/mfrinstseg/mfrinstseg_aag_schema.json tests/test_mfrinstseg_aag_extractor.py
git commit -m "feat: extract MFRInstSeg STEP geometry"
```

---

### Task 5: Build the exact BrepMFR graph and preserve instance labels

**Files:**
- Create: `tools/mfrinstseg/conversion.py`
- Create: `tests/test_mfrinstseg_conversion.py`

**Interfaces:**
- Consumes: Task 1 `LabelRecord` and Task 4 extraction dictionary/schema.
- Produces:

```python
build_brepmfr_graph(
    extracted: Mapping[str, object],
    labels: LabelRecord,
    schema: Mapping[str, object],
    max_hop: int = 6,
    histogram_bins: int = 64,
    pair_samples: int = 512,
) -> Tuple[dgl.DGLGraph, Dict[str, torch.Tensor]]
```

- [ ] **Step 1: Write failing synthetic graph-contract tests**

Build a complete two-node, two-directed-edge extraction fixture. Assert:

```python
graph, metadata = build_brepmfr_graph(extracted, labels, schema)
self.assertEqual(set(graph.ndata), {"x", "y", "z", "l", "a", "f"})
self.assertEqual(set(graph.edata), {"x", "t", "l", "a", "c"})
self.assertEqual(tuple(graph.ndata["x"].shape), (2, 5, 5, 7))
self.assertEqual(tuple(graph.edata["x"].shape), (2, 5, 7))
self.assertEqual(tuple(metadata["edges_path"].shape), (2, 2, 6))
self.assertEqual(tuple(metadata["d2_distance"].shape), (2, 2, 64))
self.assertEqual(tuple(metadata["angle_distance"].shape), (2, 2, 64))
self.assertTrue(torch.equal(metadata["instance"], torch.tensor([[1, 0], [0, 1]])))
self.assertTrue(torch.equal(metadata["bottom"], torch.tensor([0, 1])))
self.assertTrue(torch.equal(graph.ndata["f"], torch.tensor([2, 4], dtype=torch.int32)))
self.assertEqual(int(metadata["edges_path"][0, 1, 0]), 0)
self.assertEqual(int(metadata["edges_path"][0, 0, 0]), -1)
self.assertTrue(torch.allclose(
    graph.edata["x"][:, :, 6],
    graph.edata["a"].unsqueeze(1).expand(-1, 5),
))
```

Add focused tests for:

- node/label count mismatch;
- asymmetric or missing reverse graph edges;
- NaN in any extracted floating feature;
- loop counts preserved rather than replaced by ones;
- a nonzero edge angle computed from left/right edge-grid normals;
- the saved edge grid's seventh channel equals the signed dihedral angle;
- D2/A3 histogram diagonals explicitly zeroed;
- repeated conversion of the same sample produces identical histograms.

- [ ] **Step 2: Run graph-contract tests and verify RED**

Run:

```powershell
conda run -n aagmspenv python -m unittest tests.test_mfrinstseg_conversion -v
```

Expected: import failure for `tools.mfrinstseg.conversion`.

- [ ] **Step 3: Implement feature mappings**

Use schema-name lookups rather than hard-coded column offsets.

Face mappings:

```text
ndata["x"] = transpose [N,7,5,5] -> [N,5,5,7], float32
ndata["y"] = FaceAreaAttribute, float32
ndata["z"] = Plane=0, Cylinder=1, Cone=2, Sphere=3, Torus=4, otherwise=5
ndata["l"] = graph_face_loop_count, int32
ndata["a"] = in-degree, int32
ndata["f"] = semantic labels, int32
```

Edge mappings:

```text
edata["x"] = [XYZ(3), tangent(3), signed dihedral angle(1)], [E,5,7], float32
edata["l"] = EdgeLengthAttribute, float32
edata["c"] = Smooth=0, Concave=1, Convex=2, int32
edata["t"] = unknown=0, Straight=1, Circular=2, Elliptical=3,
             NonRationalBSpline=4, RationalBSpline=5, int32
```

Compute `edata["a"]` from edge-grid left/right normals:

```python
left = edge_grid[:, 6:9, :].transpose(0, 2, 1)
right = edge_grid[:, 9:12, :].transpose(0, 2, 1)
dots = np.clip(np.sum(left * right, axis=-1), -1.0, 1.0)
valid = (np.linalg.norm(left, axis=-1) > 0) & (
    np.linalg.norm(right, axis=-1) > 0
)
angle = np.divide(
    (np.arccos(dots) * valid).sum(axis=1),
    valid.sum(axis=1),
    out=np.zeros(edge_grid.shape[0], dtype=np.float32),
    where=valid.sum(axis=1) > 0,
)
```

Negate concave angles so the sign convention remains available to the edge encoder.
Repeat each edge's signed angle over all five curve samples to form channel 7; do not copy channel 7 from the raw 12-channel extractor grid.

- [ ] **Step 4: Implement path and pairwise encodings**

For `spatial_pos`, use unweighted undirected shortest-path hop counts with zero on the diagonal. Reject disconnected graphs because existing BrepMFR treats zero as self/padding.

For `edges_path`:

- allocate `[N,N,6]` filled with `-1`;
- reconstruct the shortest path from SciPy predecessors;
- store zero-based DGL edge IDs;
- verify each reconstructed `(u,v)` exists;
- truncate only beyond six hops.

Implement the paper's Section 3.1.3 proximity descriptors, not the old center-point RBF approximation:

- derive a deterministic RNG seed from `SHA-256(sample_id)` plus ordered face indices;
- for D2 between unordered faces `(i,j)`, draw 512 point pairs with replacement from their trimmed point pools, divide Euclidean distances by the whole scaled-solid bounding-box diagonal, clip to `[0,1]`, and normalize a 64-bin `[0,1]` histogram by 512; copy the result to `(j,i)`;
- for directed A3 `(i,j)`, draw two points from source face `i` and one from target face `j`, calculate the angle at the first source point, divide by `π`, and normalize a 64-bin `[0,1]` histogram by 512;
- set both diagonal 64-vectors to zero.

This follows Section 3.1.3 of the [BrepMFR paper](https://doi.org/10.1016/j.cagd.2024.102318), which specifies 512 random samples and 64-bin marginal distributions. Fixed per-sample seeds make the nominally random descriptor reproducible. Preserve:

```python
metadata["instance"] = torch.from_numpy(labels.instance).to(torch.uint8)
metadata["bottom"] = torch.from_numpy(labels.bottom).to(torch.uint8)
```

- [ ] **Step 5: Run graph-contract tests and verify GREEN**

Run:

```powershell
conda run -n aagmspenv python -m unittest tests.test_mfrinstseg_conversion -v
```

Expected: all graph-contract tests pass.

- [ ] **Step 6: Commit Task 5**

```powershell
git add tools/mfrinstseg/conversion.py tests/test_mfrinstseg_conversion.py
git commit -m "feat: build BrepMFR graphs with instance metadata"
```

---

### Task 6: Add resumable, atomic batch conversion CLI

**Files:**
- Create: `tools/convert_mfrinstseg_to_brepmfr.py`
- Create: `tests/test_mfrinstseg_conversion_cli.py`

**Interfaces:**
- Consumes: `output_root/manifests/<name>.txt`.
- Produces: canonical `output_root/bins/<sample_id>.bin`, `reports/conversion_report.json`, and `reports/failed_samples.txt`.
- Public worker:

```python
convert_one(
    sample_id: str,
    dataset_root: Path,
    bins_root: Path,
    overwrite: bool = False,
) -> ConversionResult
```

- [ ] **Step 1: Write failing save/reload and resume tests**

Use one real sample when locally available. Invoke:

```powershell
conda run -n aagmspenv python tools/convert_mfrinstseg_to_brepmfr.py `
  --dataset-root D:\Code\Datasets\MFRInstSeg `
  --output-root <temporary-output> `
  --manifests smoke `
  --workers 1
```

The test creates `manifests/smoke.txt` containing `20221121_154647_0`. After the first run:

```python
graphs, metadata = load_graphs(str(bin_path))
graph = graphs[0]
self.assertEqual(graph.num_nodes(), 41)
self.assertTrue(torch.equal(graph.ndata["f"], expected_semantic))
self.assertTrue(torch.equal(metadata["instance"], expected_instance))
self.assertTrue(torch.equal(metadata["bottom"], expected_bottom))
```

Record modification time, run the command again without `--overwrite`, and assert the timestamp is unchanged and the report status is `skipped_valid`. Add a manifest containing a nonexistent ID and assert nonzero exit plus that ID in `failed_samples.txt`.

- [ ] **Step 2: Run CLI tests and verify RED**

Run:

```powershell
$env:MFRINSTSEG_ROOT='D:\Code\Datasets\MFRInstSeg'
conda run -n aagmspenv python -m unittest tests.test_mfrinstseg_conversion_cli -v
```

Expected: failure because the conversion CLI does not exist.

- [ ] **Step 3: Implement strict saved-bin validation**

Provide:

```python
REQUIRED_NODE_KEYS = {"x", "y", "z", "l", "a", "f"}
REQUIRED_EDGE_KEYS = {"x", "t", "l", "a", "c"}
REQUIRED_METADATA_KEYS = {
    "edges_path",
    "spatial_pos",
    "d2_distance",
    "angle_distance",
    "instance",
    "bottom",
}

def validate_saved_bin(path: Path, labels: LabelRecord) -> None:
    graphs, metadata = load_graphs(str(path))
    if len(graphs) != 1:
        raise BinValidationError(f"{path}: expected one graph")
    graph = graphs[0]
    if not REQUIRED_NODE_KEYS.issubset(graph.ndata):
        raise BinValidationError(f"{path}: missing node fields")
    if not REQUIRED_EDGE_KEYS.issubset(graph.edata):
        raise BinValidationError(f"{path}: missing edge fields")
    if not REQUIRED_METADATA_KEYS.issubset(metadata):
        raise BinValidationError(f"{path}: missing metadata fields")
    n, e = graph.num_nodes(), graph.num_edges()
    expected_shapes = {
        "node_x": (n, 5, 5, 7),
        "edge_x": (e, 5, 7),
        "edges_path": (n, n, 6),
        "spatial_pos": (n, n),
        "d2_distance": (n, n, 64),
        "angle_distance": (n, n, 64),
        "instance": (n, n),
        "bottom": (n,),
    }
    actual_shapes = {
        "node_x": tuple(graph.ndata["x"].shape),
        "edge_x": tuple(graph.edata["x"].shape),
        **{key: tuple(metadata[key].shape) for key in expected_shapes if key not in {"node_x", "edge_x"}},
    }
    if actual_shapes != expected_shapes:
        raise BinValidationError(f"{path}: shape mismatch {actual_shapes}")
    if not torch.equal(graph.ndata["f"].cpu(), torch.from_numpy(labels.semantic).int()):
        raise BinValidationError(f"{path}: semantic labels differ")
    if not torch.equal(metadata["instance"].cpu(), torch.from_numpy(labels.instance)):
        raise BinValidationError(f"{path}: instance labels differ")
    if not torch.equal(metadata["bottom"].cpu(), torch.from_numpy(labels.bottom)):
        raise BinValidationError(f"{path}: bottom labels differ")
    tensors = list(graph.ndata.values()) + list(graph.edata.values()) + list(metadata.values())
    if any(tensor.is_floating_point() and not torch.isfinite(tensor).all() for tensor in tensors):
        raise BinValidationError(f"{path}: non-finite floating feature")
```

It must reload using `dgl.data.utils.load_graphs`, require one graph, compare all required keys and shapes, compare semantic/instance/bottom tensors exactly, and reject NaN/Inf. Existing files are skipped only after this validation succeeds; an invalid existing file is rebuilt.

- [ ] **Step 4: Implement atomic conversion**

For each sample:

1. resolve `steps/<id>.step` and `labels/<id>.json`;
2. parse labels;
3. extract and build graph;
4. write to a uniquely named temporary file inside `bins/`;
5. reload and validate the temporary file;
6. replace `<id>.bin` with `os.replace`;
7. remove the temporary file in `finally`;
8. return a serializable result with status, node/edge count, elapsed seconds, and error text.

Use `concurrent.futures.ProcessPoolExecutor` only when `workers > 1`; use a normal loop for `workers == 1` so debugging preserves the original traceback.

- [ ] **Step 5: Implement CLI/report behavior**

CLI arguments:

```text
--dataset-root  required Path
--output-root   required Path
--manifests     one or more names, without .txt
--workers       positive int, default 1
--overwrite     flag
```

Load all requested manifests, preserve first occurrence order, and deduplicate IDs before scheduling. Write a sorted conversion report containing requested manifests, unique sample count, converted/skipped/failed counts, per-sample results, package versions, and elapsed time.

If any sample fails:

- write one tab-separated line per failure to `failed_samples.txt`;
- finish other scheduled samples;
- return exit code 1.

If none fail, remove a stale `failed_samples.txt` only via `Path.unlink(missing_ok=True)` after resolving and verifying that it is exactly `<output_root>/reports/failed_samples.txt`.

- [ ] **Step 6: Run CLI tests and verify GREEN**

Run:

```powershell
$env:MFRINSTSEG_ROOT='D:\Code\Datasets\MFRInstSeg'
conda run -n aagmspenv python -m unittest tests.test_mfrinstseg_conversion_cli -v
```

Expected: save/reload, resume, and failed-sample tests pass.

- [ ] **Step 7: Verify compatibility with current BrepMFR loader**

In the CLI test, add:

```python
from data.dataset import CADSynth

loaded = CADSynth.load_one_graph(CADSynth.__new__(CADSynth), bin_path)
self.assertEqual(loaded.node_data.shape[0], 41)
self.assertTrue(torch.equal(loaded.label_feature, expected_semantic.int()))
```

Run:

```powershell
conda run -n aagmspenv python -m unittest tests.test_mfrinstseg_conversion_cli -v
```

Expected: existing loader reads the generated file without changes.

- [ ] **Step 8: Commit Task 6**

```powershell
git add tools/convert_mfrinstseg_to_brepmfr.py tests/test_mfrinstseg_conversion_cli.py
git commit -m "feat: convert MFRInstSeg manifests to bin"
```

---

### Task 7: Document and run the accepted 437/437 workflow

**Files:**
- Create: `docs/mfrinstseg_data_preparation.md`
- Modify: `README.md`

**Interfaces:**
- Documents the two public CLIs and exact experiment workflow.
- Produces real manifests in `D:\Code\Datasets\MFRInstSeg_BrepMFR`.
- Does not start a 874-model conversion without first passing the one-sample smoke test.

- [ ] **Step 1: Write usage documentation**

Document:

1. environment activation;
2. exact subset command;
3. output layout;
4. why 1% applies only to official training data;
5. why `val_small=437` runs every epoch and `val_full` runs once;
6. exact smoke conversion;
7. exact 874-model conversion;
8. later `val_full test` conversion;
9. how to read failure reports and resume;
10. the rule that the test set cannot tune checkpoints, thresholds, or postprocessing.

Add a short “MFRInstSeg data preparation” link to the root README.

- [ ] **Step 2: Run the complete automated suite**

Run:

```powershell
$env:MFRINSTSEG_ROOT='D:\Code\Datasets\MFRInstSeg'
conda run -n aagmspenv python -m unittest discover -s tests -p "test_mfrinstseg_*.py" -v
```

Expected: all tests pass and the two real-sample tests are not skipped.

- [ ] **Step 3: Generate real manifests**

Run:

```powershell
conda run -n aagmspenv python tools/build_mfrinstseg_subset.py `
  --dataset-root D:\Code\Datasets\MFRInstSeg `
  --output-root D:\Code\Datasets\MFRInstSeg_BrepMFR `
  --train-ratio 0.01 `
  --val-small-size 437 `
  --seed 2026
```

Expected:

```text
train_1pct=437
val_small=437
val_full=9375
test=9375
official split intersections=0
```

- [ ] **Step 4: Run one-sample smoke conversion**

Create a temporary `manifests/smoke.txt` under the output root containing the known sample ID, run with `--workers 1`, validate the file, then remove only that temporary smoke manifest. Do not delete the generated bin because it is a valid cache entry.

- [ ] **Step 5: Stop at the compute-cost checkpoint**

Report:

- exact balanced class occurrence and face-count statistics;
- smoke sample node/edge counts;
- test count and elapsed smoke conversion time;
- estimated 874-model conversion time based on the smoke measurement;
- available free disk space and estimated output size.

Do not launch the 874-model conversion until the user sees this checkpoint, because it is a long-running local compute action even though it is already in scope.

- [ ] **Step 6: Commit Task 7**

```powershell
git add README.md docs/mfrinstseg_data_preparation.md
git commit -m "docs: explain MFRInstSeg data preparation"
```

---

## Final Verification Checklist

Before claiming implementation complete:

- [ ] Run the full `unittest` discovery command fresh and record tests, skips, failures, and exit code.
- [ ] Run `git diff --check`.
- [ ] Run `git status --short` and verify only the user's pre-existing `tmp/` remains untracked.
- [ ] Re-read the design spec and map every requirement to Tasks 1–7.
- [ ] Confirm no production function was added without first observing its test fail for the intended reason.
- [ ] Confirm the real manifests contain 437/437/9,375/9,375 unique IDs.
- [ ] Confirm official train/val/test intersections are zero.
- [ ] Confirm the generated smoke `.bin` reloads through DGL and `CADSynth.load_one_graph`.
- [ ] Confirm semantic labels, instance matrix, and bottom labels are exactly equal before and after save/reload.
- [ ] Confirm no command read `aag/graphs.json`.
- [ ] Confirm no test or full validation data influenced balanced training selection.
