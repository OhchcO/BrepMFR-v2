# STEP End-to-End Inference CLI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Python package and thin CLI that convert one unlabeled STEP file into a reusable BrepMFR BIN, run the selected dynamic-affinity checkpoint, cluster feature instances, and atomically write a versioned JSON result.

**Architecture:** The new `brepmfr.inference` package owns production inference boundaries while reusing the repository's proven AAG extraction, geometry encoding, model, collator, and topology-constrained clustering behavior. A content-addressed sidecar validates BIN reuse independently of checkpoint identity, and a single pipeline orchestrator writes success or structured failure JSON. The existing dataset evaluator imports shared production helpers so there is one geometry-requirement and clustering implementation.

**Tech Stack:** Python 3.8.19, PyTorch 1.13.1, PyTorch Lightning 1.9.5-compatible checkpoint loading, DGL, PyTorch Geometric data records, NumPy, pythonocc-core/occwl, pytest, Windows PowerShell.

**Spec:** `docs/superpowers/specs/2026-08-21-step-inference-cli-design.md`

## Global Constraints

- Work only in `D:\Code\BrepMFR\.worktrees\step-inference-cli` on branch `feature/step-inference-cli`; do not alter the dirty `main` checkout.
- Keep all new Python syntax compatible with Python 3.8.
- Use `F:\ProgramData\anaconda3\envs\aagmspenv\python.exe` for tests and real inference.
- Do not import or modify `MFR_Recognition_Project`; UG/NX, TXT output, batch inference, HTTP service, training changes, and bottom-face prediction are out of scope.
- The inference path must not read an MFRInstSeg label JSON or use semantic/instance ground truth.
- The inference BIN stores semantic sentinel `-1` and omits metadata `instance`; existing labeled conversion behavior must remain byte-contract compatible.
- Require a `dynamic` or `dynamic_affinity` checkpoint; class 24 is stock and never becomes a feature instance.
- `face_index` is the 0-based OCC/AAG graph node index and is not an NX Face Tag contract.
- Cache identity includes STEP bytes, schema/extractor/conversion/geometry implementation bytes, seed, point-sampling settings, requested geometry fields, and inference contract version; checkpoint bytes are excluded.
- Use temporary sibling files, validate before replacement, and never accept mismatched BIN/sidecar generations as a cache hit.
- Treat each output directory as single-writer state; concurrent CLI processes must use different output directories.
- Success exits `0`; failure exits nonzero and writes a structured error JSON whenever the output directory is writable.
- Preserve the baseline failure set: `402 passed, 25 failed, 9 warnings` before implementation.

## File Map

**Create**

- `brepmfr/__init__.py` — top-level package marker.
- `brepmfr/inference/__init__.py` — stable public imports for programmatic inference.
- `brepmfr/inference/postprocess.py` — checkpoint geometry requirements, MFRInstSeg names, stable affinity sigmoid, topology-constrained clustering, and face/instance records.
- `brepmfr/inference/cache.py` — SHA-256 identities, sidecar contract, cache decision, and atomic metadata writing.
- `brepmfr/inference/conversion.py` — one unlabeled STEP to validated inference BIN.
- `brepmfr/inference/loading.py` — one inference BIN to a device-ready batch.
- `brepmfr/inference/result.py` — success/error documents and atomic JSON writing.
- `brepmfr/inference/pipeline.py` — checkpoint-to-JSON orchestration and stage-aware errors.
- `tools/infer_step.py` — argument parsing and process exit behavior.
- `tests/test_inference_postprocess.py` — production postprocessing contracts.
- `tests/test_inference_cache.py` — identity and cache invalidation contracts.
- `tests/test_inference_conversion.py` — unlabeled conversion contracts.
- `tests/test_inference_loading.py` — sentinel-aware single-BIN loading.
- `tests/test_inference_result.py` — JSON schema and atomic serialization.
- `tests/test_inference_pipeline.py` — mocked orchestration and CLI behavior.
- `tests/test_inference_golden.py` — opt-in real checkpoint/STEP numerical parity and cache reuse.
- `tests/golden/20221121_154647_0.json` — recorded environment, checkpoint hash, class/instance output, and numerical summaries.

**Modify**

- `tools/test_mfrinstseg.py` — import production `GeometryRequirements`, `geometry_requirements`, and `cluster_instances`; keep evaluator CLI and metrics unchanged.
- `tools/mfrinstseg_to_bin.py` — allow `build_graph(extraction, None, schema, topology_utils)` and write `f == -1` only for inference callers.
- `data/dataset.py` — permit an explicit inference-only unlabeled sentinel without changing default training validation.

---

### Task 1: Move checkpoint requirements and clustering into a production module

**Files:**

- Create: `brepmfr/__init__.py`
- Create: `brepmfr/inference/__init__.py`
- Create: `brepmfr/inference/postprocess.py`
- Create: `tests/test_inference_postprocess.py`
- Modify: `tools/test_mfrinstseg.py:27-69,301-357`
- Test: `tests/test_instance_evaluation.py`

**Interfaces:**

- Produces: `FEATURE_NAMES: Tuple[str, ...]` with exactly 25 MFRInstSeg names.
- Produces: `GeometryRequirements(use_face_uv: bool, require_face_pointcloud: bool, require_parametric_surface: bool, require_cylindrical_pos: bool)`.
- Produces: `configuration_value(configuration: object, name: str, default: object) -> object` for mapping and attribute-backed checkpoint configuration.
- Produces: `geometry_requirements(configuration: object) -> GeometryRequirements`.
- Produces: `cluster_instances(affinity: np.ndarray, semantic: np.ndarray, threshold: float = 0.5, background_class_id: int = 24, adjacency: Optional[np.ndarray] = None) -> np.ndarray`.
- Preserves: importing `cluster_instances` and `geometry_requirements` from `tools.test_mfrinstseg` continues to work through module-level imports.

- [ ] **Step 1: Write failing production-helper tests**

Create `tests/test_inference_postprocess.py` with direct tests for the class table, mapping/dict checkpoint configurations, parametric-surface implication, stable same-class adjacency clustering, background exclusion, and deterministic cluster IDs:

```python
import numpy as np

from brepmfr.inference.postprocess import (
    FEATURE_NAMES,
    cluster_instances,
    geometry_requirements,
)


def test_feature_names_match_mfrinstseg_contract():
    assert len(FEATURE_NAMES) == 25
    assert FEATURE_NAMES[0] == "chamfer"
    assert FEATURE_NAMES[1] == "through_hole"
    assert FEATURE_NAMES[23] == "round"
    assert FEATURE_NAMES[24] == "stock"


def test_geometry_requirements_are_derived_from_checkpoint_mapping():
    requirements = geometry_requirements({
        "use_face_uv": False,
        "use_face_pointcloud": True,
        "face_geometry_encoder": "param_surface",
        "use_cylindrical_pos": True,
    })
    assert requirements.use_face_uv is False
    assert requirements.require_face_pointcloud is True
    assert requirements.require_parametric_surface is True
    assert requirements.require_cylindrical_pos is True


def test_cluster_instances_uses_semantic_topology_and_excludes_stock():
    affinity = np.ones((5, 5), dtype=np.float32)
    semantic = np.array([1, 1, 1, 2, 24], dtype=np.int64)
    adjacency = np.array([
        [0, 1, 0, 0, 0],
        [1, 0, 1, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 0, 0, 1],
        [0, 0, 0, 1, 0],
    ], dtype=bool)
    assert cluster_instances(affinity, semantic, adjacency=adjacency).tolist() == [0, 0, 0, 1, -1]
```

- [ ] **Step 2: Run the new tests and verify import failure**

Run:

```powershell
& 'F:\ProgramData\anaconda3\envs\aagmspenv\python.exe' -m pytest tests/test_inference_postprocess.py -q
```

Expected: FAIL during collection with `ModuleNotFoundError: No module named 'brepmfr'`.

- [ ] **Step 3: Implement the production helper module**

Create the package markers and implement `postprocess.py` by moving, not behaviorally rewriting, the verified requirement and union-find logic from `tools/test_mfrinstseg.py`. Define the names as an immutable tuple:

```python
FEATURE_NAMES = (
    "chamfer", "through_hole", "triangular_passage",
    "rectangular_passage", "6sides_passage",
    "triangular_through_slot", "rectangular_through_slot",
    "circular_through_slot", "rectangular_through_step",
    "2sides_through_step", "slanted_through_step", "Oring",
    "blind_hole", "triangular_pocket", "rectangular_pocket",
    "6sides_pocket", "circular_end_pocket", "rectangular_blind_slot",
    "v_circular_end_blind_slot", "h_circular_end_blind_slot",
    "triangular_blind_step", "circular_blind_step",
    "rectangular_blind_step", "round", "stock",
)
```

Keep `_validate_square_matrix` private inside the production module so the evaluator's unrelated pairwise metrics retain their own validator.

- [ ] **Step 4: Switch the evaluator to production imports**

At the top of `tools/test_mfrinstseg.py`, import:

```python
from brepmfr.inference.postprocess import (
    configuration_value,
    GeometryRequirements,
    cluster_instances,
    geometry_requirements,
)
```

Delete only the local `GeometryRequirements`, `_configuration_value`, `geometry_requirements`, and `cluster_instances` definitions. Update `_model_configuration_value` to call imported `configuration_value(getattr(model, "hparams", {}), name, default)`. Do not change `requires_instance_for_test`, `supports_instance_evaluation`, evaluator metrics, output fields, or CLI arguments.

- [ ] **Step 5: Run focused regression tests**

Run:

```powershell
& 'F:\ProgramData\anaconda3\envs\aagmspenv\python.exe' -m pytest tests/test_inference_postprocess.py tests/test_instance_evaluation.py -q
```

Expected: all tests PASS.

- [ ] **Step 6: Commit the production postprocessing boundary**

```powershell
git add brepmfr tools/test_mfrinstseg.py tests/test_inference_postprocess.py
git commit -m "refactor: share inference postprocessing"
```

---

### Task 2: Add an explicit unlabeled graph contract

**Files:**

- Modify: `tools/mfrinstseg_to_bin.py:396-490`
- Modify: `data/dataset.py:36-69,433-509`
- Create: `tests/test_inference_conversion.py`
- Modify: `tests/test_mfrinstseg_to_bin_geometry.py`

**Interfaces:**

- Changes: `build_graph(extraction: dict, labels: Optional[LabelData], schema: dict, topology_utils: object) -> dgl.DGLGraph`.
- Produces: an unlabeled graph with `graph.ndata["f"] == -1` for every node.
- Changes: `CADSynth.__init__` gains `allow_unlabeled: bool = False`; only `allow_unlabeled=True` accepts all-`-1` feature labels.
- Preserves: labeled callers retain node-count validation and exact semantic tensor values.

- [ ] **Step 1: Write failing graph-contract tests**

In `tests/test_inference_conversion.py`, define a finite two-node extraction and schema, then monkeypatch the two topology-dependent numeric helpers:

```python
def _two_face_contract():
    face_names = list(FACE_TYPE) + ["FaceAreaAttribute"]
    edge_names = list(EDGE_TYPE) + [
        "Concave edge", "Convex edge", "Smooth", "EdgeLengthAttribute",
    ]
    extraction = {
        "graph": {"edges": ([0], [1]), "num_nodes": 2},
        "graph_face_grid": np.zeros((2, 7, 5, 5), dtype=np.float32),
        "graph_face_attr": np.zeros((2, len(face_names)), dtype=np.float32),
        "graph_edge_grid": np.zeros((1, 12, 5), dtype=np.float32),
        "graph_edge_attr": np.zeros((1, len(edge_names)), dtype=np.float32),
    }
    extraction["graph_face_attr"][:, face_names.index("FaceAreaAttribute")] = 1.0
    extraction["graph_edge_attr"][:, edge_names.index("EdgeLengthAttribute")] = 1.0
    return extraction, {
        "face_attributes": face_names,
        "edge_attributes": edge_names,
    }


def test_build_graph_without_labels_uses_unlabeled_sentinel(monkeypatch):
    extraction, schema = _two_face_contract()
    monkeypatch.setattr(converter, "_signed_edge_angles", lambda grid, convexity: np.zeros(1, dtype=np.float32))
    monkeypatch.setattr(converter, "_face_loop_counts", lambda extraction, count, topology: np.ones(count, dtype=np.int32))
    graph = build_graph(extraction, None, schema, object())
    assert graph.ndata["f"].dtype == torch.int32
    assert graph.ndata["f"].tolist() == [-1, -1]


def test_build_graph_with_labels_preserves_training_contract(monkeypatch):
    extraction, schema = _two_face_contract()
    monkeypatch.setattr(converter, "_signed_edge_angles", lambda grid, convexity: np.zeros(1, dtype=np.float32))
    monkeypatch.setattr(converter, "_face_loop_counts", lambda extraction, count, topology: np.ones(count, dtype=np.int32))
    labels = LabelData(
        sample_id="sample",
        semantic=np.array([1, 24], dtype=np.int32),
        instance=np.eye(2, dtype=np.int8),
    )
    graph = build_graph(extraction, labels, schema, object())
    assert graph.ndata["f"].tolist() == [1, 24]
```

Import the converter module as `converter` and its `EDGE_TYPE`, `FACE_TYPE`, `LabelData`, and `build_graph` symbols explicitly. In `tests/test_dataset_geometry_fields.py`, use that file's existing `_write_bin` helper to add a dataset-level test confirming default `CADSynth` rejects negative labels while `allow_unlabeled=True` accepts an all-`-1` vector. The rejection must be `ValueError`, not a diagnostic `print`.

- [ ] **Step 2: Run tests and verify they fail on the old signatures**

Run:

```powershell
& 'F:\ProgramData\anaconda3\envs\aagmspenv\python.exe' -m pytest tests/test_inference_conversion.py -q
```

Expected: FAIL because `build_graph` dereferences `labels.semantic` and `CADSynth` has no `allow_unlabeled` argument.

- [ ] **Step 3: Implement the semantic sentinel**

Change `build_graph` to validate the label count only when `labels is not None`, then assign:

```python
semantic = (
    labels.semantic
    if labels is not None
    else np.full(node_count, -1, dtype=np.int32)
)
graph.ndata["f"] = torch.from_numpy(semantic).to(torch.int32)
```

Keep every other node/edge field unchanged.

- [ ] **Step 4: Make unlabeled loading an explicit dataset opt-in**

Add `allow_unlabeled=False` to `CADSynth.__init__`, store it, and replace the existing label-range `print` with:

```python
labels = pyg_graph.label_feature
if self.allow_unlabeled:
    if not bool((labels == -1).all()):
        raise ValueError("Unlabeled inference BIN must use only semantic sentinel -1")
elif bool(((labels < 0) | (labels >= self.num_class)).any()):
    raise ValueError("Semantic labels are outside the configured class range")
```

Default dataset behavior remains labeled and strict.

- [ ] **Step 5: Run converter and dataset regression tests**

Run:

```powershell
& 'F:\ProgramData\anaconda3\envs\aagmspenv\python.exe' -m pytest tests/test_inference_conversion.py tests/test_mfrinstseg_to_bin_geometry.py tests/test_dataset_geometry_fields.py -q
```

Expected: all tests PASS.

- [ ] **Step 6: Commit the unlabeled contract**

```powershell
git add tools/mfrinstseg_to_bin.py data/dataset.py tests/test_inference_conversion.py tests/test_mfrinstseg_to_bin_geometry.py
git commit -m "feat: support unlabeled inference graphs"
```

---

### Task 3: Implement content-validated BIN cache metadata

**Files:**

- Create: `brepmfr/inference/cache.py`
- Create: `tests/test_inference_cache.py`

**Interfaces:**

- Produces: `INFERENCE_BIN_CONTRACT_VERSION = 2`.
- Produces: `CacheDecision(hit: bool, fingerprint: str, bin_path: Path, metadata_path: Path, reason: str)`.
- Produces: `sha256_file(path: Path) -> str`.
- Produces: `build_cache_identity(step_path: Path, implementation_paths: Sequence[Path], seed: int, point_sampling: Mapping[str, object], geometry_fields: Mapping[str, bool]) -> Dict[str, object]`.
- Produces: `cache_fingerprint(identity: Mapping[str, object]) -> str` using canonical UTF-8 JSON with `sort_keys=True` and compact separators.
- Produces: `inspect_cache(bin_path: Path, metadata_path: Path, identity: Mapping[str, object], validate_bin_file: Callable[[Path], None], force_rebuild: bool = False) -> CacheDecision`.
- Produces: `write_cache_metadata_atomic(path: Path, identity: Mapping[str, object], fingerprint: str, bin_path: Path) -> None`.

- [ ] **Step 1: Write failing cache tests**

Create temp STEP, implementation, BIN, and metadata files. Test exact hit, changed STEP bytes, changed implementation bytes, changed seed, changed geometry fields, `--force-rebuild`, malformed JSON, missing files, and a validator exception:

```python
def test_cache_hits_only_when_identity_and_bin_validation_match(tmp_path):
    step = tmp_path / "part.step"
    implementation = tmp_path / "converter.py"
    bin_path = tmp_path / "input.bin"
    metadata_path = tmp_path / "input.bin.meta.json"
    step.write_bytes(b"STEP-A")
    implementation.write_bytes(b"converter-v1")
    bin_path.write_bytes(b"valid-bin")
    identity = build_cache_identity(
        step, [implementation], 42,
        {"mode": "fixed", "points_per_face": 96},
        {"pointcloud": True, "parametric_surface": True, "cylindrical": True},
    )
    fingerprint = cache_fingerprint(identity)
    write_cache_metadata_atomic(metadata_path, identity, fingerprint, bin_path)
    decision = inspect_cache(bin_path, metadata_path, identity, lambda path: None)
    assert decision.hit is True
    assert decision.fingerprint == fingerprint
```

- [ ] **Step 2: Run tests and verify the module is missing**

Run:

```powershell
& 'F:\ProgramData\anaconda3\envs\aagmspenv\python.exe' -m pytest tests/test_inference_cache.py -q
```

Expected: FAIL during import of `brepmfr.inference.cache`.

- [ ] **Step 3: Implement canonical identity and cache inspection**

The sidecar shape must be exactly:

```python
document = {
    "schema_version": INFERENCE_BIN_CONTRACT_VERSION,
    "fingerprint": fingerprint,
    "identity": dict(identity),
    "bin_sha256": sha256_file(bin_path),
}
```

Hash files in 1 MiB chunks. Resolve paths before reading, but store each implementation as its ordered basename plus content SHA-256 rather than an absolute source path so moving the repository does not invalidate identical bytes. `inspect_cache` returns a miss with a stable reason (`forced`, `missing-bin`, `missing-metadata`, `invalid-metadata`, `identity-mismatch`, or `invalid-bin`) rather than throwing for ordinary cache invalidation. It verifies the current BIN content hash and calls the supplied validator only after the sidecar fingerprint matches, so a BIN from a different sidecar generation cannot be accepted.

- [ ] **Step 4: Implement atomic sidecar writing and read-back**

Write `path.with_suffix(path.suffix + ".tmp")`, flush and close it, parse it back with `json.loads`, compare it to the intended document, then call `Path.replace(path)`. Remove a leftover temp file in `finally`.

- [ ] **Step 5: Run cache tests**

Run:

```powershell
& 'F:\ProgramData\anaconda3\envs\aagmspenv\python.exe' -m pytest tests/test_inference_cache.py -q
```

Expected: all tests PASS.

- [ ] **Step 6: Commit the cache contract**

```powershell
git add brepmfr/inference/cache.py tests/test_inference_cache.py
git commit -m "feat: add inference BIN cache identity"
```

---

### Task 4: Convert one unlabeled STEP into a validated inference BIN

**Files:**

- Create: `brepmfr/inference/conversion.py`
- Modify: `tests/test_inference_conversion.py`

**Interfaces:**

- Produces: `default_conversion_implementation_paths() -> Tuple[Path, ...]` for `brepmfr/inference/conversion.py`, `tools/AAGExtractor.py`, `tools/schema.json`, `tools/mfrinstseg_to_bin.py`, and `tools/geometry_features.py`.
- Produces: `conversion_options_from_requirements(requirements: GeometryRequirements, seed: int) -> ConversionOptions`.
- Produces: `validate_inference_bin(path: Path, requirements: GeometryRequirements, point_sampling: PointSamplingConfig, expected_nodes: Optional[int] = None) -> None`.
- Produces: `convert_step_to_inference_bin(step_path: Path, target_path: Path, requirements: GeometryRequirements, seed: int, extractor_file: Path = DEFAULT_EXTRACTOR_FILE, schema_file: Path = DEFAULT_SCHEMA_FILE) -> int`, returning face count.

- [ ] **Step 1: Write failing orchestration tests with fake extractor dependencies**

Use monkeypatch to replace `load_extractor`, `load_schema`, `build_graph`, `shortest_path_metadata`, geometry extraction, DGL save, and validation. Assert no call to `load_label`, no `instance` metadata, `build_graph` receives `labels=None`, checkpoint requirements select pointcloud/parametric/cylindrical fields, and the final target appears only after temp validation:

```python
def test_conversion_builds_unlabeled_bin_without_instance_truth(tmp_path, monkeypatch):
    step_path = tmp_path / "part.step"
    step_path.write_text("ISO-10303-21;END-ISO-10303-21;", encoding="ascii")
    graph = dgl.graph(([0], [1]), num_nodes=2)
    graph.ndata["x"] = torch.zeros((2, 5, 5, 7), dtype=torch.float32)
    class FakeExtractor:
        def __init__(self, step_file, attribute_schema):
            self.step_file = step_file
        def process(self):
            return {}

    calls = []
    saved_metadata = []
    monkeypatch.setattr(conversion, "load_extractor", lambda path: (FakeExtractor, object()))
    monkeypatch.setattr(conversion, "load_schema", lambda path: {})
    monkeypatch.setattr(conversion, "build_graph", lambda extraction, labels, schema, topology: calls.append(labels) or graph)
    monkeypatch.setattr(conversion, "shortest_path_metadata", lambda graph, node_grid: {})
    monkeypatch.setattr(conversion, "extract_geometry_features", lambda *args, **kwargs: object())
    monkeypatch.setattr(conversion, "attach_geometry_features", lambda *args, **kwargs: None)
    monkeypatch.setattr(conversion, "geometry_metadata", lambda point_sampling: {})
    monkeypatch.setattr(conversion, "surface_geometry_metadata", lambda: {})
    monkeypatch.setattr(
        conversion,
        "save_graphs",
        lambda path, graphs, metadata: (
            Path(path).write_bytes(b"test-bin"),
            saved_metadata.append(dict(metadata)),
        ),
    )
    monkeypatch.setattr(conversion, "validate_inference_bin", lambda path, requirements, point_sampling, expected_nodes=None: None)
    count = conversion.convert_step_to_inference_bin(
        step_path,
        tmp_path / "input.bin",
        GeometryRequirements(False, True, True, True),
        seed=42,
    )
    assert calls == [None]
    assert "instance" not in saved_metadata[0]
    assert count == 2
```

- [ ] **Step 2: Run the conversion tests and verify import failure**

Run:

```powershell
& 'F:\ProgramData\anaconda3\envs\aagmspenv\python.exe' -m pytest tests/test_inference_conversion.py -q
```

Expected: FAIL because `brepmfr.inference.conversion` does not exist.

- [ ] **Step 3: Implement option derivation and inference validation**

Map requirements as follows:

```python
ConversionOptions(
    seed=seed,
    include_face_pointcloud=requirements.require_face_pointcloud,
    include_parametric_surface=requirements.require_parametric_surface,
    include_cylindrical_pos=requirements.require_cylindrical_pos,
)
```

Call existing `validate_bin` with `BinRequirements(require_instance=False, require_pointcloud=requirements.require_face_pointcloud, require_parametric_surface=requirements.require_parametric_surface, require_cylindrical=requirements.require_cylindrical_pos)` and the same point-sampling configuration.

- [ ] **Step 4: Implement atomic STEP conversion**

Follow the proven `convert_one` stages: validate extension/file, dynamically load extractor/schema, process AAG, set `extraction["_extractor"]`, call `build_graph(extraction, None, schema, topology_utils)`, compute shortest-path metadata, extract and attach requested geometry, add point/surface metadata, write the temporary BIN, validate it, and replace the target. Never add `instance` metadata and always remove the temp file in `finally`.

- [ ] **Step 5: Run conversion tests and existing converter regression**

Run:

```powershell
& 'F:\ProgramData\anaconda3\envs\aagmspenv\python.exe' -m pytest tests/test_inference_conversion.py tests/test_mfrinstseg_to_bin_geometry.py tests/test_parallel_conversion.py -q
```

Expected: all tests PASS.

- [ ] **Step 6: Commit single-STEP conversion**

```powershell
git add brepmfr/inference/conversion.py tests/test_inference_conversion.py
git commit -m "feat: convert unlabeled STEP for inference"
```

---

### Task 5: Load one inference BIN into a device-ready batch

**Files:**

- Create: `brepmfr/inference/loading.py`
- Create: `tests/test_inference_loading.py`

**Interfaces:**

- Produces: `SingleBinDataset(CADSynth)` with constructor `(bin_path: Path, requirements: GeometryRequirements, num_classes: int, seed: int)`.
- Produces: `select_device(requested: str) -> torch.device` for `auto`, `cuda`, and `cpu`.
- Produces: `load_inference_batch(bin_path: Path, requirements: GeometryRequirements, num_classes: int, seed: int, device: torch.device) -> Dict[str, object]`.
- Guarantees: returned batch size is one, `label_feature` is all `-1`, `instance_available` is false, and tensor-like values are moved to the selected device.

- [ ] **Step 1: Write failing single-BIN loading tests**

Build a minimal valid serialized graph in the new test module. The helper supplies every core tensor and metadata field consumed by `CADSynth.load_one_graph`:

```python
def write_inference_contract_bin(path, labels):
    graph = dgl.graph(([0, 1], [1, 0]), num_nodes=2)
    graph.ndata["x"] = torch.zeros((2, 5, 5, 7), dtype=torch.float32)
    graph.ndata["y"] = torch.ones(2, dtype=torch.float32)
    graph.ndata["z"] = torch.zeros(2, dtype=torch.int32)
    graph.ndata["l"] = torch.ones(2, dtype=torch.int32)
    graph.ndata["a"] = torch.ones(2, dtype=torch.int32)
    graph.ndata["f"] = torch.tensor(labels, dtype=torch.int32)
    graph.edata["x"] = torch.zeros((2, 5, 7), dtype=torch.float32)
    graph.edata["t"] = torch.zeros(2, dtype=torch.int32)
    graph.edata["l"] = torch.ones(2, dtype=torch.float32)
    graph.edata["a"] = torch.zeros(2, dtype=torch.float32)
    graph.edata["c"] = torch.zeros(2, dtype=torch.int32)
    metadata = {
        "edges_path": torch.zeros((2, 2, 16), dtype=torch.long),
        "spatial_pos": torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
        "d2_distance": torch.zeros((2, 2, 64), dtype=torch.float32),
        "angle_distance": torch.zeros((2, 2, 64), dtype=torch.float32),
    }
    save_graphs(str(path), [graph], metadata)
    return path


def test_single_bin_loader_accepts_only_unlabeled_sentinel(tmp_path):
    bin_path = write_inference_contract_bin(tmp_path / "input.bin", labels=[-1, -1])
    batch = load_inference_batch(
        bin_path,
        GeometryRequirements(True, False, False, False),
        num_classes=25,
        seed=42,
        device=torch.device("cpu"),
    )
    assert batch["label_feature"].tolist() == [-1, -1]
    assert batch["instance_available"].tolist() == [False]
    assert batch["padding_mask"].shape == (1, 2)
```

Also assert mixed labels such as `[-1, 1]` fail and `select_device("cuda")` fails clearly when CUDA is monkeypatched unavailable.

- [ ] **Step 2: Run tests and verify the loading module is missing**

Run:

```powershell
& 'F:\ProgramData\anaconda3\envs\aagmspenv\python.exe' -m pytest tests/test_inference_loading.py -q
```

Expected: FAIL during import of `brepmfr.inference.loading`.

- [ ] **Step 3: Implement the manifest-free dataset adapter**

Subclass `CADSynth` but do not call its manifest-based constructor. Set the exact fields used by `load_one_graph`:

```python
self.split = "inference"
self.num_class = int(num_classes)
self.random_rotate = False
self.require_instance = False
self.require_face_pointcloud = requirements.require_face_pointcloud
self.require_parametric_surface = requirements.require_parametric_surface
self.require_cylindrical_pos = requirements.require_cylindrical_pos
self.allow_unlabeled = True
self.seed = int(seed)
self.file_paths = [Path(bin_path).resolve()]
```

- [ ] **Step 4: Implement device selection and batch movement**

Use the existing `data.collator.collator([sample], multi_hop_max_dist=16, spatial_pos_max=32)`. Move each value exposing `.to(device)` and preserve strings/lists unchanged. Validate `label_feature == -1` and `instance_available == False` after collation.

- [ ] **Step 5: Run loading and dataset tests**

Run:

```powershell
& 'F:\ProgramData\anaconda3\envs\aagmspenv\python.exe' -m pytest tests/test_inference_loading.py tests/test_dataset_geometry_fields.py -q
```

Expected: all tests PASS.

- [ ] **Step 6: Commit the single-BIN loader**

```powershell
git add brepmfr/inference/loading.py tests/test_inference_loading.py
git commit -m "feat: load one inference BIN"
```

---

### Task 6: Build stable face/instance records and JSON documents

**Files:**

- Modify: `brepmfr/inference/postprocess.py`
- Create: `brepmfr/inference/result.py`
- Modify: `tests/test_inference_postprocess.py`
- Create: `tests/test_inference_result.py`

**Interfaces:**

- Produces: `PostprocessResult(faces: List[Dict[str, object]], instances: List[Dict[str, object]], predicted_classes: np.ndarray, face_confidences: np.ndarray, affinity_probabilities: np.ndarray, cluster_ids: np.ndarray)`.
- Produces: `postprocess_predictions(semantic_probabilities: np.ndarray, affinity_logits: np.ndarray, spatial_pos: np.ndarray, threshold: float = 0.5, background_class_id: int = 24) -> PostprocessResult`.
- Produces: `build_success_document(step_path: Path, step_sha256: str, face_count: int, bin_path: Path, metadata_path: Path, cache_hit: bool, cache_fingerprint: str, checkpoint_path: Path, checkpoint_sha256: str, checkpoint_epoch: int, experiment_mode: str, device: str, num_classes: int, affinity_threshold: float, faces: Sequence[Mapping[str, object]], instances: Sequence[Mapping[str, object]], timings_seconds: Mapping[str, float], warnings: Sequence[str]) -> Dict[str, object]` with schema version `1.0` and the exact spec field order.
- Produces: `build_error_document(stage: str, error: BaseException, step_path: Path) -> Dict[str, object]`.
- Produces: `write_json_atomic(path: Path, document: Mapping[str, object]) -> None`.

- [ ] **Step 1: Write failing postprocessing record tests**

Use finite 25-class probabilities and three-face affinity logits. Assert stable sigmoid clipping, direct-neighbor interpretation `spatial_pos == 2`, class-name mapping, stock omission from instances, sorted face indices, deterministic instance IDs, and mean instance confidence:

```python
def test_postprocess_builds_consistent_face_and_instance_records():
    probabilities = np.zeros((3, 25), dtype=np.float32)
    probabilities[0, [0, 1, 24]] = [0.1, 0.8, 0.1]
    probabilities[1, [0, 1, 24]] = [0.1, 0.7, 0.2]
    probabilities[2, [0, 1, 24]] = [0.1, 0.1, 0.8]
    logits = np.array([
        [10.0, 10.0, -10.0],
        [10.0, 10.0, -10.0],
        [-10.0, -10.0, 10.0],
    ], dtype=np.float32)
    spatial_pos = np.array([[1, 2, 3], [2, 1, 3], [3, 3, 1]])
    result = postprocess_predictions(probabilities, logits, spatial_pos)
    assert result.cluster_ids.tolist() == [0, 0, -1]
    assert result.instances[0]["face_indices"] == [0, 1]
    assert result.instances[0]["confidence"] == pytest.approx(0.75)
    assert [face["instance_id"] for face in result.faces] == [0, 0, None]
```

- [ ] **Step 2: Write failing result-document tests**

Assert success field order, resolved POSIX-style absolute paths from `Path.as_posix()`, finite float rejection, allowed error stages, UTF-8 round trip, newline termination, and removal of temporary files after a write exception.

- [ ] **Step 3: Run tests and verify missing interfaces**

Run:

```powershell
& 'F:\ProgramData\anaconda3\envs\aagmspenv\python.exe' -m pytest tests/test_inference_postprocess.py tests/test_inference_result.py -q
```

Expected: FAIL because `PostprocessResult`, `postprocess_predictions`, and `result.py` are absent.

- [ ] **Step 4: Implement stable prediction records**

Validate shapes `(N, 25)`, `(N, N)`, and `(N, N)` plus finite arrays. Compute affinity with:

```python
affinity = 1.0 / (1.0 + np.exp(-np.clip(affinity_logits, -60.0, 60.0)))
```

Call `cluster_instances(affinity, predicted_classes, threshold=threshold, background_class_id=background_class_id, adjacency=spatial_pos == 2)`. Build instances by ascending nonnegative cluster ID, assign every non-stock face exactly one matching `instance_id`, and convert all NumPy scalars to Python `int`/`float`.

- [ ] **Step 5: Implement versioned success/error documents and atomic JSON**

Define `RESULT_SCHEMA_VERSION = "1.0"` and `ALLOWED_ERROR_STAGES`. Recursively reject non-finite floats before serialization. Write with `json.dumps(document, ensure_ascii=False, indent=2) + "\n"`, parse the temp file back, compare to the input mapping, then replace the destination.

- [ ] **Step 6: Run postprocessing and JSON tests**

Run:

```powershell
& 'F:\ProgramData\anaconda3\envs\aagmspenv\python.exe' -m pytest tests/test_inference_postprocess.py tests/test_inference_result.py tests/test_instance_evaluation.py -q
```

Expected: all tests PASS.

- [ ] **Step 7: Commit result contracts**

```powershell
git add brepmfr/inference/postprocess.py brepmfr/inference/result.py tests/test_inference_postprocess.py tests/test_inference_result.py
git commit -m "feat: serialize inference predictions"
```

---

### Task 7: Orchestrate the pipeline and expose the CLI

**Files:**

- Create: `brepmfr/inference/pipeline.py`
- Modify: `brepmfr/inference/__init__.py`
- Create: `tools/infer_step.py`
- Create: `tests/test_inference_pipeline.py`

**Interfaces:**

- Produces: `InferenceRequest(step_path: Path, checkpoint_path: Path, output_dir: Path, device: str = "auto", affinity_threshold: float = 0.5, force_rebuild: bool = False, seed: Optional[int] = None)`.
- Produces: `InferenceOutcome(document: Dict[str, object], result_path: Path, cache_hit: bool, semantic_probabilities: np.ndarray, affinity_probabilities: np.ndarray)`.
- Produces internally: `LoadedCheckpoint(model: BrepSeg, epoch: int, experiment_mode: str, requirements: GeometryRequirements, seed: int, num_classes: int)` and `_load_checkpoint(path: Path) -> LoadedCheckpoint` as the single checkpoint metadata boundary.
- Produces: `InferencePipelineError(stage: str, result_path: Optional[Path])` chained from the original exception.
- Produces: `run_step_inference(request: InferenceRequest) -> InferenceOutcome`.
- Produces: `tools.infer_step.main(argv: Optional[Sequence[str]] = None) -> int`.

- [ ] **Step 1: Write failing mocked pipeline tests**

Monkeypatch checkpoint loading, device selection, cache inspection, conversion, loading, model output, postprocessing, and JSON writing. Test cache miss order, cache hit conversion skip, checkpoint-derived seed, explicit seed override, non-dynamic checkpoint rejection, stage names for each failure, success document fields, and error document writing:

```python
def make_request_and_patch_runtime(tmp_path, monkeypatch):
    step = tmp_path / "part.step"
    checkpoint = tmp_path / "model.ckpt"
    step.write_bytes(b"STEP")
    checkpoint.write_bytes(b"CHECKPOINT")
    request = InferenceRequest(step, checkpoint, tmp_path / "output", device="cpu")

    class FakeModel:
        def to(self, device):
            return self
        def eval(self):
            return self
        def predict_with_affinity(self, batch):
            probabilities = torch.zeros((2, 25), dtype=torch.float32)
            probabilities[:, 1] = 1.0
            affinity_logits = torch.full((1, 2, 2), 10.0, dtype=torch.float32)
            return probabilities, affinity_logits

    loaded = LoadedCheckpoint(
        model=FakeModel(),
        epoch=187,
        experiment_mode="dynamic_affinity",
        requirements=GeometryRequirements(False, False, False, False),
        seed=42,
        num_classes=25,
    )
    monkeypatch.setattr(pipeline, "_load_checkpoint", lambda path: loaded)
    monkeypatch.setattr(pipeline, "select_device", lambda requested: torch.device("cpu"))
    monkeypatch.setattr(pipeline, "default_conversion_implementation_paths", lambda: (checkpoint,))
    monkeypatch.setattr(pipeline, "build_cache_identity", lambda *args, **kwargs: {"identity": "test"})
    monkeypatch.setattr(pipeline, "load_inference_batch", lambda *args, **kwargs: {
        "padding_mask": torch.tensor([[False, False]]),
        "spatial_pos": torch.tensor([[[1, 2], [2, 1]]]),
    })
    monkeypatch.setattr(pipeline, "write_json_atomic", lambda path, document: Path(path).write_text(json.dumps(document), encoding="utf-8"))
    return request


def test_pipeline_reuses_valid_bin_and_skips_conversion(tmp_path, monkeypatch):
    calls = []
    request = make_request_and_patch_runtime(tmp_path, monkeypatch)
    monkeypatch.setattr(pipeline, "inspect_cache", lambda **kwargs: CacheDecision(True, "fp", request.output_dir / "input.bin", request.output_dir / "input.bin.meta.json", "hit"))
    monkeypatch.setattr(pipeline, "convert_step_to_inference_bin", lambda *args, **kwargs: calls.append("convert"))
    outcome = pipeline.run_step_inference(request)
    assert calls == []
    assert outcome.cache_hit is True
    assert outcome.document["cache"]["hit"] is True
```

- [ ] **Step 2: Write failing CLI tests**

Call `main(["--step", str(step), "--checkpoint", str(checkpoint), "--output-dir", str(output_dir)])` directly. Assert required paths, extension, threshold, and device validation; successful summary and exit `0`; pipeline error traceback to stderr and exit `1`; and `--help` parser construction without importing PyOCC at module import time.

- [ ] **Step 3: Run tests and verify missing pipeline/CLI**

Run:

```powershell
& 'F:\ProgramData\anaconda3\envs\aagmspenv\python.exe' -m pytest tests/test_inference_pipeline.py -q
```

Expected: FAIL importing `brepmfr.inference.pipeline` or `tools.infer_step`.

- [ ] **Step 4: Implement checkpoint and cache orchestration**

Implement `_load_checkpoint` by reading the checkpoint mapping with `torch.load(path, map_location="cpu")` for stored epoch and hyperparameters, then calling `BrepSeg.load_from_checkpoint(str(path), map_location="cpu")` for the model. Derive `GeometryRequirements`, `experiment_mode`, seed, and class count through the same mapping/object fallback used by the evaluator, and return `LoadedCheckpoint` with the raw stored epoch value (`187` for the accepted checkpoint and its filename).

In `run_step_inference`, create the output directory first, set a `stage` variable before every boundary, and record `time.perf_counter()` timings. Enforce `experiment_mode in ("dynamic", "dynamic_affinity")`, choose checkpoint seed unless the request overrides it, build the cache identity, and call `inspect_cache` with a validator closure using current requirements.

On a miss, call `convert_step_to_inference_bin`, validate the completed target again through the same validator closure, then call `write_cache_metadata_atomic(metadata_path, identity, decision.fingerprint, decision.bin_path)`. On a hit, skip conversion and metadata writing. Hash the checkpoint for result provenance but do not place it in BIN identity.

- [ ] **Step 5: Implement model inference and result assembly**

Move model and batch to the selected device, call `model.eval()` and `model.predict_with_affinity(batch)` inside `torch.inference_mode()`, require non-`None` affinity logits, slice batch size one to real node count, and pass CPU NumPy arrays plus `spatial_pos` to `postprocess_predictions`. Build/write the success document with timing keys `checkpoint_load`, `conversion`, `inference`, and `total`, plus an initially empty warnings list. Return an `InferenceOutcome` containing arrays for golden parity.

On exception, build and attempt to write the error document with the current stage, then raise `InferencePipelineError` from the original exception. If serialization itself fails, preserve `stage="serialization"` and stderr traceback behavior.

- [ ] **Step 6: Implement the thin CLI**

Add repository root to `sys.path` exactly as other tools do. Use `argparse`, validate files and suffixes before constructing `InferenceRequest`, and expose the documented arguments. Catch `InferencePipelineError`, call `traceback.print_exc(file=sys.stderr)`, print the error result path when available, and return `1`; print result path, face/instance counts, cache status, and return `0` on success.

- [ ] **Step 7: Run mocked pipeline and CLI tests**

Run:

```powershell
& 'F:\ProgramData\anaconda3\envs\aagmspenv\python.exe' -m pytest tests/test_inference_pipeline.py tests/test_inference_cache.py tests/test_inference_loading.py tests/test_inference_result.py -q
& 'F:\ProgramData\anaconda3\envs\aagmspenv\python.exe' tools/infer_step.py --help
```

Expected: tests PASS and help exits `0` without loading OCC or a checkpoint.

- [ ] **Step 8: Commit the end-to-end programmatic and CLI entry points**

```powershell
git add brepmfr/inference/pipeline.py brepmfr/inference/__init__.py tools/infer_step.py tests/test_inference_pipeline.py
git commit -m "feat: add STEP inference CLI"
```

---

### Task 8: Record real golden parity, cache reuse, and final regression evidence

**Files:**

- Create: `tests/test_inference_golden.py`
- Create: `tests/golden/20221121_154647_0.json`
- Modify: `README.md`

**Interfaces:**

- Consumes: `run_step_inference(InferenceRequest) -> InferenceOutcome`.
- Produces: a committed golden record keyed by sample `20221121_154647_0` and checkpoint SHA-256.
- Produces: an environment-gated pytest that skips only when external STEP, prepared BIN, or checkpoint assets are absent.

- [ ] **Step 1: Write the golden parity test before recording expected output**

Use these exact assets:

```python
RAW_STEP = Path(r"D:\Code\Datasets\MFRInstSeg\steps\20221121_154647_0.step")
RAW_LABEL = Path(r"D:\Code\Datasets\MFRInstSeg\labels\20221121_154647_0.json")
PREPARED_BIN = Path(r"D:\Code\Datasets\MFRInstSeg_BrepMFR_ParamSurface_v4_shared_seed42\bin\20221121_154647_0.bin")
CHECKPOINT = Path(r"D:\Code\BrepMFR\results\paper_core_experiments\MFRInstSeg_1pct_spe_instance_positive_param_surface_intrinsic_untrimmed_gated_seed42\0821\092832\best-mIoU-187-0.939778.ckpt")
```

The test must monkeypatch `Path.open` to raise only if the resolved path equals `RAW_LABEL` during `run_step_inference`. Build the reference batch from `PREPARED_BIN` using a normally configured labeled `CADSynth.load_one_graph` call plus the existing collator, call the same checkpoint's `predict_with_affinity`, and compare it to a first `run_step_inference` call on `RAW_STEP`. Load the committed golden document before the comparisons:

```python
np.testing.assert_allclose(actual.semantic_probabilities, reference_semantic, rtol=0.0, atol=1e-5)
np.testing.assert_allclose(actual.affinity_probabilities, reference_affinity, rtol=0.0, atol=1e-5)
assert_prediction_records_close(actual.document["faces"], golden["faces"], atol=1e-5)
assert_prediction_records_close(actual.document["instances"], golden["instances"], atol=1e-5)
assert actual.document["cache"]["hit"] is False
```

`assert_prediction_records_close` must compare IDs, names, face lists, and nullable instance assignments exactly while comparing only `confidence` with `pytest.approx(abs=atol)`. Keep the `Path.open` label guard active while calling the pipeline a second time in the same output directory; assert `cache.hit is True` plus identical faces/instances. Only then restore `Path.open` and load `RAW_LABEL` for posthoc evaluation. Use the existing label loader and instance metric helpers to calculate per-face accuracy, clustering pairwise F1, adjusted Rand index, predicted instance count, and ground-truth instance count; compare them to `golden["posthoc_metrics"]` with `1e-8` absolute tolerance for floats and exact integer counts.

- [ ] **Step 2: Run the real test and verify it fails because the golden record is absent**

Run:

```powershell
& 'F:\ProgramData\anaconda3\envs\aagmspenv\python.exe' -m pytest tests/test_inference_golden.py -q -s
```

Expected: FAIL with a missing `tests/golden/20221121_154647_0.json` assertion after successfully locating the external assets.

- [ ] **Step 3: Generate and inspect the golden record from the reference path**

Record, without copying the dataset label into the repository, by constructing the complete dictionary from computed values:

```python
golden_path = Path(__file__).resolve().parent / "golden" / "20221121_154647_0.json"
golden_path.parent.mkdir(parents=True, exist_ok=True)
reference_affinity = 1.0 / (
    1.0 + np.exp(-np.clip(reference_affinity_logits, -60.0, 60.0))
)
reference_postprocess = postprocess_predictions(
    reference_semantic,
    reference_affinity_logits,
    reference_spatial_pos,
    threshold=0.5,
    background_class_id=24,
)
evaluation = evaluate_instance_graph(
    sample_id="20221121_154647_0",
    affinity_logits=reference_affinity_logits,
    predicted_semantic=reference_semantic.argmax(axis=1),
    ground_truth_semantic=label.semantic,
    target_affinity=label.instance,
    spatial_pos=reference_spatial_pos,
    threshold=0.5,
    background_class_id=24,
)
posthoc_metrics = {
    "per_face_accuracy": float(
        np.mean(reference_semantic.argmax(axis=1) == label.semantic)
    ),
    "clustering_pairwise_f1": float(
        evaluation["clustering"]["pairwise"]["f1"]
    ),
    "adjusted_rand_index": float(
        evaluation["clustering"]["adjusted_rand_index"]
    ),
    "predicted_instance_count": int(
        evaluation["clustering"]["predicted_instance_count"]
    ),
    "ground_truth_instance_count": int(
        evaluation["clustering"]["ground_truth_instance_count"]
    ),
}
record = {
    "schema_version": "1.0",
    "sample_id": "20221121_154647_0",
    "face_count": 41,
    "checkpoint_filename": CHECKPOINT.name,
    "checkpoint_sha256": sha256_file(CHECKPOINT),
    "python": platform.python_version(),
    "torch": torch.__version__,
    "pytorch_lightning": pytorch_lightning.__version__,
    "semantic_probability_sha256": hashlib.sha256(
        np.ascontiguousarray(reference_semantic, dtype=np.float32).tobytes()
    ).hexdigest(),
    "affinity_probability_sha256": hashlib.sha256(
        np.ascontiguousarray(reference_affinity, dtype=np.float32).tobytes()
    ).hexdigest(),
    "faces": reference_postprocess.faces,
    "instances": reference_postprocess.instances,
    "posthoc_metrics": posthoc_metrics,
}
golden_path.write_text(
    json.dumps(record, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
```

Gate this write block behind `UPDATE_STEP_INFERENCE_GOLDEN=1`; without that environment variable, the test only reads and validates the committed file. Generate it with `$env:UPDATE_STEP_INFERENCE_GOLDEN='1'` for one pytest invocation, then remove the variable. Reread the generated file, assert exact dictionary equality with `record`, assert both hashes contain 64 lowercase hexadecimal characters, and inspect the face/instance records before staging it. The committed file must contain no absolute user path and no label ground truth; the posthoc block contains aggregate metrics only.

- [ ] **Step 4: Run the real CLI twice**

Use a task-specific output directory under the worktree:

```powershell
& 'F:\ProgramData\anaconda3\envs\aagmspenv\python.exe' tools/infer_step.py `
  --step 'D:\Code\Datasets\MFRInstSeg\steps\20221121_154647_0.step' `
  --checkpoint 'D:\Code\BrepMFR\results\paper_core_experiments\MFRInstSeg_1pct_spe_instance_positive_param_surface_intrinsic_untrimmed_gated_seed42\0821\092832\best-mIoU-187-0.939778.ckpt' `
  --output-dir '.tmp_step_inference_golden' `
  --device cuda
```

Run the identical command a second time. Expected: first reports cache miss, second reports cache hit, both exit `0`, and `result.json` parses successfully.

- [ ] **Step 5: Run golden and focused suites**

Run:

```powershell
& 'F:\ProgramData\anaconda3\envs\aagmspenv\python.exe' -m pytest tests/test_inference_golden.py tests/test_inference_postprocess.py tests/test_inference_cache.py tests/test_inference_conversion.py tests/test_inference_loading.py tests/test_inference_result.py tests/test_inference_pipeline.py tests/test_instance_evaluation.py tests/test_mfrinstseg_to_bin_geometry.py tests/test_dataset_geometry_fields.py -q
```

Expected: all focused tests PASS.

- [ ] **Step 6: Document the CLI**

Add a concise `STEP inference` section to `README.md` with the command, output files, cache reuse rule, `face_index` limitation, and the statement that the pipeline does not read dataset labels.

- [ ] **Step 7: Run final verification**

Run:

```powershell
& 'F:\ProgramData\anaconda3\envs\aagmspenv\python.exe' -m py_compile brepmfr/inference/cache.py brepmfr/inference/conversion.py brepmfr/inference/loading.py brepmfr/inference/postprocess.py brepmfr/inference/result.py brepmfr/inference/pipeline.py tools/infer_step.py
& 'F:\ProgramData\anaconda3\envs\aagmspenv\python.exe' -m pytest -q
git diff --check
git status --short
```

Expected: Python compilation succeeds; the full suite has no more than the same 25 baseline failures and no changed failure names; diff check is clean; status contains only intended task files before commit.

- [ ] **Step 8: Commit golden evidence and documentation**

```powershell
git add tests/test_inference_golden.py tests/golden/20221121_154647_0.json README.md
git commit -m "test: verify STEP inference golden parity"
```

- [ ] **Step 9: Inspect final history and worktree state**

Run:

```powershell
git log --oneline --decorate -10
git status --short
```

Expected: design plus eight focused implementation commits are visible and the worktree is clean.
