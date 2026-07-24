# MFRInstSeg BIN Converter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create one standalone CLI that converts sampled MFRInstSeg STEP models into DGL BIN files accepted by the current BrepMFR baseline.

**Architecture:** The CLI dynamically imports the existing PyOCC/occwl `AAGExtractor`, reads official semantic/instance/bottom labels, constructs the exact node, edge, shortest-path, and RBF tensors expected by `data/dataset.py`, then saves each graph atomically. It processes only IDs listed by the sampled `train.txt` and `val.txt` and writes output manifests containing successful BIN files.

**Tech Stack:** Python 3.8, PyTorch 1.13, DGL 1.x, NumPy, SciPy, PyOCC, occwl.

## Global Constraints

- Create only `tools/mfrinstseg_to_bin.py`.
- Reuse `D:\Code\BrepMFR-g\data_set_line\AAGExtractor.py`; do not copy or refactor it.
- Do not read `aag/graphs.json`.
- Save all fields required by `data/dataset.py`.
- Save `instance` and `bottom` as extra DGL metadata tensors.
- Do not modify the network, loss, loader, original labels, STEP files, or sampled manifests.
- Do not run the converter, PyOCC, or dataset validation in this task, per the user's explicit instruction.
- Keep the implementation compatible with Python 3.8.

---

### Task 1: Implement the sampled STEP-to-BIN converter

**Files:**
- Create: `tools/mfrinstseg_to_bin.py`
- Reference only: `data/dataset.py`
- Reference only: `D:\Code\BrepMFR-g\data_set_line\AAGExtractor.py`
- Reference only: `D:\Code\BrepMFR-g\data_set_line\schema.json`

**Interfaces:**

- Consumes:
  - `dataset_root: Path`
  - `subset_root: Path`
  - `output_root: Path`
  - `extractor_file: Path`
  - `schema_file: Path`
  - split selector `train | val | all`
- Produces:
  - `output_root/bin/<sample_id>.bin`
  - `output_root/train.txt`
  - `output_root/val.txt`
  - `output_root/conversion_failures.json`

- [ ] **Step 1: Implement label and dependency loading**

Define:

```python
def load_extractor(extractor_file: Path):
    """Dynamically import and return AAGExtractor and TopologyUtils."""

def load_schema(schema_file: Path) -> dict:
    """Read and validate face_attributes, edge_attributes, and UV-grid."""

def load_label(label_file: Path) -> LabelData:
    """Return dense semantic, instance, and bottom arrays."""
```

`LabelData` is a frozen dataclass with `sample_id`, `semantic`, `instance`, and `bottom`. Validate contiguous face indices and square instance dimensions.

- [ ] **Step 2: Implement geometric tensor construction**

Define:

```python
def build_graph(
    extraction: dict,
    labels: LabelData,
    schema: dict,
    topology_utils: object,
):
    """Return a DGL graph containing x/y/z/l/a/f and x/t/l/a/c."""
```

Required mappings:

```python
face_type = {
    "Plane": 0,
    "Cylinder": 1,
    "Cone": 2,
    "SphereFaceAttribute": 3,
    "TorusFaceAttribute": 4,
}

edge_type = {
    "StraightEdgeAttribute": 0,
    "CircularEdgeAttribute": 1,
    "EllipticalEdgeAttribute": 2,
    "NonRationalBSplineEdgeAttribute": 3,
    "RationalBSplineEdgeAttribute": 4,
}
```

Construct node `x` as `[N,5,5,7]`. Construct edge `x` as the first six edge-grid channels plus the signed angle. Use convexity values `0=smooth`, `1=concave`, `2=convex`; concave angles are positive and convex angles negative.

- [ ] **Step 3: Implement attention metadata**

Define:

```python
def rbf_encode(values: np.ndarray, bins: int = 64) -> np.ndarray:
    """Encode normalized pairwise values with 64 Gaussian bases."""

def shortest_path_metadata(graph, node_grid: np.ndarray) -> Dict[str, torch.Tensor]:
    """Build spatial_pos, edges_path, d2_distance, angle_distance."""
```

`edges_path` must use zero-based edge IDs and `-1` padding with shape `[N,N,6]`. `spatial_pos` uses zero on the diagonal, one for adjacent faces, and a large value for unreachable pairs. Pairwise center distance and normal angle are RBF encoded to 64 channels.

- [ ] **Step 4: Implement atomic conversion and BIN validation**

Define:

```python
def validate_bin(path: Path, expected_nodes: int) -> None:
    """Reload one BIN and validate required keys and exact leading shapes."""

def convert_one(
    sample_id: str,
    paths: ConversionPaths,
    extractor_class: object,
    topology_utils: object,
    schema: dict,
    overwrite: bool,
) -> str:
    """Return converted or skipped; raise on invalid input/output."""
```

Save to `<sample_id>.bin.tmp`, call `validate_bin`, then use `Path.replace` to publish `<sample_id>.bin`. Metadata includes:

```python
{
    "spatial_pos": ...,
    "edges_path": ...,
    "d2_distance": ...,
    "angle_distance": ...,
    "instance": torch.int8,
    "bottom": torch.int32,
}
```

- [ ] **Step 5: Implement batch CLI and output safety**

Expose:

```text
--dataset-root
--subset-root
--output-root
--extractor-file
--schema-file
--split {train,val,all}
--overwrite
--limit
```

Defaults point to the local MFRInstSeg, sampled subset, and old extractor paths. Reject output roots resolving to the source dataset or subset root. Deduplicate IDs while preserving first occurrence. Catch each model failure, remove its temporary file, continue, and write structured failure objects containing split, sample ID, exception type, and message.

Output `train.txt` and/or `val.txt` with only successful or validated-existing sample IDs.

- [ ] **Step 6: Perform static review without running code**

Review the new file against:

```text
ndata: x y z l a f
edata: x t l a c
metadata: edges_path spatial_pos d2_distance angle_distance instance bottom
```

Check every tensor dtype and shape directly from source. Check that Python syntax uses no features newer than Python 3.8. Do not import, compile, execute, or convert data.

- [ ] **Step 7: Commit the generated code**

```powershell
git add -- tools/mfrinstseg_to_bin.py
git commit -m "feat: add MFRInstSeg BIN converter"
```

Do not include the user's existing `.gitignore` modification.
