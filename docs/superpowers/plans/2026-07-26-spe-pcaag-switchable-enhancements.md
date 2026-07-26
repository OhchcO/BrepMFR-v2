# SPE-PCAAG Switchable Enhancements Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend BrepMFR so a training command can deterministically prepare STEP/JSON data, encode per-face point clouds and full-part cylindrical positions, and run the complete SPE-PCAAG plus instance-relation ablation matrix through independent switches.

**Architecture:** A preparation layer runs once before Lightning is created, produces a fingerprinted BIN cache, and exposes the existing `train.txt/val.txt/test.txt` contract. Geometry extraction writes fixed-shape DGL node tensors; the data loader validates requested fields; lightweight PointNet and cylindrical MLP branches add residual embeddings to the existing graph-node feature before both the Transformer and dynamic instance-bias predictor. The current baseline/oracle/dynamic/dynamic-affinity paths remain intact, and PowerShell aliases only translate experiment names into explicit switches.

**Tech Stack:** Python 3, PyTorch 1.13, DGL 1.1, PyTorch Geometric, PyTorch Lightning 1.9, NumPy, pythonocc-core/PyOCC, occwl, PowerShell, pytest.

## Global Constraints

- Only generate or modify source, test, documentation, and launcher files in this implementation; do not execute BIN conversion, training, validation, test inference, GPU benchmarks, or generated pytest commands.
- Raw input is exactly `steps/<sample_id>.step|.stp` plus `labels/<sample_id>.json`; labels require only semantic `seg` and instance `inst`; `bottom` is neither read nor generated.
- If all official manifests exist, reuse their model-level split; if none exist, create deterministic 80/10/10 model-level splits; partial manifest sets are an error.
- Balance-sample train and validation independently using configurable `sample_percent` (default `0.01`); keep test complete.
- One configurable `--seed` (default `2026`) controls split, sampling, conversion, Python, NumPy, PyTorch, CUDA, DGL, DataLoader workers/generator, augmentation, initialization, training, validation, and test.
- Per-purpose seeds use `SHA256(base_seed + purpose + sample_id) mod 2^32`; never use Python `hash()`.
- The default point strategy is area-adaptive: `round8(clip(64 * (area / median_area)^0.5, 32, 96))`; retain fixed-64 as a switch.
- Candidate UV-grid resolutions are `20,32,48,64,80`; select in 3D using deterministic FPS and deterministically repeat the FPS order only when 80×80 still provides too few valid candidates.
- New BIN node fields are `face_points float32 [N,96,6]`, `face_point_mask bool [N,96]`, `face_point_count int32 [N]`, and `cylindrical_pos float32 [N,4]`.
- New BIN metadata fields are `point_sampling_mode int8 [1]`, `point_sampling_config float32 [4]`, and `point_candidate_grid int32 [1]`.
- Cylindrical encoding is `[r_norm,sin(theta),cos(theta),z_norm]`, based on PyOCC surface centroids and full-model ranges about the global Z axis.
- PointNet is `6→64→128→256`, masked max pooling, then `256→hidden`; use Linear, LayerNorm, and GELU, not BatchNorm.
- Cylindrical MLP is `4→64→hidden`; both geometry branches use independently switchable residual additions and default scale `1.0`.
- Enhanced node features feed both the graph Transformer and the dynamic instance relation predictor.
- Old BIN is valid only when both geometry switches are off; geometry switches must fail immediately on missing fields.
- Automatic preparation always materializes both point-cloud and cylindrical fields, independent of the current model switches; model switches control consumption, not cache contents.
- Formal new ablations all use the same regenerated BIN directory; baseline and instance-only runs ignore the extra fields.
- Automatic preparation runs before construction of `pl.Trainer`, never from `Dataset.__getitem__`, a DataLoader worker, a training step, or an epoch hook.
- Any conversion failure stops training; configuration or input fingerprint mismatch stops with an instruction to choose a new prepared directory or explicitly rebuild it.
- Do not overwrite `D:\Code\Datasets\MFRInstSeg_BrepMFR`; examples use `D:\Code\Datasets\MFRInstSeg_BrepMFR_SPEPCAAG`.
- With cylindrical position enabled, arbitrary-axis rotation is invalid. Formal ablations use `random_rotate=False` for every row so augmentation does not become a confounding variable.
- The post-experiment framework rewrite is outside this plan.

---

## File Structure

**Create**

- `utils/__init__.py` — project utility package marker.
- `utils/reproducibility.py` — stable seed derivation and process/DataLoader seeding.
- `tools/geometry_features.py` — pure point-count, FPS, packing, centroid, and cylindrical calculations with lazy OCC imports.
- `tools/prepare_dataset.py` — raw-data discovery, manifest resolution, balanced subsampling, cache fingerprinting, and converter orchestration.
- `models/modules/pointcloud_encoder.py` — masked per-face PointNet.
- `models/modules/cylindrical_encoder.py` — cylindrical-position MLP.
- `tests/test_reproducibility.py` — deterministic seed contract.
- `tests/test_geometry_features.py` — sampling and cylindrical math.
- `tests/test_prepare_dataset.py` — split, balance, fingerprint, and fail-closed cache behavior.
- `tests/test_mfrinstseg_to_bin_geometry.py` — BIN field/metadata validation without loading a real STEP.
- `tests/test_geometry_encoders.py` — encoder shapes, mask handling, and finite output.
- `tests/test_dataset_geometry_fields.py` — strict/compatible BIN loading and collation.
- `tests/test_experiment_configuration.py` — alias-to-switch and CLI preparation ordering contracts.
- `docs/spepcaag_experiments.md` — user-run preparation, training, and test commands.

**Modify**

- `tools/mfrinstseg_to_bin.py` — reusable conversion API, geometry extraction, new fields, feature-aware validation, and CLI options.
- `data/dataset.py` — geometry field loading/validation, stable IDs, DataLoader seeding, and rotation guard.
- `data/collator.py` — concatenate new per-node tensors into the batch.
- `models/modules/layers/brep_encoder_layer.py` — residual geometry fusion in `GraphNodeFeature`.
- `models/modules/brep_encoder.py` — accept geometry switches/tensors and pass enhanced features to all downstream consumers.
- `models/brepseg_model.py` — save geometry hyperparameters and pass batch fields to the encoder.
- `segmentation.py` — import-safe parser/main, one-time preparation before Trainer, deterministic setup, and new CLI switches.
- `tools/test_mfrinstseg.py` — checkpoint-driven geometry requirements and unified deterministic setup.
- `tools/run_mfrinstseg_experiments.ps1` — seven ablation aliases plus preparation/geometry parameters.
- `tools/run_mfrinstseg_test.ps1` — use the prepared full test split and forward the unified seed.

---

### Task 1: Unified Reproducibility Contract

**Files:**
- Create: `utils/__init__.py`
- Create: `utils/reproducibility.py`
- Create: `tests/test_reproducibility.py`

**Interfaces:**
- Produces: `derive_seed(base_seed: int, purpose: str, sample_id: str = "") -> int`
- Produces: `seed_everything(base_seed: int, deterministic: bool = True) -> None`
- Produces: `seed_worker(worker_id: int) -> None`
- Produces: `make_torch_generator(base_seed: int, purpose: str) -> torch.Generator`

- [ ] **Step 1: Write the seed-contract tests**

```python
from utils.reproducibility import derive_seed, make_torch_generator


def test_derive_seed_is_stable_and_purpose_specific():
    first = derive_seed(2026, "split", "model_001")
    assert first == derive_seed(2026, "split", "model_001")
    assert 0 <= first < 2**32
    assert first != derive_seed(2026, "point_sampling", "model_001")


def test_torch_generator_replays_sequence():
    import torch
    left = torch.rand(4, generator=make_torch_generator(2026, "train_loader"))
    right = torch.rand(4, generator=make_torch_generator(2026, "train_loader"))
    assert torch.equal(left, right)
```

- [ ] **Step 2: Record the user-run failure command**

Run later: `pytest tests/test_reproducibility.py -v`

Expected before implementation: import failure for `utils.reproducibility`.

- [ ] **Step 3: Implement stable seed helpers**

```python
def derive_seed(base_seed: int, purpose: str, sample_id: str = "") -> int:
    payload = "{}\0{}\0{}".format(int(base_seed), purpose, sample_id).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)
```

`seed_everything` must seed `random`, NumPy, CPU/CUDA PyTorch, and DGL; set cuDNN benchmark false; enable deterministic algorithms when supported; warn explicitly if the installed PyTorch cannot guarantee an operator. `make_torch_generator` must manually seed from `derive_seed`.

- [ ] **Step 4: Record the user-run pass command**

Run later: `pytest tests/test_reproducibility.py -v`

Expected after implementation: two passing tests.

- [ ] **Step 5: Commit**

```powershell
git add utils tests/test_reproducibility.py
git commit -m "feat: centralize reproducible seed handling"
```

---

### Task 2: Deterministic Raw-Dataset Preparation and Cache

**Files:**
- Create: `tools/prepare_dataset.py`
- Create: `tests/test_prepare_dataset.py`
- Modify: `tools/sample_mfrinstseg_1pct.py`

**Interfaces:**
- Consumes: `derive_seed(base_seed: int, purpose: str, sample_id: str = "") -> int` from Task 1.
- Produces: immutable `PreparationConfig` and `PreparedDataset` dataclasses.
- Produces: `discover_raw_samples(raw_root: Path) -> dict[str, RawSample]`
- Produces: `resolve_model_splits(raw_root: Path, sample_ids: Sequence[str], split_ratios: Tuple[float, float, float], seed: int) -> Dict[str, Sequence[str]]`
- Produces: `select_split_subsets(raw_root: Path, splits: Mapping[str, Sequence[str]], sample_percent: float, seed: int) -> Dict[str, Sequence[str]]`
- Produces: `build_preparation_manifest(config: PreparationConfig, selected_ids: Mapping[str, Sequence[str]]) -> dict`
- Produces: `ensure_prepared_dataset(config: PreparationConfig) -> PreparedDataset`
- Consumes later: `convert_selected_samples(sample_ids_by_split, paths, extractor_file, schema_file, options, overwrite)` from Task 3; import it inside `ensure_prepared_dataset` so importing the preparation module does not require OCC.

- [ ] **Step 1: Write split and cache tests using temporary STEP placeholders and real JSON labels**

```python
def test_generated_split_is_model_level_disjoint_and_reproducible(tmp_path):
    raw = make_raw_fixture(tmp_path, sample_count=20)
    ids = tuple(sorted(discover_raw_samples(raw)))
    left = resolve_model_splits(raw, ids, (0.8, 0.1, 0.1), seed=2026)
    right = resolve_model_splits(raw, ids, (0.8, 0.1, 0.1), seed=2026)
    assert left == right
    assert not (set(left["train"]) & set(left["val"]))
    assert not (set(left["train"]) & set(left["test"]))
    assert not (set(left["val"]) & set(left["test"]))


def test_sample_percent_applies_to_train_and_val_but_not_test(tmp_path):
    raw = make_raw_fixture(tmp_path, sample_count=100)
    splits = resolve_model_splits(raw, tuple(sorted(discover_raw_samples(raw))), (0.8, 0.1, 0.1), 2026)
    selected = select_split_subsets(raw, splits, sample_percent=0.25, seed=2026)
    assert len(selected["train"]) < len(splits["train"])
    assert len(selected["val"]) < len(splits["val"])
    assert selected["test"] == splits["test"]


def test_partial_official_manifests_are_rejected(tmp_path):
    raw = make_raw_fixture(tmp_path, sample_count=10)
    (raw / "train.txt").write_text("sample_000\n", encoding="utf-8")
    with pytest.raises(ValueError, match="all three"):
        resolve_model_splits(raw, tuple(sorted(discover_raw_samples(raw))), (0.8, 0.1, 0.1), 2026)


def test_cache_fingerprint_mismatch_fails_closed(tmp_path, monkeypatch):
    config = make_preparation_config(tmp_path, sample_percent=0.01)
    write_incompatible_manifest(config.prepared_root, sample_percent=0.05)
    with pytest.raises(PreparationMismatchError, match="new prepared directory|rebuild"):
        ensure_prepared_dataset(config)
```

- [ ] **Step 2: Record the user-run failure command**

Run later: `pytest tests/test_prepare_dataset.py -v`

Expected before implementation: import failure for `tools.prepare_dataset`.

- [ ] **Step 3: Extract reusable balanced-selection functions from the existing sampler**

Keep the existing CLI behavior, but expose class-statistics and balanced-selection functions without parsing arguments or writing files. Validate `0 < sample_percent <= 1`, exactly three positive ratios summing to `1.0`, unique non-empty sample IDs, exact STEP/label stem equality, and complete official-manifest membership.

- [ ] **Step 4: Implement preparation manifest and cache decisions**

`PreparationConfig` must include raw/prepared roots, extractor/schema paths, seed, split ratios, sample percentage, point mode, fixed/min/max counts, exponent, candidate grids, the fixed output-feature contract (`face_pointcloud=True`, `cylindrical_pos=True`), and explicit `rebuild`. It must not include the current model-consumption switches because baseline through full share one prepared cache. The manifest must include:

```json
{
  "format_version": 1,
  "converter_format_version": 2,
  "base_seed": 2026,
  "split_ratios": [0.8, 0.1, 0.1],
  "sample_percent": 0.01,
  "source_files": {
    "sample_001": {
      "step": {"path": "steps/sample_001.step", "sha256": "<64 lowercase hexadecimal characters>"},
      "label": {"path": "labels/sample_001.json", "sha256": "<64 lowercase hexadecimal characters>"}
    }
  },
  "selected_ids": {"train": [], "val": [], "test": []},
  "conversion": {},
  "successful_ids": {"train": [], "val": [], "test": []},
  "failed_ids": []
}
```

Fingerprint every input by streamed SHA-256 content, not modification time. If a complete manifest and all listed BIN files validate against the requested features, reuse them. If configuration/input differs and `rebuild=False`, raise `PreparationMismatchError`. If `rebuild=True`, overwrite only manifest-listed target BIN/manifests through atomic file replacement; do not recursively delete directories.

- [ ] **Step 5: Implement one-shot conversion orchestration**

Write selected manifests atomically, call `convert_selected_samples` once, always write `conversion_failures.json`, and raise `PreparationError` whenever `failed_ids` is non-empty or a selected manifest has missing output. Return a `PreparedDataset(root, train_ids, val_ids, test_ids, reused)` only after validation succeeds.

- [ ] **Step 6: Record the user-run pass command**

Run later: `pytest tests/test_prepare_dataset.py -v`

Expected after implementation: split, balancing, manifest, reuse, mismatch, and failure-propagation tests pass without importing PyOCC.

- [ ] **Step 7: Commit**

```powershell
git add tools/prepare_dataset.py tools/sample_mfrinstseg_1pct.py tests/test_prepare_dataset.py
git commit -m "feat: add deterministic raw dataset preparation"
```

---

### Task 3: Area-Adaptive Per-Face Geometry Extraction and BIN Contract

**Files:**
- Create: `tools/geometry_features.py`
- Create: `tests/test_geometry_features.py`
- Create: `tests/test_mfrinstseg_to_bin_geometry.py`
- Modify: `tools/mfrinstseg_to_bin.py`

**Interfaces:**
- Consumes: `derive_seed(base_seed: int, purpose: str, sample_id: str = "") -> int` from Task 1.
- Produces: `PointSamplingConfig(mode: str = "area_adaptive", fixed_count: int = 64, min_count: int = 32, max_count: int = 96, area_exponent: float = 0.5, candidate_grids: Tuple[int, int, int, int, int] = (20, 32, 48, 64, 80))`
- Produces: `GeometryFeatureBatch(face_points: np.ndarray, face_point_mask: np.ndarray, face_point_count: np.ndarray, cylindrical_pos: np.ndarray)`
- Produces: `adaptive_point_count(area: float, median_area: float, config: PointSamplingConfig) -> int`
- Produces: `farthest_point_indices(points_xyz: np.ndarray, count: int) -> np.ndarray`
- Produces: `cylindrical_positions(surface_centroids: np.ndarray, eps: float = 1e-12) -> np.ndarray`
- Produces: `extract_geometry_features(extraction: Mapping, config: PointSamplingConfig, base_seed: int, sample_id: str) -> GeometryFeatureBatch`
- Produces from converter: `BinRequirements(require_instance: bool, require_pointcloud: bool, require_cylindrical: bool)`
- Produces from converter: `ConversionOptions(seed: int, point_sampling: PointSamplingConfig, include_face_pointcloud: bool, include_cylindrical_pos: bool)`
- Produces from converter: `convert_selected_samples(sample_ids_by_split: Mapping[str, Sequence[str]], paths: ConversionPaths, extractor_file: Path, schema_file: Path, options: ConversionOptions, overwrite: bool) -> ConversionReport`

- [ ] **Step 1: Write pure NumPy geometry tests**

```python
def test_area_adaptive_counts_are_clipped_and_rounded_to_eight():
    config = PointSamplingConfig()
    assert adaptive_point_count(0.01, 1.0, config) == 32
    assert adaptive_point_count(1.0, 1.0, config) == 64
    assert adaptive_point_count(100.0, 1.0, config) == 96


def test_fps_is_deterministic_and_uses_centroid_farthest_first():
    points = np.array([[0., 0., 0.], [1., 0., 0.], [4., 0., 0.]], dtype=np.float32)
    indices = farthest_point_indices(points, 2)
    assert indices.tolist() == [2, 0]
    assert np.array_equal(indices, farthest_point_indices(points, 2))


def test_cylindrical_position_uses_full_input_ranges():
    centers = np.array([[1., 0., 0.], [0., 2., 10.]], dtype=np.float64)
    encoded = cylindrical_positions(centers)
    assert np.allclose(encoded[:, 0], [0., 1.])
    assert np.allclose(encoded[:, 3], [0., 1.])
    assert np.allclose(encoded[0, 1:3], [0., 1.])
```

- [ ] **Step 2: Write mocked BIN-contract tests**

Construct a two-node DGL graph without STEP parsing, attach `GeometryFeatureBatch`, save it, and assert exact dtypes/shapes plus feature-aware validation. Include failures for NaN, incorrect mask/count, missing requested fields, and invalid adaptive counts.

- [ ] **Step 3: Record the user-run failure commands**

Run later:

```powershell
pytest tests/test_geometry_features.py -v
pytest tests/test_mfrinstseg_to_bin_geometry.py -v
```

Expected before implementation: missing geometry module and converter interfaces.

- [ ] **Step 4: Implement point counts, deterministic FPS, and fixed-shape packing**

Use nearest-multiple-of-eight rounding with deterministic half-up behavior:

```python
rounded = int(math.floor(clipped / 8.0 + 0.5)) * 8
return min(config.max_count, max(config.min_count, rounded))
```

FPS first chooses the candidate farthest from candidate centroid, then repeatedly chooses the largest minimum squared 3D distance; `np.argmax` provides lowest-index tie breaking. If candidate count remains below the target at grid 80, repeat the selected FPS index order until the target is filled. Pad remaining positions through index 95 with zero and mark them false.

- [ ] **Step 5: Implement lazy OCC geometry access**

Import `occwl.uvgrid`, `OCC.Core.BRepGProp`, and `OCC.Core.GProp` inside extraction functions. For each face:

1. Compute true surface area and surface centroid from `BRepGProp.SurfaceProperties`.
2. Evaluate candidate points and normals on grids `20,32,48,64,80`.
3. Remove entries with invalid visibility masks, non-finite XYZ/normal values, or near-zero normal norm.
4. Normalize valid normals to unit length.
5. Stop grid escalation as soon as candidates meet the target.
6. Run deterministic 3D FPS and pack `[XYZ,Normal]`.

Compute area median, all target counts, all centroids, and cylindrical min/max ranges over the complete STEP extraction before any downstream subgraph crop.

- [ ] **Step 6: Extend converter construction, validation, and metadata**

Change core-field validation from exact key-set equality to required-subset validation. Attach the four node fields and three metadata fields with exact shapes/dtypes. Validate semantic labels and instance matrix as today; never access `bottom`. Add CLI options:

```text
--seed 2026
--include-face-pointcloud
--include-cylindrical-pos
--point-sampling-mode fixed|area_adaptive
--fixed-point-count 64
--min-point-count 32
--max-point-count 96
--area-sampling-exponent 0.5
--candidate-grids 20 32 48 64 80
```

The standalone CLI and automatic preparation must call the same `convert_selected_samples` function. Successful output is written as `<sample>.bin.tmp` and atomically replaced only after `validate_bin` succeeds.

- [ ] **Step 7: Record the user-run pass commands**

Run later:

```powershell
pytest tests/test_geometry_features.py -v
pytest tests/test_mfrinstseg_to_bin_geometry.py -v
```

Expected after implementation: all pure calculations and mocked BIN validation pass; these commands do not convert real STEP data.

- [ ] **Step 8: Commit**

```powershell
git add tools/geometry_features.py tools/mfrinstseg_to_bin.py tests/test_geometry_features.py tests/test_mfrinstseg_to_bin_geometry.py
git commit -m "feat: write SPE-PCAAG geometry fields to BIN"
```

---

### Task 4: Strict Geometry-Aware Dataset Loading and Collation

**Files:**
- Modify: `data/dataset.py`
- Modify: `data/collator.py`
- Create: `tests/test_dataset_geometry_fields.py`

**Interfaces:**
- Consumes BIN node fields from Task 3.
- Extends `CADSynth.__init__(root_dir, split="train", random_rotate=False, num_class=25, require_instance=False, require_face_pointcloud=False, require_cylindrical_pos=False, seed=2026)`.
- Extends `CADSynth.get_dataloader(batch_size, shuffle=True, num_workers=0, seed_purpose=None)`.
- Produces batch keys `face_points`, `face_point_mask`, `face_point_count`, and `cylindrical_pos`, concatenated over real faces without graph-level padding.

- [ ] **Step 1: Write old/new BIN compatibility tests**

```python
def test_old_bin_loads_when_geometry_is_disabled(old_bin_dataset):
    item = CADSynth(old_bin_dataset, "train", require_face_pointcloud=False, require_cylindrical_pos=False)[0]
    assert item.face_points is None
    assert item.cylindrical_pos is None


def test_missing_point_fields_fail_when_enabled(old_bin_dataset):
    dataset = CADSynth(old_bin_dataset, "train", require_face_pointcloud=True)
    with pytest.raises(KeyError, match="face_points.*sample"):
        dataset[0]


def test_new_fields_are_concatenated_by_collator(new_bin_dataset):
    dataset = CADSynth(new_bin_dataset, "train", require_face_pointcloud=True, require_cylindrical_pos=True)
    batch = dataset._collate([dataset[0], dataset[1]])
    assert batch["face_points"].shape == (batch["node_data"].shape[0], 96, 6)
    assert batch["cylindrical_pos"].shape == (batch["node_data"].shape[0], 4)
```

- [ ] **Step 2: Record the user-run failure command**

Run later: `pytest tests/test_dataset_geometry_fields.py -v`

Expected before implementation: constructor and batch-key failures.

- [ ] **Step 3: Implement strict conditional loading**

When a feature is enabled, verify all related keys, node count, shape, dtype-compatible values, finite values, and mask/count consistency; errors include the absolute BIN path and missing/invalid key. When disabled, leave the corresponding item attributes as `None`; do not allocate zero substitutes. Change sample ID handling from `int(basename.split("_")[-1])` to the full stem string, and return `data_ids` as an immutable tuple from the collator.

- [ ] **Step 4: Add deterministic DataLoader construction and rotation guard**

Pass `generator=make_torch_generator(self.seed, purpose)` and `worker_init_fn=seed_worker`. Default purposes are `train_loader`, `val_loader`, and `test_loader`. Preserve existing rotation only for runs without cylindrical positions; raise `ValueError` if `random_rotate=True` and `require_cylindrical_pos=True`. If point clouds are enabled without cylindrical positions and rotation is requested, rotate point XYZ and normals with the same matrix as node/edge UV grids.

- [ ] **Step 5: Extend collation**

Only concatenate an optional field when every item provides it; reject a mixed batch. Keep per-graph instance matrices padded as today. Add no graph-batch/max-node padding to the new per-face tensors because `GraphNodeFeature` consumes the concatenated face order.

- [ ] **Step 6: Record the user-run pass command**

Run later: `pytest tests/test_dataset_geometry_fields.py -v`

Expected after implementation: old compatibility, strict errors, seeded loading, and new-field collation pass.

- [ ] **Step 7: Commit**

```powershell
git add data/dataset.py data/collator.py tests/test_dataset_geometry_fields.py
git commit -m "feat: load and collate optional face geometry"
```

---

### Task 5: Point-Cloud and Cylindrical Encoders

**Files:**
- Create: `models/modules/pointcloud_encoder.py`
- Create: `models/modules/cylindrical_encoder.py`
- Create: `tests/test_geometry_encoders.py`

**Interfaces:**
- Produces: `PointCloudFaceEncoder(hidden_dim: int, eps: float = 1e-8)`
- `PointCloudFaceEncoder.forward(face_points: Tensor[F,96,6], face_point_mask: BoolTensor[F,96]) -> Tensor[F,H]`
- Produces: `CylindricalPositionEncoder(hidden_dim: int)`
- `CylindricalPositionEncoder.forward(cylindrical_pos: Tensor[F,4]) -> Tensor[F,H]`

- [ ] **Step 1: Write encoder contract tests**

```python
def test_point_encoder_ignores_padded_values():
    encoder = PointCloudFaceEncoder(hidden_dim=128).eval()
    points = torch.randn(2, 96, 6)
    mask = torch.zeros(2, 96, dtype=torch.bool)
    mask[:, :32] = True
    changed = points.clone()
    changed[:, 32:] = 1e6
    with torch.no_grad():
        assert torch.allclose(encoder(points, mask), encoder(changed, mask))


def test_encoders_return_finite_hidden_vectors():
    points, mask = valid_face_point_batch(face_count=3)
    assert torch.isfinite(PointCloudFaceEncoder(128)(points, mask)).all()
    assert CylindricalPositionEncoder(128)(torch.randn(3, 4)).shape == (3, 128)
```

- [ ] **Step 2: Record the user-run failure command**

Run later: `pytest tests/test_geometry_encoders.py -v`

Expected before implementation: missing encoder modules.

- [ ] **Step 3: Implement masked PointNet**

For each face, calculate XYZ center and maximum radius using valid points only. Divide centered XYZ by `max(radius, eps)` and concatenate the original unit normals. Apply pointwise `Linear→LayerNorm→GELU` blocks `6→64→128→256`; fill invalid rows with `torch.finfo(dtype).min` immediately before max pooling; reject a face with zero valid points; project pooled vectors through `Linear(256,H)→LayerNorm(H)→GELU`.

- [ ] **Step 4: Implement cylindrical MLP**

Use `Linear(4,64)→GELU→Linear(64,H)`. Validate rank, last dimension, and finite values in both encoders so malformed data fails near the source.

- [ ] **Step 5: Record the user-run pass command**

Run later: `pytest tests/test_geometry_encoders.py -v`

Expected after implementation: mask invariance, output shape, gradient, and finite-value tests pass.

- [ ] **Step 6: Commit**

```powershell
git add models/modules/pointcloud_encoder.py models/modules/cylindrical_encoder.py tests/test_geometry_encoders.py
git commit -m "feat: add lightweight SPE-PCAAG encoders"
```

---

### Task 6: Residual Geometry Fusion Into Transformer and Instance Bias

**Files:**
- Modify: `models/modules/layers/brep_encoder_layer.py`
- Modify: `models/modules/brep_encoder.py`
- Modify: `models/brepseg_model.py`
- Extend: `tests/test_geometry_encoders.py`

**Interfaces:**
- Consumes encoders from Task 5 and batch fields from Task 4.
- Extends `GraphNodeFeature.__init__(num_heads, num_degree, hidden_dim, n_layers, use_face_pointcloud=False, use_cylindrical_pos=False, pointcloud_scale=1.0, cylindrical_scale=1.0)`.
- Extends `GraphNodeFeature.forward(x, face_area, face_type, face_loop, face_degree, padding_mask, face_points=None, face_point_mask=None, cylindrical_pos=None)`.
- Extends `BrepEncoder` and `BrepSeg` constructors with the same four geometry arguments.

- [ ] **Step 1: Write integration tests**

```python
def test_all_geometry_switches_off_preserve_original_parameter_names():
    baseline = make_encoder(use_face_pointcloud=False, use_cylindrical_pos=False)
    assert not any(name.startswith("pointcloud_encoder") for name, _ in baseline.named_parameters())
    assert not any(name.startswith("cylindrical_encoder") for name, _ in baseline.named_parameters())


def test_zero_geometry_scales_match_base_node_feature():
    torch.manual_seed(2026)
    base = make_graph_node_feature(False, False)
    torch.manual_seed(2026)
    enhanced = make_graph_node_feature(True, True, pointcloud_scale=0.0, cylindrical_scale=0.0)
    enhanced.load_state_dict(base.state_dict(), strict=False)
    assert torch.allclose(run_base(base), run_enhanced(enhanced), atol=0.0, rtol=0.0)


def test_enhanced_feature_is_used_by_dynamic_relation_predictor():
    model = make_dynamic_model(use_face_pointcloud=True, use_cylindrical_pos=True)
    relation_input = capture_relation_predictor_input(model, make_geometry_batch())
    transformer_input = capture_first_transformer_input(model, make_geometry_batch())
    assert torch.equal(relation_input, transformer_input)
```

- [ ] **Step 2: Record the user-run failure command**

Run later: `pytest tests/test_geometry_encoders.py -v`

Expected before integration: missing constructor/forward arguments.

- [ ] **Step 3: Fuse optional residual branches**

Keep the current surface/area/type/loop/degree concatenation unchanged, then compute:

```python
node_feature = base_node_feature
if self.pointcloud_encoder is not None:
    node_feature = node_feature + self.pointcloud_scale * self.pointcloud_encoder(face_points, face_point_mask)
if self.cylindrical_encoder is not None:
    node_feature = node_feature + self.cylindrical_scale * self.cylindrical_encoder(cylindrical_pos)
```

Do not instantiate disabled branches. Construct existing base modules first, preserve CPU/CUDA RNG states while constructing optional branches, and restore them before constructing later Transformer/instance modules. This keeps downstream initialization identical for equal base seeds across ablations.

- [ ] **Step 4: Thread fields through model boundaries**

Save all four geometry options in Lightning hyperparameters/checkpoints. `BrepSeg.forward` passes optional batch tensors to `BrepEncoder`; `BrepEncoder` builds the enhanced node tensor once and passes that same tensor to both graph layers and the dynamic relation predictor. Oracle GT bias still uses labels only according to its existing experiment-mode rules; inference in dynamic/full modes never requests instance labels.

- [ ] **Step 5: Record the user-run pass command**

Run later: `pytest tests/test_geometry_encoders.py -v`

Expected after integration: disabled-branch compatibility, zero-scale equality, checkpoint hyperparameters, and shared enhanced-feature tests pass.

- [ ] **Step 6: Commit**

```powershell
git add models/modules/layers/brep_encoder_layer.py models/modules/brep_encoder.py models/brepseg_model.py tests/test_geometry_encoders.py
git commit -m "feat: fuse SPE-PCAAG features before graph attention"
```

---

### Task 7: Training Entry, Preparation Ordering, and Ablation Aliases

**Files:**
- Modify: `segmentation.py`
- Modify: `tools/run_mfrinstseg_experiments.ps1`
- Create: `tests/test_experiment_configuration.py`

**Interfaces:**
- Consumes `PreparationConfig`/`ensure_prepared_dataset` from Task 2.
- Produces `build_parser() -> argparse.ArgumentParser`
- Produces `resolve_experiment_alias(alias: str) -> ExperimentSwitches`
- Produces `prepare_training_data(args: argparse.Namespace) -> Path`
- Produces `main(argv: Sequence[str] | None = None) -> int`

- [ ] **Step 1: Write import, alias, and ordering tests**

```python
def test_importing_segmentation_does_not_parse_process_arguments():
    module = importlib.import_module("segmentation")
    assert callable(module.build_parser)


@pytest.mark.parametrize(("alias", "mode", "point", "cyl", "bias_scale"), [
    ("baseline", "baseline", False, False, 1.0),
    ("cyl", "baseline", False, True, 1.0),
    ("point", "baseline", True, False, 1.0),
    ("spe_pcaag", "baseline", True, True, 1.0),
    ("instance", "dynamic_affinity", False, False, 1.0),
    ("affinity_only", "dynamic_affinity", False, False, 0.0),
    ("full", "dynamic_affinity", True, True, 1.0),
])
def test_alias_mapping(alias, mode, point, cyl, bias_scale):
    switches = resolve_experiment_alias(alias)
    assert (switches.experiment_mode, switches.use_face_pointcloud,
            switches.use_cylindrical_pos, switches.instance_bias_scale) == (
            mode, point, cyl, bias_scale)


def test_preparation_occurs_before_trainer_construction(monkeypatch):
    events = []
    monkeypatch.setattr(segmentation, "prepare_training_data", lambda args: events.append("prepare") or args.prepared_dataset_path)
    monkeypatch.setattr(segmentation.pl, "Trainer", lambda **kwargs: events.append("trainer") or FakeTrainer())
    segmentation.main(minimal_cli_args())
    assert events[:2] == ["prepare", "trainer"]
```

- [ ] **Step 2: Record the user-run failure command**

Run later: `pytest tests/test_experiment_configuration.py -v`

Expected before implementation: import-time argument parsing or missing configuration helpers.

- [ ] **Step 3: Refactor training entry without changing baseline math**

Move parser construction and all execution under `main`. Call `seed_everything(args.seed)` first, resolve or prepare the dataset second, instantiate datasets/model/callbacks third, and create `pl.Trainer` fourth. Preserve `--dataset_path` as the old-BIN path when `--auto_prepare` is absent. Automatic preparation always requests both new geometry fields even for the baseline alias; the two model switches only decide whether `CADSynth` validates/loads them and whether the model uses them. Add:

```text
--raw_dataset_path
--prepared_dataset_path
--auto_prepare
--rebuild_prepared_data
--sample_percent 0.01
--split_ratios 0.8 0.1 0.1
--extractor_file
--schema_file
--use_face_pointcloud
--use_cylindrical_pos
--pointcloud_scale 1.0
--cylindrical_scale 1.0
--point_sampling_mode area_adaptive
--fixed_point_count 64
--min_point_count 32
--max_point_count 96
--area_sampling_exponent 0.5
--candidate_grids 20 32 48 64 80
--random_rotate
```

Reject `--random_rotate` with cylindrical positions. Pass `require_face_pointcloud` and `require_cylindrical_pos` into all train/val datasets. Keep formal launcher runs rotation-free.

- [ ] **Step 4: Make run identity and logs auditable**

Log raw/prepared paths, preparation reuse status, base seed, selected split counts, sampling configuration, geometry switches/scales, instance mode/scales/loss weight, and rotation setting to stdout and existing `log.txt`. Save the same values as checkpoint hyperparameters. Configure checkpoint filenames to contain epoch and monitored validation loss while keeping `monitor="eval_loss"` and a single unambiguous best checkpoint.

- [ ] **Step 5: Extend PowerShell launcher**

Use:

```powershell
[ValidateSet("baseline","cyl","point","spe_pcaag","instance","affinity_only","full")]
[string]$Mode = "baseline"
```

Add raw/prepared/extractor/schema, preparation, sampling, geometry-scale, and seed parameters. `-All` executes the seven aliases in table order. Build explicit Python argument arrays; do not implement network behavior in PowerShell. Default the prepared path to `D:\Code\Datasets\MFRInstSeg_BrepMFR_SPEPCAAG`. Every alias must use that same path, seed, sample percentage, output-feature contract, and rotation setting. The first alias may build the cache; subsequent aliases must reuse it without a model-switch-dependent fingerprint change.

- [ ] **Step 6: Record the user-run pass command**

Run later: `pytest tests/test_experiment_configuration.py -v`

Expected after implementation: import safety, exact alias matrix, one-time preparation ordering, and invalid-option tests pass without constructing a real Trainer.

- [ ] **Step 7: Commit**

```powershell
git add segmentation.py tools/run_mfrinstseg_experiments.ps1 tests/test_experiment_configuration.py
git commit -m "feat: expose automatic preparation and ablation switches"
```

---

### Task 8: Checkpoint-Driven Test Evaluation and User Commands

**Files:**
- Modify: `tools/test_mfrinstseg.py`
- Modify: `tools/run_mfrinstseg_test.ps1`
- Create: `docs/spepcaag_experiments.md`
- Extend: `tests/test_experiment_configuration.py`

**Interfaces:**
- Consumes geometry hyperparameters saved by Task 6.
- Produces evaluator output fields `seed`, `use_face_pointcloud`, `use_cylindrical_pos`, `pointcloud_scale`, `cylindrical_scale`, and `prepared_dataset_path` in `test_metrics.json` and `test_log.txt`.

- [ ] **Step 1: Add evaluator configuration tests**

```python
def test_test_dataset_requirements_come_from_checkpoint_hparams():
    hparams = {"use_face_pointcloud": True, "use_cylindrical_pos": False}
    requirements = test_mfrinstseg.geometry_requirements(hparams)
    assert requirements.require_face_pointcloud is True
    assert requirements.require_cylindrical_pos is False


def test_dynamic_full_test_does_not_require_instance_labels():
    assert test_mfrinstseg.requires_instance_for_test("dynamic_affinity") is False
    assert test_mfrinstseg.requires_instance_for_test("oracle") is True
```

- [ ] **Step 2: Record the user-run failure command**

Run later: `pytest tests/test_experiment_configuration.py -v`

Expected before evaluator changes: missing checkpoint-requirement helpers.

- [ ] **Step 3: Update evaluation**

Replace ad-hoc PyTorch seeding with `seed_everything(args.seed)`. Read geometry switches/scales from the loaded checkpoint model, construct `CADSynth` with strict requirements, and use a seeded non-shuffling DataLoader. Formal test uses the prepared directory's complete `test.txt`; do not resample test. Keep face accuracy, mIoU, macro-F1, per-class metrics, and confusion matrix, and append full configuration provenance to both result files.

- [ ] **Step 4: Simplify the PowerShell test launcher**

Accept `-PreparedDataset`, `-Checkpoint`, `-Seed`, `-BatchSize`, and `-NumWorkers`. By default, do not reconvert test in the evaluator launcher because training auto-preparation already created and fingerprinted it. If preparation is requested, invoke `tools/prepare_dataset.py` with the same raw/prepared/configuration arguments before evaluation; never run the old test-only conversion path with a different configuration.

- [ ] **Step 5: Write exact user-run documentation**

Document these commands without executing them:

```powershell
# Prepare and train one full model; preparation is cached.
powershell.exe -ExecutionPolicy Bypass -File `
  "D:\Code\BrepMFR\tools\run_mfrinstseg_experiments.ps1" `
  -Mode full `
  -RawDataset "D:\Code\Datasets\MFRInstSeg" `
  -PreparedDataset "D:\Code\Datasets\MFRInstSeg_BrepMFR_SPEPCAAG" `
  -SamplePercent 0.01 `
  -Seed 2026

# Run all seven ablations against the same prepared cache.
powershell.exe -ExecutionPolicy Bypass -File `
  "D:\Code\BrepMFR\tools\run_mfrinstseg_experiments.ps1" `
  -All `
  -RawDataset "D:\Code\Datasets\MFRInstSeg" `
  -PreparedDataset "D:\Code\Datasets\MFRInstSeg_BrepMFR_SPEPCAAG" `
  -SamplePercent 0.01 `
  -Seed 2026

# Evaluate a selected checkpoint on the full prepared test.txt split.
powershell.exe -ExecutionPolicy Bypass -File `
  "D:\Code\BrepMFR\tools\run_mfrinstseg_test.ps1" `
  -Checkpoint "D:\Code\BrepMFR\results\<experiment>\<run>\best.ckpt" `
  -PreparedDataset "D:\Code\Datasets\MFRInstSeg_BrepMFR_SPEPCAAG" `
  -Seed 2026
```

Also document the fixed-64 comparison, `sample_percent=1.0`, cache mismatch resolution, expected `conversion_failures.json`, and why the formal matrix must not mix old/new BIN directories.

- [ ] **Step 6: Record the user-run static test suite**

Run later:

```powershell
pytest tests/test_reproducibility.py tests/test_prepare_dataset.py tests/test_geometry_features.py tests/test_mfrinstseg_to_bin_geometry.py tests/test_dataset_geometry_fields.py tests/test_geometry_encoders.py tests/test_experiment_configuration.py -v
```

Expected after implementation: all tests pass without converting a real STEP or starting training.

- [ ] **Step 7: Perform code-only verification**

Without executing Python, conversion, training, evaluation, or tests:

- Search for hard-coded seeds other than CLI defaults and test fixtures.
- Search for `hash(` in all seed paths.
- Confirm no PyOCC import occurs at `segmentation.py`, dataset, or model import time.
- Confirm every geometry CLI option appears in the preparation fingerprint.
- Confirm all seven aliases map to the specified matrix.
- Confirm `bottom` is absent from required label keys.
- Confirm old BIN is accepted only with both geometry switches disabled.
- Confirm new formal baseline and full commands point to the same prepared directory.

- [ ] **Step 8: Commit**

```powershell
git add tools/test_mfrinstseg.py tools/run_mfrinstseg_test.ps1 docs/spepcaag_experiments.md tests/test_experiment_configuration.py
git commit -m "docs: add reproducible SPE-PCAAG experiment workflow"
```

---

## Completion Criteria

- Training can start from raw `steps/` and `labels/` with a single command, while STEP conversion occurs once before Trainer construction and is reused only under an exact fingerprint match.
- A single seed is traceable through every stochastic component and every output record.
- Newly generated BIN contains valid adaptive/fixed point clouds, masks/counts, full-part cylindrical positions, and sampling metadata.
- Baseline and old instance experiments still read old BIN only when geometry is off.
- Point, cylindrical, SPE-PCAAG, and full modes fail clearly on old or malformed BIN.
- Geometry embeddings enter both semantic graph attention and the learned instance-relation bias path.
- `baseline`, `cyl`, `point`, `spe_pcaag`, `instance`, `affinity_only`, and `full` are independent, auditable parameter combinations.
- The user has exact Windows PowerShell commands for preparation, training, and full-test evaluation.
- No conversion, training, validation, test inference, benchmark, or generated test command is executed by the implementation agent.
