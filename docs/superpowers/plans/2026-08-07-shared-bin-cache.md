# MFRInstSeg Shared BIN Cache Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let seed42 1%/5%/10% preparations own independent split manifests while reading and incrementally filling one compatible ParamSurface BIN cache.

**Architecture:** Separate the manifest/report root from the BIN root throughout conversion, preparation, loading, training, and evaluation. A percentage-independent cache fingerprint prevents incompatible geometry from mixing; optional compatible donor roots contribute selected BINs through NTFS hard links, while missing models use the normal converter.

**Tech Stack:** Python 3.8, pathlib, DGL, PyTorch, pytest, PowerShell, NTFS hard links.

## Global Constraints

- Preserve legacy single-root preparation when no shared BIN cache is supplied.
- Keep 1%/5%/10% split manifests and preparation manifests independent.
- Never overwrite an existing valid shared BIN during percentage rebuilds.
- Fail closed on shared-cache fingerprint or BIN-contract mismatch.
- Keep seed 42, split ratios 0.8/0.1/0.1, and the current parameter-surface feature contract unchanged.
- Do not copy or delete donor BIN contents; use hard links only when supported.

---

### Task 1: Separate manifest roots from BIN roots

**Files:**
- Modify: `tools/mfrinstseg_to_bin.py`
- Modify: `data/dataset.py`
- Test: `tests/test_mfrinstseg_to_bin_geometry.py`
- Test: `tests/test_dataset_geometry_fields.py`

**Interfaces:**
- `ConversionPaths(..., bin_cache_root: Optional[Path] = None)` stores BINs under `(bin_cache_root or output_root) / "bin"` while reports remain under `output_root`.
- `CADSynth(..., bin_root: Optional[Path] = None)` reads manifests from `root_dir` and resolves BINs only below `(bin_root or root_dir) / "bin"`.

- [ ] Write converter and dataset tests proving separate roots work and legacy defaults remain unchanged.
- [ ] Run focused tests and confirm failures are caused by missing `bin_cache_root`/`bin_root` arguments.
- [ ] Add the optional fields and route BIN lookup through them.
- [ ] Run `pytest tests/test_mfrinstseg_to_bin_geometry.py tests/test_dataset_geometry_fields.py -q`.
- [ ] Commit as `feat: separate MFRInstSeg manifests and bins`.

### Task 2: Add the safe shared-cache preparation layer

**Files:**
- Modify: `tools/prepare_dataset.py`
- Test: `tests/test_prepare_dataset.py`

**Interfaces:**
- `PreparationConfig.shared_bin_cache_root: Optional[Path] = None`.
- `PreparationConfig.donor_prepared_roots: Tuple[Path, ...] = ()`.
- `PreparedDataset.bin_root: Path` reports the root passed to `CADSynth`.
- `bin_cache_manifest.json` fingerprints converter format, seed, all raw source hashes, extractor/schema hashes, output feature contracts, and point-sampling options, excluding sample percentage and selected IDs.

- [ ] Write failing tests for percentage-independent identity, incompatible-cache rejection, shared output routing, compatible donor hard links, and missing-donor fallback.
- [ ] Run focused tests and verify expected API/behavior failures.
- [ ] Implement shared manifest creation/validation, selected donor linking, shared BIN validation, and converter routing.
- [ ] Preserve the existing recovery/replacement behavior and legacy path when the optional root is absent.
- [ ] Run `pytest tests/test_prepare_dataset.py -q`.
- [ ] Commit as `feat: prepare reusable shared BIN caches`.

### Task 3: Propagate shared BIN roots through training and evaluation

**Files:**
- Modify: `segmentation.py`
- Modify: `tools/test_mfrinstseg.py`
- Modify: `tools/run_mfrinstseg_experiments.ps1`
- Modify: `tools/run_mfrinstseg_test.ps1`
- Test: `tests/test_experiment_configuration.py`
- Test: `tests/test_experiment_commands.py`
- Test: `tests/test_instance_evaluation.py`

**Interfaces:**
- CLI option `--shared_bin_cache_path PATH` is optional.
- Training preparation passes it into `PreparationConfig`, saves the resolved BIN root in run configuration, and supplies it to every `CADSynth` split.
- Evaluation accepts a separate BIN root and preserves legacy single-root operation.

- [ ] Write failing parser, argument-propagation, dataset-construction, and launcher tests.
- [ ] Run focused tests and verify failures identify the missing option propagation.
- [ ] Implement Python and PowerShell propagation with explicit path validation.
- [ ] Run the three focused test modules and PowerShell AST parsing.
- [ ] Commit as `feat: consume shared MFRInstSeg BIN caches`.

### Task 4: Make the seed42 runner reuse one cache

**Files:**
- Modify: `tools/run_mfrinstseg_small_sample_seed42.ps1`
- Test: `tests/test_small_sample_seed42_runner.py`
- Modify: `docs/spe_pcaag_experiments.md`

**Interfaces:**
- `-SharedBinCachePath` defaults to `F:\temp_pan\D\Code\Datasets\MFRInstSeg_BrepMFR_ParamSurface_v4_shared_seed42` only when the caller does not supply a path.
- `-DonorPreparedDatasetPaths` is optional and empty for the current no-BIN state.
- Every supported percentage passes the identical shared root and its own percentage-specific prepared root.

- [ ] Write failing source/behavior tests for one shared root and three independent prepared roots.
- [ ] Run the focused test and confirm the old runner lacks shared-cache propagation.
- [ ] Update the runner and operating documentation, including preparation-only commands before training.
- [ ] Run `pytest tests/test_small_sample_seed42_runner.py tests/test_experiment_commands.py -q`.
- [ ] Commit as `feat: share seed42 small-sample BINs`.

### Task 5: Verify without starting formal conversion or training

**Files:**
- Verify only; no formal dataset/result files are committed.

**Interfaces:**
- A tiny synthetic fixture proves independent manifests can reuse one BIN root.
- A one-model real STEP smoke conversion proves the repository AAG extractor still writes the full parameter-surface contract through the shared path.

- [ ] Run all shared-cache focused tests.
- [ ] Run `python -m pytest -q` in `aagmspenv`.
- [ ] Run Python compilation, PowerShell AST parsing, `git diff --check`, and old-path scanning.
- [ ] Run and clean a one-model shared-root smoke fixture; inspect BIN field shapes and empty failures.
- [ ] Record the exact preparation-only command for the first 10% seed42 conversion.
