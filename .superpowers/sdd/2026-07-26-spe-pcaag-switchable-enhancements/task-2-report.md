# Task 2 Report: Deterministic Raw-Dataset Preparation and Cache

## Files

- Added `tools/prepare_dataset.py`: immutable preparation configuration and
  output records; strict raw-model discovery; official/generated model-level
  splits; balanced train/validation selection; streamed SHA-256 request
  fingerprinting; atomic manifest publication; fail-closed cache reuse; and
  one-shot, lazy converter orchestration.
- Added `tests/test_prepare_dataset.py`: temporary raw STEP/JSON fixtures and
  tests for deterministic disjoint splits, train/validation-only sampling,
  partial manifest rejection, stem mismatch rejection, cache mismatch,
  cache reuse without converter loading, one-shot conversion, and persisted
  conversion failure reporting.
- Updated `tools/sample_mfrinstseg_1pct.py`: exposed import-safe reusable
  sample-percent validation, class-statistics, and balanced-ID selection while
  preserving the existing CLI flow.

## Decisions

- Raw discovery accepts only `steps/<id>.step` and `labels/<id>.json`, and
  requires their stem sets to be equal exactly.
- Existing `train.txt`, `val.txt`, and `test.txt` are used only when all three
  are present, model-disjoint, and collectively contain every discovered raw
  model once. Otherwise all three splits are deterministically generated from
  a seed derived with `derive_seed(seed, "model_split")`.
- Train and validation subsets use the sampler's instance-balanced selector
  with a derived per-split seed; test is never sampled down.
- The request fingerprint hashes every raw STEP and label using streamed
  SHA-256 content, includes conversion parameters, and fixes the output
  contract to both face point clouds and cylindrical positions. It deliberately
  excludes current model-consumption switches so all ablation modes share one
  cache.
- A matching complete cache is reused without importing the converter. A
  changed or incomplete cache fails closed unless `rebuild=True`; rebuilding
  publishes only selected manifests and lets the converter atomically replace
  its selected targets, without directory deletion.
- `_load_converter()` is the explicit lazy boundary for Task 3. Task 3 must
  provide `convert_selected_samples(...)` and return either a mapping or an
  object with `successful_ids`, `failed_ids`, and optional `conversion`.

## Static Checks Performed

- `git diff --check` completed with no whitespace-error diagnostics.
- `rg` confirmed that `tools.prepare_dataset` uses `derive_seed`, keeps the
  face-point-cloud/cylindrical contract fixed, and imports
  `tools.mfrinstseg_to_bin` only inside `_load_converter()`.
- Repository status and diff statistics were inspected. Git emitted only the
  pre-existing line-ending conversion warning for the edited sampler file.

## Intentionally Not Run

- `pytest tests/test_prepare_dataset.py -v`
- Any Python module import/execution, generated-code execution, converter run,
  training, validation, inference, or benchmark.

## Concerns / Follow-up

- Task 3 has not yet supplied `convert_selected_samples`; this task's lazy
  adapter and tests define the required result boundary, but conversion cannot
  execute until that API exists.
- Cache reuse verifies the selected manifests and that every selected BIN is a
  non-empty file. Task 3 should extend the shared BIN validator to check the
  requested point-cloud and cylindrical fields during cache reuse without
  importing PyOCC at module import time.
