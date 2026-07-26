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
- The Task 3 validator implementation is still pending. Its required lazy,
  OCC-import-safe `validate_prepared_bin` boundary is now invoked by this task
  for cache reuse and conversion publication, and Task 3 must implement the
  complete field/shape/dtype/finite/mask/sampling checks it receives.

## Fix Round 1

### Review findings addressed

- Cache reuse and post-conversion publication now use the lazy Task 3 boundary
  `validate_prepared_bin(path, requirements, point_sampling)`. The required
  `BinRequirements` request instance, point-cloud, and cylindrical fields;
  Task 3's validator contract is explicitly OCC-import-safe and must reject
  unreadable/corrupt BINs, missing required fields, invalid shapes/dtypes,
  non-finite values, inconsistent masks/counts, and incompatible sampling
  metadata. Contract doubles use a structured feature artifact rather than
  treating non-empty bytes as valid.
- Extractor and schema paths are independently streamed through SHA-256 and
  recorded under `converter_sources`, which participates in the cache request
  fingerprint.
- Discovery now supports case-insensitive `.step` and `.stp` suffixes and
  rejects duplicate STEP or label stems.
- The lazy converter boundary now constructs Task 3's published
  `PointSamplingConfig`, `ConversionOptions`, and `BinRequirements` types.
  Automatic preparation consistently supplies one prepared root for selected
  manifests and BINs; Task 3 must keep its old source/output path guard
  CLI-only for this direct selected-ID API.
- Tiny generated datasets use deterministic largest-remainder allocation and,
  for three or more models, transfer models as necessary so every positive
  split receives one. Empty split sequences are valid only where unavoidable;
  blank/duplicate sample IDs remain invalid.
- Full-percent train/validation subsets return their original official order.
  Fractional balancing now reduces the objective to classes actually present
  in that split instead of rejecting globally absent classes.
- Malformed, legacy, and incomplete cache manifests normalize to a mismatch:
  they fail closed normally and take the scoped atomic rebuild route with
  `rebuild=True`. No recursive deletion is used.
- Missing or feature-invalid converter outputs become structured failures in
  both `conversion_failures.json` and `preparation_manifest.json` before
  `PreparationError` is raised.

### Added focused tests

- Complete official manifest ordering/membership, `.stp` suffix handling and
  duplicate stems, tiny-N counts, sparse fractional balancing, ratio bounds,
  extractor/schema content invalidation, invalid cached BINs, typed converter
  options/path contract, malformed/legacy rebuild without deletion, and
  persisted missing-output failures.

### Static checks and intentionally unrun commands

- `git diff --check` was run after the changes; no whitespace errors were
  reported.
- Python/pytest, generated code, conversion, training, validation, inference,
  and benchmarks remain intentionally unrun under the user constraint.
