# MFRInstSeg Conversion-Failure Replacement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deterministically replace unconvertible Train/Validation models while preserving subset size, class balance, official Test membership, existing valid BINs, and a complete audit trail.

**Architecture:** Add one incremental balance selector beside the existing sampler, then make `prepare_dataset.py` distinguish the immutable initial request from the final effective selection.  A recovery coordinator handles both new conversion failures and an existing matching failed manifest, converts only replacement candidates, and atomically publishes final manifests plus `conversion_exclusions.json`.

**Tech Stack:** Python 3.8, NumPy, pathlib, JSON, pytest, existing MFRInstSeg converter/preparation APIs.

## Global Constraints

- Work only on branch `tmp/param-surface-instance-combination`; do not merge into `main`.
- Do not delete or rename raw STEP/JSON files.
- Replace failures only in Train or Validation; any Test/global/malformed failure remains fatal.
- Keep the original requested Train/Validation sizes and the complete official Test set.
- Use the existing 24-class feature-instance counts and balance objective.
- Derive every replacement tie-break seed from the configured base seed.
- Preserve complete legacy cache reuse and failure-free conversion behaviour.
- Reuse the 14,685 valid BINs in the existing seed42 10% cache.
- Do not run formal raw-dataset conversion, training, or evaluation.
- The separate shared-BIN-cache design is outside this implementation plan.

---

### Task 1: Deterministic Incremental Replacement Selector

**Files:**
- Create: `tests/test_balanced_replacement.py`
- Modify: `tools/sample_mfrinstseg_1pct.py`

**Interfaces:**
- Consumes: existing `SampleRecord`, `load_record`, `class_statistics`, `balance_objective`, `DEFAULT_CLASSES`, and `derive_seed`.
- Produces:

```python
@dataclass(frozen=True)
class ReplacementSelection:
    sample_id: str
    balance_before: Objective
    balance_after: Objective
```

The exact callable signature is
`select_balanced_replacement_id(dataset_root: Path, successful_ids:
Sequence[str], candidate_ids: Sequence[str], base_seed: int, split_name: str,
attempt: int) -> ReplacementSelection`.

- The selector chooses exactly one candidate and performs no filesystem writes.

- [ ] **Step 1: Write failing tests for balance improvement and deterministic ties**

Create fixtures whose `inst` matrices yield hand-checked instance-count rows.
Assert that a successful set missing class 1 chooses the candidate containing
class 1, not a candidate that increases an already dominant class.  Add a tie
fixture and assert two calls with seed42 and the same attempt return the same
ID and objective values.

```python
def make_label_root(tmp_path, rows):
    root = tmp_path / "dataset"
    labels = root / "labels"
    labels.mkdir(parents=True)
    for sample_id, instance_counts in rows.items():
        semantic = []
        instance_ids = []
        next_instance = 0
        for class_id, count in enumerate(instance_counts):
            for _ in range(count):
                semantic.append(class_id)
                instance_ids.append(next_instance)
                next_instance += 1
        matrix = [
            [int(left == right) for right in instance_ids]
            for left in instance_ids
        ]
        (labels / (sample_id + ".json")).write_text(
            json.dumps({"seg": semantic, "inst": matrix}),
            encoding="utf-8",
        )
    return root


def test_replacement_minimizes_balance_objective(tmp_path):
    root = make_label_root(
        tmp_path,
        {
            "kept": [4, 0],
            "fills_missing": [0, 1],
            "adds_dominant": [2, 0],
        },
    )
    result = select_balanced_replacement_id(
        root,
        successful_ids=("kept",),
        candidate_ids=("fills_missing", "adds_dominant"),
        base_seed=42,
        split_name="train",
        attempt=1,
    )
    assert result.sample_id == "fills_missing"
    assert result.balance_before[0] == 1
    assert result.balance_after[0] == 0
```

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```powershell
& 'G:\ProgramData\anaconda3\envs\aagmspenv\python.exe' `
  -m pytest tests/test_balanced_replacement.py -q
```

Expected: collection/import failure because `ReplacementSelection` and
`select_balanced_replacement_id` do not exist.

- [ ] **Step 3: Implement the selector with the existing objective**

Load successful and candidate records with `load_record`.  Restrict comparison
to columns present in their combined matrix.  For each candidate, calculate the
objective after addition, then order candidates by:

```python
(
    proposed_uncovered,
    proposed_coefficient,
    proposed_spread,
    -candidate_rarity,
    deterministic_tie_rank,
)
```

Create the tie permutation with:

```python
replacement_seed = derive_seed(
    base_seed,
    "conversion_replacement:" + split_name,
    str(attempt),
)
rng = np.random.default_rng(replacement_seed)
```

Validate non-empty, unique, disjoint ID lists, `split_name in ("train",
"val")`, and `attempt >= 1`.

- [ ] **Step 4: Run selector tests and preparation sampling tests**

Run:

```powershell
& 'G:\ProgramData\anaconda3\envs\aagmspenv\python.exe' `
  -m pytest -q tests/test_balanced_replacement.py `
  tests/test_prepare_dataset.py::test_sample_percent_applies_to_train_and_val_but_not_test `
  tests/test_prepare_dataset.py::test_fractional_sparse_split_selects_available_classes
```

Expected: all pass.

- [ ] **Step 5: Commit Task 1**

```powershell
git add -- tools/sample_mfrinstseg_1pct.py tests/test_balanced_replacement.py
git commit -m "feat: add deterministic balanced replacement selector"
```

---

### Task 2: Initial-versus-Effective Manifest Contract

**Files:**
- Modify: `tools/prepare_dataset.py`
- Modify: `tests/test_prepare_dataset.py`

**Interfaces:**
- Consumes: unchanged `build_preparation_manifest(config, selected_ids, samples)` request inputs.
- Produces:

```python
REPLACEMENT_POLICY_VERSION = 1


def _effective_selected_ids(manifest: Mapping[str, object]) -> dict:
    """Return effective IDs, falling back to legacy selected_ids."""


def _conversion_exclusions_path(prepared_root: Path) -> Path:
    return prepared_root / "conversion_exclusions.json"
```

- New manifests contain `effective_selected_ids`, `conversion_exclusions`, and
  `replacement_policy_version`; `_request_fingerprint` continues to use initial
  `selected_ids` and ignores those recovery outputs.

- [ ] **Step 1: Write failing manifest compatibility tests**

Add tests that assert:

1. a new request publishes initial and effective IDs identically;
2. changing only effective IDs or exclusion records does not change the request
   fingerprint;
3. `_effective_selected_ids` falls back to `selected_ids` for a legacy manifest;
4. `_validate_prepared_outputs` validates split files and BINs against effective
   IDs when present.

```python
def test_replacement_outputs_do_not_change_request_fingerprint(tmp_path):
    manifest = build_test_manifest(tmp_path)
    recovered = json.loads(json.dumps(manifest))
    recovered["effective_selected_ids"]["train"] = ["replacement"]
    recovered["conversion_exclusions"] = [{"failed_sample_id": "failed"}]
    assert prepare_dataset._request_fingerprint(manifest) == (
        prepare_dataset._request_fingerprint(recovered)
    )
```

- [ ] **Step 2: Run the focused manifest tests and verify RED**

Run the four new test node IDs with `pytest -q`.  Expected failures: missing
effective-selection fields/helper and validation still reading `selected_ids`.

- [ ] **Step 3: Implement backward-compatible manifest semantics**

Update `build_preparation_manifest` to publish:

```python
"selected_ids": normalized,
"effective_selected_ids": {name: list(normalized[name]) for name in SPLIT_NAMES},
"replacement_policy_version": REPLACEMENT_POLICY_VERSION,
"conversion_exclusions": [],
```

Use `_effective_selected_ids(existing)` inside `_validate_prepared_outputs` and
when returning a reused `PreparedDataset`.  Keep the request fingerprint field
list unchanged so complete legacy caches remain compatible.

Initialize `conversion_exclusions.json` atomically to `[]\n` for new/rebuilt
preparation.  Do not yet change failure recovery flow.

- [ ] **Step 4: Run all preparation tests**

```powershell
& 'G:\ProgramData\anaconda3\envs\aagmspenv\python.exe' `
  -m pytest tests/test_prepare_dataset.py -q
```

Expected: all existing and new tests pass.

- [ ] **Step 5: Commit Task 2**

```powershell
git add -- tools/prepare_dataset.py tests/test_prepare_dataset.py
git commit -m "feat: separate requested and effective dataset selections"
```

---

### Task 3: Replacement Conversion Coordinator

**Files:**
- Modify: `tools/prepare_dataset.py`
- Modify: `tests/test_prepare_dataset.py`

**Interfaces:**
- Consumes: Task 1 `select_balanced_replacement_id` and Task 2 effective-selection helpers.
- Produces private helper `_replaceable_failure(failure:
  Mapping[str, object], requested_ids: Mapping[str, Sequence[str]]) -> bool`.

The exact coordinator signature is `_recover_replaceable_failures(config:
PreparationConfig, model_splits: Mapping[str, Sequence[str]], requested_ids:
Mapping[str, Sequence[str]], successful_ids: Mapping[str, Sequence[str]],
failures: Sequence[Mapping[str, object]], converter: object, paths: object,
options: object, existing_conversion: Optional[Mapping[str, object]] = None) ->
tuple[dict, list, list, dict]`.  Its four returns are effective successes,
unresolved failures, audit rows, and accumulated conversion counts.

- [ ] **Step 1: Write failing tests for successful replacement and retry**

Add a fake converter that fails one initial Train ID, succeeds a deterministic
replacement, and records every invocation.  Assert:

- final Train length equals requested length;
- all original successes remain;
- the failed ID is absent;
- Test IDs are unchanged;
- only the replacement is present in the second converter call;
- `conversion_exclusions.json` maps failure to replacement;
- `conversion_failures.json` is empty;
- final split files equal effective IDs.

Add a second test where the first replacement fails and the next succeeds;
assert two ordered audit records and no repeated candidate.

- [ ] **Step 2: Run both coordinator tests and verify RED**

Expected: `ensure_prepared_dataset` raises on the first converter failure
instead of selecting replacements.

- [ ] **Step 3: Implement failure classification and replacement loop**

Import `select_balanced_replacement_id`.  Treat a failure as replaceable only
when its split and sample ID satisfy the design contract.  For each split,
maintain:

```python
effective = set(successful_ids[split_name])
failed = set(recorded_failed_ids)
candidates = [
    sample_id
    for sample_id in model_splits[split_name]
    if sample_id not in effective
    and sample_id not in requested_ids[split_name]
    and sample_id not in failed
]
```

Call the Task 1 selector, then invoke the converter with all split keys and only
the chosen ID in its source split:

```python
replacement_batch = {name: [] for name in SPLIT_NAMES}
replacement_batch[split_name] = [selection.sample_id]
result = converter.convert_selected_samples(
    replacement_batch,
    paths,
    config.extractor_file,
    config.schema_file,
    options,
    False,
)
```

Accumulate converted/skipped counts, append an audit record for each attempt,
and repeat until the effective split reaches the requested size.  Preserve
fatal failures and raise after atomically publishing them.

- [ ] **Step 4: Integrate the coordinator into new preparation**

After the initial conversion result:

1. call `_recover_replaceable_failures`;
2. set `request["effective_selected_ids"]` and
   `request["conversion_exclusions"]`;
3. publish full final split files after all replacement rounds;
4. validate final effective outputs;
5. write unresolved failures to `conversion_failures.json` and audit rows to
   `conversion_exclusions.json`;
6. return effective IDs through `PreparedDataset`.

- [ ] **Step 5: Add and run fatal-policy tests**

Add parameterized tests proving that Test failure, missing sample ID, unknown
split, unknown selected ID, and candidate exhaustion remain fatal and are
persisted.  Run all preparation tests; expected all pass.

- [ ] **Step 6: Commit Task 3**

```powershell
git add -- tools/prepare_dataset.py tests/test_prepare_dataset.py
git commit -m "feat: replace unconvertible training subset models"
```

---

### Task 4: Resume an Existing Recoverable Failed Cache

**Files:**
- Modify: `tools/prepare_dataset.py`
- Modify: `tests/test_prepare_dataset.py`

**Interfaces:**
- Consumes: Task 3 `_recover_replaceable_failures`.
- Produces: a matching existing manifest with only replaceable Train/Validation
  failures enters targeted recovery without `rebuild=True`.

- [ ] **Step 1: Write a failing integration test matching the current 10% state**

Construct a matching request manifest with:

```python
manifest["successful_ids"] = {
    "train": list(selected["train"][:-1]),
    "val": list(selected["val"]),
    "test": list(selected["test"]),
}
manifest["failed_ids"] = [{
    "split": "train",
    "sample_id": selected["train"][-1],
    "exception_type": "ValueError",
    "message": "Face 14 has no valid UV candidates",
}]
```

Write valid BIN fixtures for every successful ID.  The fake converter must
fail the test if any successful ID is submitted and succeed only for the one
replacement.  Assert `prepared.reused is False`, exact final counts, empty
unresolved failures, and one persisted exclusion.

- [ ] **Step 2: Run the integration test and verify RED**

Expected: `PreparationMismatchError("Prepared cache is incomplete; choose a new prepared directory or rebuild it")`.

- [ ] **Step 3: Add the targeted existing-cache recovery branch**

When fingerprints match but output validation fails:

- recover only if `failed_ids` is non-empty and every failure is replaceable;
- use existing `successful_ids` as the fixed effective base;
- validate existing successful BINs before choosing replacements;
- call Task 3 coordinator with existing conversion counts;
- never pass existing successful IDs to the converter;
- atomically publish the upgraded final manifest and audit files.

If the existing manifest is request-only (`successful_ids` all empty,
`failed_ids` empty), preserve the current incomplete-cache error.  If any
failure is fatal, preserve the current error path.

- [ ] **Step 4: Run recovery and non-resume regression tests**

Run:

```powershell
& 'G:\ProgramData\anaconda3\envs\aagmspenv\python.exe' `
  -m pytest -q tests/test_prepare_dataset.py
```

Expected: existing recoverable cache test passes; malformed, mismatched,
request-only interrupted, and invalid complete caches still fail closed.

- [ ] **Step 5: Commit Task 4**

```powershell
git add -- tools/prepare_dataset.py tests/test_prepare_dataset.py
git commit -m "fix: resume preparation through balanced replacement"
```

---

### Task 5: Verification and Handoff

**Files:**
- Verify only; modify production files only if a test exposes a defect in the
  approved scope.

**Interfaces:**
- Consumes: Tasks 1-4.
- Produces: verified tmp-branch implementation and the exact user command for
  targeted 10% recovery.

- [ ] **Step 1: Run syntax and focused suites**

```powershell
& 'G:\ProgramData\anaconda3\envs\aagmspenv\python.exe' `
  -m py_compile tools/sample_mfrinstseg_1pct.py tools/prepare_dataset.py

& 'G:\ProgramData\anaconda3\envs\aagmspenv\python.exe' `
  -m pytest -q tests/test_balanced_replacement.py `
  tests/test_prepare_dataset.py tests/test_small_sample_seed42_runner.py
```

- [ ] **Step 2: Run the full available unit suite**

```powershell
& 'G:\ProgramData\anaconda3\envs\aagmspenv\python.exe' `
  -m pytest -q --ignore=tests/test_model_result_paths.py
```

The ignored file is the known environment-only `prettytable` dependency
blocker; report it explicitly rather than claiming an unqualified full-suite
pass.

- [ ] **Step 3: Review repository state**

```powershell
git diff --check
git status --short
git log --oneline -8
```

Confirm no raw datasets, BINs, result logs, or checkpoints are staged.

- [ ] **Step 4: Perform an inline code review**

Check the implementation against every success criterion in
`docs/superpowers/specs/2026-08-04-conversion-failure-replacement-design.md`.
Pay particular attention to Test immutability, deterministic replacement,
legacy manifest fallback, audit persistence, and avoiding reconversion of
existing successes.

- [ ] **Step 5: Commit review fixes if any**

If review requires an in-scope correction, repeat its focused RED/GREEN test
and commit only that correction.  Otherwise create no empty commit.

- [ ] **Step 6: Hand off without running formal conversion**

Provide this recovery command without `-RebuildPreparedData`:

```powershell
powershell.exe -ExecutionPolicy Bypass -File `
  "D:\Code\BrepMFR\tools\run_mfrinstseg_small_sample_seed42.ps1" `
  -SamplePercent 0.10
```

Explain that preparation should reuse 14,685 existing BINs, select and convert
one deterministic replacement, publish 4,374/937/9,375 IDs, then enter the
four training runs.  Keep the tmp branch for user validation; do not merge or
delete it.
