# MFRInstSeg Conversion-Failure Replacement Design

## Goal

Allow a balanced Train or Validation subset to recover deterministically when
individual STEP models cannot be converted, without deleting raw data,
shrinking the requested subset, changing the official Test set, or rebuilding
already valid BINs.

The immediate recovery target is the seed42 10% cache.  Its conversion has
already produced 14,685 valid BINs and recorded one Train failure:

```text
sample_id: 20221121_154648_12065
reason: Face 14 has no valid UV candidates
```

The official Test conversion is complete and must remain unchanged.

## Scope

The implementation will:

- retain the initial balanced Train and Validation selections;
- classify conversion failures as replaceable or fatal;
- replace failed Train/Validation models from the same official split;
- select replacements against the remaining successful subset's class counts;
- retry replacement conversion until the requested subset size is restored or
  no eligible candidate remains;
- preserve every failure and replacement in an audit file and preparation
  manifest;
- reopen the existing failed 10% cache and convert only replacement models;
- keep complete old caches and failure-free preparation behaviour compatible.

It will not delete or rename STEP/JSON files, replace Test models, silently
ignore global failures, run formal conversion/training, or implement the
separate shared-BIN-cache design in the same change.

## Failure Policy

A conversion failure is replaceable only when all of the following hold:

- `split` is `train` or `val`;
- `sample_id` is a valid ID from that split's initial selected set;
- the error was raised while converting that individual sample;
- at least one unselected, non-failed model remains in the same official split.

The following remain fatal:

- any Test failure;
- a global/setup failure with no sample ID;
- an unknown split or sample ID;
- final BIN validation failure without an attributable replaceable sample;
- exhaustion of eligible replacement candidates.

Fatal failures continue to raise `PreparationError` before training.

## Deterministic Replacement Selection

Replacement selection keeps all successfully converted selected models fixed.
It does not rerun the full balanced sampler because doing so could unnecessarily
change many already converted IDs.

For each failed Train or Validation model:

1. Remove the failed ID from the effective selected set.
2. Calculate the 24-class feature-instance totals of the remaining effective
   set using the same label parser and class definition as the original
   balanced sampler.
3. Form candidates from the same official split, excluding initial/effective
   selected IDs and every model that failed in the current recovery chain.
4. For each candidate, calculate the balance objective after adding it:
   uncovered-class count, coefficient of variation, then count spread.
5. Choose the lexicographically best objective.  Break remaining ties using
   rarity score and a purpose-derived seed:

   ```text
   derive_seed(
       base_seed,
       "conversion_replacement:" + split_name,
       str(attempt),
   )
   ```

6. Convert the chosen candidate.  If it fails, audit it and repeat with the
   next deterministic candidate.
7. Stop when the effective set returns to its original requested size.

The selection helper will be added beside the existing balanced-sampling logic
so it shares label parsing, instance-count definitions, and objective
calculation rather than duplicating them in the converter.

## Manifest Model

The request fingerprint must remain stable across recovery.  Existing
`selected_ids` continues to mean the initial deterministic request and remains
part of the fingerprint.

The preparation manifest gains:

```json
{
  "effective_selected_ids": {
    "train": [],
    "val": [],
    "test": []
  },
  "conversion_exclusions": [],
  "replacement_policy_version": 1
}
```

`effective_selected_ids` contains the final IDs actually used by training.
For old manifests without this field, readers fall back to `selected_ids`.

`successful_ids`, split text files, output validation, and `PreparedDataset`
must match `effective_selected_ids`.  The initial `selected_ids` is retained for
reproducibility and cache fingerprint comparison.

`failed_ids` contains unresolved fatal failures only.  A cache is complete when
`failed_ids` is empty and every effective selected ID has a valid BIN.

## Audit File

Every percentage root gains `conversion_exclusions.json`.  It is an array of
records with this contract:

```json
[
  {
    "split": "train",
    "failed_sample_id": "20221121_154648_12065",
    "replacement_sample_id": "<deterministic replacement>",
    "exception_type": "ValueError",
    "message": "Face 14 has no valid UV candidates",
    "attempt": 1,
    "balance_before": {
      "uncovered": 0,
      "coefficient_of_variation": 0.0,
      "spread": 0
    },
    "balance_after": {
      "uncovered": 0,
      "coefficient_of_variation": 0.0,
      "spread": 0
    }
  }
]
```

The actual balance values are calculated from labels.  The file is written
atomically and remains present after successful preparation.  It is an audit
record, not an unresolved error report.

`conversion_failures.json` keeps its existing meaning: unresolved failures that
prevent training.  It becomes `[]` after all replaceable failures are resolved.

## New Preparation Flow

### New cache

1. Resolve official splits and initial balanced subsets as before.
2. Convert the initial selected IDs.
3. Separate individual Train/Validation failures from fatal failures.
4. If fatal failures exist, publish them and stop.
5. Incrementally choose and convert replacements for replaceable failures.
6. Repeat until Train and Validation sizes equal their initial requested sizes.
7. Publish final split manifests, audit records, successful IDs, and an empty
   unresolved-failure list.
8. Validate all effective selected BINs and return the prepared dataset.

### Existing recoverable failed cache

When the request fingerprint matches an existing manifest:

1. If the cache is already complete, reuse it exactly as before.
2. If every recorded failure is replaceable, load the recorded successful IDs
   and continue at replacement selection.
3. Do not reconvert successful initial IDs.
4. Convert only selected replacement candidates.
5. Atomically upgrade the manifest to the new effective-selection schema.

An interrupted request-only manifest with no completed conversion report is not
covered by this feature and retains the existing rebuild requirement.

## Converter Interaction

The replacement coordinator lives in `prepare_dataset.py`; the STEP converter
continues to report per-sample successes and failures.

Replacement conversion calls must not leave percentage split files containing
only the most recent replacement batch.  Preparation owns final split-file
publication and rewrites all three manifests atomically after recovery.

Conversion counts are accumulated across the initial and replacement rounds.
Existing successful BINs are never passed with overwrite enabled during
recovery.

## Compatibility

- Complete existing manifests without `effective_selected_ids` remain reusable.
- Failure-free new preparation produces the same selected IDs as before and an
  empty exclusion audit.
- Existing CLI arguments and experiment modes remain unchanged.
- The seed42 10% runner is rerun without `-RebuildPreparedData` after the code
  change; the recorded failed manifest triggers targeted recovery.
- 1% and 5% results, prepared roots, and checkpoints are not modified.

## Testing

Tests will cover:

1. Incremental replacement preserves successful IDs and requested subset size.
2. Replacement comes only from the failed model's official split.
3. Candidate choice is deterministic for a fixed seed and class matrix.
4. The best candidate is chosen using the documented balance objective and
   deterministic tie-breaking.
5. A failed replacement is audited and followed by another candidate.
6. Test, global, malformed, and exhausted-candidate failures remain fatal.
7. Existing recoverable manifests resume using successful IDs without
   reconverting them.
8. Request-only interrupted manifests remain non-resumable.
9. Initial and effective IDs have the documented fingerprint/validation roles.
10. `conversion_exclusions.json` is atomic, complete, and persists on success.
11. Old complete manifests and failure-free conversion remain compatible.
12. Relevant sampler, preparation, converter, experiment, and full unit suites
    remain green.

No full raw-dataset conversion or model training is part of verification.

## Success Criteria

- The existing 10% cache replaces `20221121_154648_12065` with one
  deterministic official-Train candidate.
- Exactly 4,374 Train, 937 Validation, and 9,375 official Test IDs are published.
- The 14,685 existing valid BINs are retained; only replacement BIN attempts
  require STEP extraction.
- The exclusion and replacement are reproducibly documented.
- Training begins only after final counts and BIN contracts validate.
