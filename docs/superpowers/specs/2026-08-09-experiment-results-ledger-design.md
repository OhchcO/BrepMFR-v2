# Experiment Results Ledger Design

## Goal

Replace the current manually assembled experiment summary with one traceable
Markdown report whose metrics and configurations can be matched to the local
`results/**/test_metrics.json` artifacts.

## Scope

- Update only `docs/experiment_summary.md` as the user-facing result report.
- Inspect every local `test_metrics.json` result, including repeated and
  exploratory runs.
- Do not add a persistent generator script or CSV ledger.
- Do not change model, training, conversion, or evaluation code.

## Report Structure

The report will contain:

1. the experiment scope and metric definitions;
2. a reviewed main table for final experiments;
3. grouped 1%, 5%, and 10% results;
4. a complete result index for all discovered metric files;
5. configuration and provenance anomalies;
6. conclusions limited to evidence supported by traceable results;
7. next actions needed before thesis publication.

Every reported experiment row will include or link to its source
`test_metrics.json` path. Repeated runs remain visible instead of being
silently overwritten.

## Classification Rules

- `approved`: complete metrics and a controlled, internally consistent
  comparison suitable for the main result table.
- `needs_review`: usable result with a seed, mapping, dataset, or provenance
  inconsistency; excluded from definitive cross-model conclusions.
- `exploratory`: parameter search, legacy mapping, Oracle, geometry study, or
  other non-final experiment; retained in the complete index.
- `invalid`: unreadable JSON or missing critical metrics/provenance.

Zero clustering scores are reported as measured results, not treated as a
training failure.

## Consistency Checks

For each result, compare:

- the experiment-directory seed with the JSON `seed`;
- the experiment-directory percentage with `prepared_dataset_path`;
- relation mapping and critical model switches within a claimed comparison;
- checkpoint path and experiment directory;
- Accuracy, mIoU, Macro-F1, raw pairwise F1, clustering F1, and ARI fields;
- repeated experiment names and timestamps.

The existing 5% summary values will not be retained unless they can be traced
to local JSON. The 10% seed-recording and relation-mapping inconsistencies will
be shown explicitly.

## Verification

After writing the report:

1. rediscover all local `test_metrics.json` files and confirm the report's
   complete index has the same count;
2. reparse the source JSON for every main-table row and compare all rendered
   metrics;
3. verify that every source path in the report exists;
4. scan the report for unsupported claims, stale manually copied figures, and
   contradictory status labels;
5. inspect the Git diff to ensure no model or experiment artifact was changed.

