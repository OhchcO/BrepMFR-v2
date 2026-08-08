# Repository-Local AAG Assets Design

## Objective

Make `tools/AAGExtractor.py` and `tools/schema.json` the only default AAG
extraction assets used by BrepMFR. Remove every active dependency on the old
external predecessor checkout while retaining optional
explicit path overrides for controlled experiments.

`TopologyUtils` remains supplied by `pythonocc-core` through
`OCC.Extend.TopologyUtils`; it is not copied, replaced, or reimplemented.

## Canonical Assets

The repository owns and versions these files:

```text
tools/AAGExtractor.py
tools/schema.json
```

The schema must satisfy the converter's current UV-grid contract:

```json
{
  "num_srf_u": 5,
  "num_srf_v": 5,
  "num_crv_u": 5
}
```

The extractor must continue to expose both `AAGExtractor` and the imported
`TopologyUtils` symbol because the converter uses `TopologyUtils` to derive
face loop counts from extracted OCC shapes.

## Path Resolution

Python entry points derive defaults from their own repository location rather
than the current working directory:

- `tools/mfrinstseg_to_bin.py` defaults to sibling `AAGExtractor.py` and
  `schema.json`.
- `tools/prepare_dataset.py` defaults to the same sibling files instead of
  requiring both flags.
- `segmentation.py` defaults to the repository root's `tools/AAGExtractor.py`
  and `tools/schema.json`.

PowerShell launchers resolve defaults from `$PSScriptRoot`:

- `tools/run_mfrinstseg_experiments.ps1`;
- `tools/run_mfrinstseg_test.ps1`.

Callers may still pass `-ExtractorFile` and `-SchemaFile` to reproduce a cache
with deliberately different assets. Cache fingerprints continue to include
the actual selected file contents, so an override cannot silently reuse a
cache built from the repository defaults.

The legacy Git Bash helper resolves both assets under its configured
`PROJECT_ROOT/tools` directory.

## Documentation Migration

Historical specs and plans that name the removed absolute external paths
are updated to point at the repository-local assets. Statements that prohibit
copying or versioning the extractor are revised because they no longer match
the current architecture.

No references to the predecessor checkout may remain in tracked files after the
migration.

## Validation and Errors

Entry points retain their existing preflight checks and fail with the resolved
local path when an asset is missing. `load_schema` remains the authoritative
schema validator; the checked-in schema is tested through that function.

The extractor is dynamically imported only when conversion is requested.
Cold import may take tens of seconds because `occwl.graph` loads the broad OCC
`DataExchange` module graph; this is expected startup cost, not a fairseq
dependency or model-training failure.

## Testing

Tests verify:

- the checked-in schema passes `load_schema`;
- the checked-in extractor defines `AAGExtractor` and `TopologyUtils`;
- Python CLI defaults resolve to the two checked-in files independently of
  the current working directory;
- PowerShell launchers default to `$PSScriptRoot` assets and still accept
  overrides;
- the Git Bash helper uses `$PROJECT_ROOT/tools`;
- tracked source and documentation contain no predecessor-checkout reference;
- PowerShell syntax parsing and focused Python suites remain green;
- a one-sample conversion smoke test is run before any full 10% preparation.

Full dataset conversion and formal training remain outside this migration.
