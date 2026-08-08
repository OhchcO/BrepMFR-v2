# MFRInstSeg Shared BIN Cache Design

## Goal

Avoid converting and storing the same MFRInstSeg model more than once when
running the seed42 1%, 5%, and 10% experiments.  The split percentage changes
which model IDs are listed in `train.txt` and `val.txt`; it does not change the
geometry extracted for a particular model.  The official `test.txt` is exactly
the same for all three percentages.

This change separates split manifests from the immutable, feature-compatible
BIN cache while preserving existing datasets and commands.

## Scope

The implementation will:

- give 1%, 5%, and 10% seed42 runs one shared ParamSurface-v4 BIN cache;
- keep percentage-specific preparation manifests and split text files;
- import compatible existing 5% BINs into the shared cache with NTFS hard
  links, without copying file contents or deleting the donor cache;
- convert only selected model IDs that are absent from the shared cache;
- keep legacy prepared directories working when no shared cache is supplied;
- run unit tests and small synthetic/smoke checks only, not formal conversion,
  training, or evaluation.

The change will not merge the temporary branch into `main`, delete old caches,
or change model architecture, losses, data selection, or evaluation metrics.

## Directory Layout

The shared cache is independent of `SamplePercent`:

```text
D:\Code\Datasets\MFRInstSeg_BrepMFR_ParamSurface_v4_shared_seed42
├── bin\
│   ├── <sample-id>.bin
│   └── ...
└── bin_cache_manifest.json
```

Percentage-specific roots remain independent:

```text
MFRInstSeg_BrepMFR_ParamSurface_v4_seed42\
MFRInstSeg_BrepMFR_ParamSurface_v4_5pct_seed42\
MFRInstSeg_BrepMFR_ParamSurface_v4_10pct_seed42\
```

Each percentage root owns only its split/preparation state:

```text
train.txt
val.txt
test.txt
preparation_manifest.json
conversion_failures.json
```

Existing local `bin\` directories are left untouched for backward
compatibility.  New shared-cache runs load BINs from the explicit shared root.

## Cache Identity and Safety

`bin_cache_manifest.json` identifies properties that affect a model's BIN but
excludes split membership and sample percentage.  Its fingerprint contains:

- converter and feature-contract versions;
- base random seed;
- STEP and label content hashes for the raw dataset;
- extractor and schema content hashes;
- point-sampling mode, bounds, exponent, and candidate grids;
- enabled point-cloud, cylindrical, and parametric-surface outputs.

Before reading, linking, or adding BINs, the requested fingerprint must equal
the shared-cache fingerprint.  A mismatch fails closed with an instruction to
use a different shared cache path.  An incompatible cache is never mixed or
silently overwritten.

The existing percentage-specific preparation fingerprint continues to include
`SamplePercent`, selected IDs, and split ratios.  This preserves exact split
reproducibility while allowing BIN identity to remain percentage-independent.

## Data Flow

For one percentage run:

1. Resolve the official train/validation/test split and the balanced percentage
   subsets exactly as before.
2. Build and verify the percentage-specific preparation fingerprint.
3. Build or verify the shared BIN-cache fingerprint.
4. On first shared-cache use, inspect configured donor prepared roots.  A donor
   is eligible only when its preparation manifest implies the same BIN-cache
   fingerprint.
5. For selected IDs only, create a hard link from each eligible donor BIN to the
   shared `bin\` directory when the target does not already exist.
6. Run the normal converter for all selected IDs with overwrite disabled.
   Existing shared BINs are validated and reported as skipped; only missing IDs
   are extracted from STEP.
7. Validate every selected BIN against the requested feature contract.
8. Train and test using the percentage root for the split manifest and the
   shared root for BIN lookup.

For the planned 10% seed42 run, the completed 5% cache supplies all 9,375 test
BINs and overlapping train/validation BINs.  Only additional 10% train and
validation models require conversion.

## Component Changes

### Dataset preparation

`PreparationConfig` gains an optional shared BIN-cache root and optional donor
roots.  When no shared root is supplied, current `prepared_root/bin` behaviour
is unchanged.

The preparation layer will own:

- building and validating the percentage-independent BIN-cache manifest;
- selecting compatible donors;
- creating hard links atomically where possible;
- directing converter BIN output to the shared root while keeping reports and
  split files in the percentage root;
- returning both the split root and BIN root to the training entry point.

Hard-link creation is attempted only when source and destination are on the
same filesystem.  If linking is unavailable, the donor is skipped and the
normal STEP converter generates the missing BIN.  The implementation will not
silently copy large BIN files.

### Converter paths

`ConversionPaths` will distinguish:

- `output_root`: split manifests and conversion reports;
- `bins_root`: BIN input/output location.

Its default remains `output_root/bin`, preserving existing CLI and test
behaviour.

### Dataset loader

`CADSynth` gains an optional BIN root.  It reads `<split>.txt` from its normal
dataset root and resolves requested IDs only under `<bin-root>/bin`.  If the
option is absent, it keeps scanning the original dataset root.

Duplicate stems and missing IDs remain hard errors.  The loader never falls
back between incompatible roots.

### Training and test launchers

The Python training entry, generic experiment PowerShell launcher, test entry,
and test PowerShell launcher gain an optional shared BIN-cache argument.

`run_mfrinstseg_small_sample_seed42.ps1` supplies the same shared path to all
three percentages and offers the existing 5% prepared directory as the first
donor.  Experiment names, result directories, seeds, and four-model matrices
remain unchanged.

## Rebuild Semantics

Legacy preparation keeps its existing rebuild behaviour.

With an explicit shared cache, rebuilding a percentage root regenerates that
root's split/preparation state but does not overwrite already valid shared
BINs.  Feature-changing rebuilds must use a new shared-cache path; this avoids
silently changing BINs used by another percentage experiment.

No automatic deletion, move, or rename is performed.  The existing 5% cache
remains a valid standalone dataset and a hard-link donor.

## Error Handling

The preparation stops before training when:

- the shared-cache fingerprint differs;
- a selected shared BIN exists but fails feature-contract validation;
- a selected model remains missing after conversion;
- duplicate model stems are visible in the configured BIN root.

Optional incompatible donors are ignored with a warning.  Conversion failures
remain recorded in the percentage root's `conversion_failures.json`.

## Testing

Tests will cover:

1. BIN-cache identity excludes percentage and split membership but includes
   seed, source contents, extractor/schema, and geometry options.
2. Compatible donors create hard links only for selected IDs and do not alter
   donor files.
3. Incompatible donors are rejected or skipped according to configuration.
4. Converter reports stay in the percentage root while BINs go to the shared
   root.
5. Existing compatible BINs are validated and skipped; missing BINs are
   converted once.
6. `CADSynth` reads manifests and BINs from separate roots and retains legacy
   single-root behaviour.
7. Training and test launchers propagate the shared root consistently.
8. The seed42 small-sample runner maps 1%, 5%, and 10% to distinct split roots
   and one identical shared BIN root.
9. Existing experiment, preparation, converter, loader, and test suites remain
   green.

## Success Criteria

- A 10% seed42 preparation does not convert any official test model already
  present in the compatible 5% donor.
- No BIN file contents are duplicated during donor import on the current NTFS
  volume.
- 1%, 5%, and 10% continue to have independent, reproducible train/validation
  manifests and result names.
- Old prepared datasets and commands work without a shared-cache argument.
- Formal training and evaluation predictions are unaffected by cache location.
