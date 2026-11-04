# Reproducible SPE-PCAAG Experiments on MFRInstSeg

Run these commands from Windows PowerShell after activating the project
environment. Dataset preparation is deterministic and cached by a fingerprint
of the raw files and every conversion option. All formal ablations below use
the same prepared directory, seed, split, sample percentage, geometry output
contract, and disabled random rotation.

## One full run and the seven-run matrix

Prepare the cache if needed and train one full model:

```powershell
powershell.exe -ExecutionPolicy Bypass -File `
  "D:\Code\BrepMFR\tools\run_mfrinstseg_experiments.ps1" `
  -Mode full `
  -RawDatasetPath "D:\Code\Datasets\MFRInstSeg" `
  -PreparedDatasetPath "D:\Code\Datasets\MFRInstSeg_BrepMFR_SPEPCAAG" `
  -SamplePercent 0.01 `
  -Seed 2026
```

Run all seven aliases in the fixed matrix order against that cache:

```powershell
powershell.exe -ExecutionPolicy Bypass -File `
  "D:\Code\BrepMFR\tools\run_mfrinstseg_experiments.ps1" `
  -All `
  -RawDatasetPath "D:\Code\Datasets\MFRInstSeg" `
  -PreparedDatasetPath "D:\Code\Datasets\MFRInstSeg_BrepMFR_SPEPCAAG" `
  -SamplePercent 0.01 `
  -Seed 2026
```

The aliases are independent model configurations:

| Alias | Instance mode | Face point cloud | Cylindrical position | Instance-bias scale |
| --- | --- | ---: | ---: | ---: |
| `baseline` | `baseline` | off | off | 1.0 |
| `cyl` | `baseline` | off | on | 1.0 |
| `point` | `baseline` | on | off | 1.0 |
| `spe_pcaag` | `baseline` | on | on | 1.0 |
| `instance` | `dynamic_affinity` | off | off | 1.0 |
| `affinity_only` | `dynamic_affinity` | off | off | 0.0 |
| `full` | `dynamic_affinity` | on | on | 1.0 |

To launch the aliases separately while preserving the same formal
configuration, run:

```powershell
$Runner = "D:\Code\BrepMFR\tools\run_mfrinstseg_experiments.ps1"
$Raw = "D:\Code\Datasets\MFRInstSeg"
$Prepared = "D:\Code\Datasets\MFRInstSeg_BrepMFR_SPEPCAAG"

powershell.exe -ExecutionPolicy Bypass -File $Runner -Mode baseline      -RawDatasetPath $Raw -PreparedDatasetPath $Prepared -SamplePercent 0.01 -Seed 2026
powershell.exe -ExecutionPolicy Bypass -File $Runner -Mode cyl           -RawDatasetPath $Raw -PreparedDatasetPath $Prepared -SamplePercent 0.01 -Seed 2026
powershell.exe -ExecutionPolicy Bypass -File $Runner -Mode point         -RawDatasetPath $Raw -PreparedDatasetPath $Prepared -SamplePercent 0.01 -Seed 2026
powershell.exe -ExecutionPolicy Bypass -File $Runner -Mode spe_pcaag     -RawDatasetPath $Raw -PreparedDatasetPath $Prepared -SamplePercent 0.01 -Seed 2026
powershell.exe -ExecutionPolicy Bypass -File $Runner -Mode instance      -RawDatasetPath $Raw -PreparedDatasetPath $Prepared -SamplePercent 0.01 -Seed 2026
powershell.exe -ExecutionPolicy Bypass -File $Runner -Mode affinity_only -RawDatasetPath $Raw -PreparedDatasetPath $Prepared -SamplePercent 0.01 -Seed 2026
powershell.exe -ExecutionPolicy Bypass -File $Runner -Mode full          -RawDatasetPath $Raw -PreparedDatasetPath $Prepared -SamplePercent 0.01 -Seed 2026
```

## Formal full-test evaluation

Evaluate a selected checkpoint on every sample listed in the prepared
directory's complete `test.txt`:

```powershell
powershell.exe -ExecutionPolicy Bypass -File `
  "D:\Code\BrepMFR\tools\run_mfrinstseg_test.ps1" `
  -Checkpoint "D:\Code\BrepMFR\results\<experiment>\<run>\best-<epoch>-<eval_loss>.ckpt" `
  -PreparedDataset "D:\Code\Datasets\MFRInstSeg_BrepMFR_SPEPCAAG" `
  -Seed 2026
```

`SamplePercent` affects the selected training and validation subsets, not the
formal test split. The evaluator never samples or reconverts `test.txt`. It
loads the checkpoint first and requires exactly the point-cloud and
cylindrical BIN fields enabled by that checkpoint. Dynamic-affinity inference
does not consume ground-truth instance labels; only `oracle` evaluation
requires them.

Replace the checkpoint placeholders with the actual Task 7
`best-<epoch>-<eval_loss>.ckpt` filename recorded in the run's `log.txt`.

The evaluator writes `test_metrics.json` and `test_log.txt` beside the
checkpoint unless `--output-dir` is supplied directly. Both outputs record the
seed, geometry switches and scales, resolved prepared path, checkpoint,
experiment mode, batch/worker settings, and device, in addition to loss,
per-face accuracy, mIoU, macro-F1, per-class metrics, and the JSON confusion
matrix.

Normally, training has already prepared and fingerprinted the cache. If a
standalone evaluation must prepare it first, use the same preparation options:

```powershell
powershell.exe -ExecutionPolicy Bypass -File `
  "D:\Code\BrepMFR\tools\run_mfrinstseg_test.ps1" `
  -Checkpoint "D:\Code\BrepMFR\results\<experiment>\<run>\best-<epoch>-<eval_loss>.ckpt" `
  -PreparedDataset "D:\Code\Datasets\MFRInstSeg_BrepMFR_SPEPCAAG" `
  -Prepare `
  -RawDatasetPath "D:\Code\Datasets\MFRInstSeg" `
  -SamplePercent 0.01 `
  -PointSamplingMode area_adaptive `
  -FixedPointCount 64 `
  -MinPointCount 32 `
  -MaxPointCount 96 `
  -AreaSamplingExponent 0.5 `
  -Seed 2026
```

This invokes `tools\prepare_dataset.py`; it does not use the old test-only
converter path.

## Prepare the shared seed42 BIN cache without training

Run 10% first. This creates the shared parameter-surface BIN pool and the
independent 10% manifests without starting a trainer:

```powershell
$Python = 'F:\ProgramData\anaconda3\envs\aagmspenv\python.exe'
$Project = 'D:\Code\BrepMFR'
$Raw = 'D:\Code\Datasets\MFRInstSeg'
$Shared = 'D:\Code\Datasets\MFRInstSeg_BrepMFR_ParamSurface_v4_shared_seed42'
$Prepared10 = 'D:\Code\Datasets\MFRInstSeg_BrepMFR_ParamSurface_v4_10pct_seed42'

& $Python "$Project\tools\prepare_dataset.py" `
  --raw-dataset $Raw `
  --prepared-dataset $Prepared10 `
  --shared-bin-cache $Shared `
  --seed 42 `
  --sample-percent 0.10 `
  --split-ratios 0.8 0.1 0.1
```

After 10% completes, prepare 5% and 1% against the same `$Shared` path.
Existing compatible BINs are validated and skipped; only absent IDs are
converted:

```powershell
foreach ($Item in @(
  @{ Percent = 0.05; Root = 'D:\Code\Datasets\MFRInstSeg_BrepMFR_ParamSurface_v4_5pct_seed42' },
  @{ Percent = 0.01; Root = 'D:\Code\Datasets\MFRInstSeg_BrepMFR_ParamSurface_v4_seed42' }
)) {
  & $Python "$Project\tools\prepare_dataset.py" `
    --raw-dataset $Raw `
    --prepared-dataset $Item.Root `
    --shared-bin-cache $Shared `
    --seed 42 `
    --sample-percent $Item.Percent `
    --split-ratios 0.8 0.1 0.1
  if ($LASTEXITCODE -ne 0) { throw "Preparation failed: $($Item.Percent)" }
}
```

Do not add `--rebuild` merely to prepare another percentage. Each percentage
uses its own preparation root while the shared cache remains incremental.

## Fixed-64 versus area-adaptive comparison

Sampling configuration is part of the cache fingerprint. Use different
prepared directories so the two conversion contracts cannot be mixed:

```powershell
# Area-adaptive point sampling.
powershell.exe -ExecutionPolicy Bypass -File `
  "D:\Code\BrepMFR\tools\run_mfrinstseg_experiments.ps1" `
  -Mode full `
  -RawDatasetPath "D:\Code\Datasets\MFRInstSeg" `
  -PreparedDatasetPath "D:\Code\Datasets\MFRInstSeg_BrepMFR_SPEPCAAG_adaptive" `
  -PointSamplingMode area_adaptive `
  -MinPointCount 32 `
  -MaxPointCount 96 `
  -AreaSamplingExponent 0.5 `
  -SamplePercent 0.01 `
  -Seed 2026

# Fixed 64 points per face.
powershell.exe -ExecutionPolicy Bypass -File `
  "D:\Code\BrepMFR\tools\run_mfrinstseg_experiments.ps1" `
  -Mode full `
  -RawDatasetPath "D:\Code\Datasets\MFRInstSeg" `
  -PreparedDatasetPath "D:\Code\Datasets\MFRInstSeg_BrepMFR_SPEPCAAG_fixed64" `
  -PointSamplingMode fixed `
  -FixedPointCount 64 `
  -SamplePercent 0.01 `
  -Seed 2026
```

Keep every other training and evaluation option equal, and evaluate each
checkpoint only against the `test.txt` and BIN files in its own matching
prepared directory.

## Full-data training

Use a new prepared directory for the `sample_percent=1.0` cache:

```powershell
powershell.exe -ExecutionPolicy Bypass -File `
  "D:\Code\BrepMFR\tools\run_mfrinstseg_experiments.ps1" `
  -Mode full `
  -RawDatasetPath "D:\Code\Datasets\MFRInstSeg" `
  -PreparedDatasetPath "D:\Code\Datasets\MFRInstSeg_BrepMFR_SPEPCAAG_100pct" `
  -SamplePercent 1.0 `
  -Seed 2026
```

The full official/generated test split is used whether `SamplePercent` is
`0.01` or `1.0`.

## Cache mismatches and conversion failures

A prepared directory is reused only when its manifest fingerprint exactly
matches the raw STEP/label content, extractor and schema, seed, split ratios,
sample percentage, point-sampling configuration, converter sources, and output
feature contract. On a mismatch, either select a new
`-PreparedDatasetPath`！preferred for comparisons！or deliberately rebuild the
same cache:

```powershell
powershell.exe -ExecutionPolicy Bypass -File `
  "D:\Code\BrepMFR\tools\run_mfrinstseg_experiments.ps1" `
  -Mode full `
  -RawDatasetPath "D:\Code\Datasets\MFRInstSeg" `
  -PreparedDatasetPath "D:\Code\Datasets\MFRInstSeg_BrepMFR_SPEPCAAG" `
  -SamplePercent 0.01 `
  -Seed 2026 `
  -RebuildPreparedData
```

Preparation always writes `conversion_failures.json`. A successful complete
cache contains:

```json
[]
```

If preparation fails, inspect that file for the sample ID, exception type, and
message. Do not train or evaluate from an incomplete cache.

## Old and new BIN fairness

Old BIN directories are valid only when both geometry switches are disabled;
point, cylindrical, SPE-PCAAG, and full checkpoints fail closed when their
required fields are absent or malformed. Legacy baseline/instance experiments
may still be run with `-AutoPrepare:$false -DatasetPath <old-bin-directory>`,
but those results must not be placed in the formal seven-way matrix.

For a fair formal matrix, prepare one new full-geometry cache and point both
`baseline` and `full`！and all five intermediate aliases！at that exact same
directory. Mixing an old baseline BIN directory with a new full BIN directory
changes the data/conversion provenance at the same time as the model, so the
result is not an ablation of model switches.

## Static test suite for maintainers

These tests use fixtures/doubles and should not convert a real STEP file or
start training:

```powershell
pytest tests/test_reproducibility.py tests/test_prepare_dataset.py tests/test_geometry_features.py tests/test_mfrinstseg_to_bin_geometry.py tests/test_dataset_geometry_fields.py tests/test_geometry_encoders.py tests/test_experiment_configuration.py -v
```
