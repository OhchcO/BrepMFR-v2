param(
    [Parameter(Mandatory = $true)]
    [double]$SamplePercent,
    [string]$RawDatasetPath = "D:\Code\Datasets\MFRInstSeg",
    [string]$PreparedDatasetBasePath = "D:\Code\Datasets\MFRInstSeg_BrepMFR_ParamSurface_v4",
    [string]$SharedBinCachePath = "D:\Code\Datasets\MFRInstSeg_BrepMFR_ParamSurface_v4_shared_seed42",
    [string[]]$DonorPreparedDatasetPaths = @(),
    [string]$ResultBasePath = "D:\Code\BrepMFR\results",
    [string]$PythonExe = "F:\ProgramData\anaconda3\envs\aagmspenv\python.exe",
    [int]$BatchSize = 16,
    [int]$MaxEpochs = 100,
    [int]$NumWorkers = 0,
    [switch]$RebuildPreparedData,
    [string]$TrainingLauncher = "",
    [string]$TestLauncher = ""
)

$ErrorActionPreference = "Stop"
$Seed = 42
$SupportedSamples = @(
    [pscustomobject]@{ Value = 0.01; Label = "1pct"; CacheSuffix = "seed42" },
    [pscustomobject]@{ Value = 0.05; Label = "5pct"; CacheSuffix = "5pct_seed42" },
    [pscustomobject]@{ Value = 0.10; Label = "10pct"; CacheSuffix = "10pct_seed42" }
)
$Selected = $SupportedSamples |
    Where-Object { [Math]::Abs($SamplePercent - $_.Value) -le 1e-12 } |
    Select-Object -First 1
if ($null -eq $Selected) {
    throw "SamplePercent must be one of 0.01, 0.05, or 0.10."
}

if ([string]::IsNullOrWhiteSpace($TrainingLauncher)) {
    $TrainingLauncher = Join-Path $PSScriptRoot "run_mfrinstseg_experiments.ps1"
}
if ([string]::IsNullOrWhiteSpace($TestLauncher)) {
    $TestLauncher = Join-Path $PSScriptRoot "run_mfrinstseg_test.ps1"
}
foreach ($Launcher in @($TrainingLauncher, $TestLauncher)) {
    if (-not (Test-Path -LiteralPath $Launcher -PathType Leaf)) {
        throw "Launcher not found: $Launcher"
    }
}

$SampleLabel = $Selected.Label
$PreparedDatasetPath = "${PreparedDatasetBasePath}_$($Selected.CacheSuffix)"
$Experiments = @(
    [pscustomobject]@{
        Mode = "baseline"
        Mapping = "signed"
        ResultLabel = "baseline"
        FaceGeometryEncoder = "mini_point"
        SurfaceNeighborMetric = "intrinsic"
        SurfaceTrimAware = $false
        SurfaceFusion = "add"
    },
    [pscustomobject]@{
        Mode = "pc_aag"
        Mapping = "signed"
        ResultLabel = "pc_aag_param_surface_intrinsic_untrimmed_gated"
        FaceGeometryEncoder = "param_surface"
        SurfaceNeighborMetric = "intrinsic"
        SurfaceTrimAware = $false
        SurfaceFusion = "gated"
    },
    [pscustomobject]@{
        Mode = "affinity_only"
        Mapping = "positive_only"
        ResultLabel = "affinity_only_positive"
        FaceGeometryEncoder = "mini_point"
        SurfaceNeighborMetric = "intrinsic"
        SurfaceTrimAware = $false
        SurfaceFusion = "add"
    },
    [pscustomobject]@{
        Mode = "instance"
        Mapping = "positive_only"
        ResultLabel = "instance_positive"
        FaceGeometryEncoder = "mini_point"
        SurfaceNeighborMetric = "intrinsic"
        SurfaceTrimAware = $false
        SurfaceFusion = "add"
    }
)

function Get-LatestNormalTestLog {
    param(
        [Parameter(Mandatory = $true)]
        [string]$ResultRoot
    )

    if (-not (Test-Path -LiteralPath $ResultRoot -PathType Container)) {
        return $null
    }
    return Get-ChildItem `
        -LiteralPath $ResultRoot `
        -Recurse `
        -Filter "test_log.txt" `
        -File `
        | Where-Object {
            $_.FullName -notmatch '[\\/]test_bias_scale_0[\\/]'
        } `
        | Sort-Object LastWriteTime -Descending `
        | Select-Object -First 1
}

Write-Host "Seed/sample: $Seed / $SamplePercent ($SampleLabel)"
Write-Host "Prepared dataset: $PreparedDatasetPath"
Write-Host "Shared BIN cache: $SharedBinCachePath"
Write-Host "Experiments: $($Experiments.Count)"

for ($Index = 0; $Index -lt $Experiments.Count; $Index++) {
    $Experiment = $Experiments[$Index]
    $ResultRoot = Join-Path $ResultBasePath (
        "MFRInstSeg_{0}_{1}_seed{2}" -f `
            $SampleLabel, $Experiment.ResultLabel, $Seed
    )
    $NormalLog = $null
    if (-not $RebuildPreparedData) {
        $NormalLog = Get-LatestNormalTestLog -ResultRoot $ResultRoot
    }
    if ($null -ne $NormalLog) {
        Write-Host ""
        Write-Host "Skipping completed experiment: $($Experiment.ResultLabel)"
        Write-Host "Existing test log: $($NormalLog.FullName)"
        continue
    }

    $ExistingCheckpointPaths = @{}
    if (Test-Path -LiteralPath $ResultRoot -PathType Container) {
        Get-ChildItem `
            -LiteralPath $ResultRoot `
            -Recurse `
            -Filter "best-*.ckpt" `
            -File `
            | ForEach-Object {
                $ExistingCheckpointPaths[$_.FullName] = $true
            }
    }

    $TrainingArguments = @{
        Mode = $Experiment.Mode
        InstanceRelationMapping = $Experiment.Mapping
        RawDatasetPath = $RawDatasetPath
        PreparedDatasetPath = $PreparedDatasetPath
        SharedBinCachePath = $SharedBinCachePath
        DonorPreparedDatasetPaths = $DonorPreparedDatasetPaths
        SamplePercent = $SamplePercent
        Seed = $Seed
        BatchSize = $BatchSize
        MaxEpochs = $MaxEpochs
        NumWorkers = $NumWorkers
        PythonExe = $PythonExe
        ResultBasePath = $ResultBasePath
        FaceGeometryEncoder = $Experiment.FaceGeometryEncoder
        SurfaceNeighborMetric = $Experiment.SurfaceNeighborMetric
        SurfaceTrimAware = $Experiment.SurfaceTrimAware
        SurfaceFusion = $Experiment.SurfaceFusion
    }
    if ($RebuildPreparedData -and $Index -eq 0) {
        $TrainingArguments.RebuildPreparedData = $true
    }

    Write-Host ""
    Write-Host "============================================================"
    Write-Host "Training: $($Experiment.ResultLabel)"
    Write-Host "Mode/mapping: $($Experiment.Mode) / $($Experiment.Mapping)"
    Write-Host "Sample/cache: $SampleLabel / $PreparedDatasetPath"
    Write-Host "============================================================"

    & $TrainingLauncher @TrainingArguments
    if (-not $?) {
        throw "Training failed: $($Experiment.ResultLabel)"
    }
    if ($null -ne $LASTEXITCODE -and $LASTEXITCODE -ne 0) {
        throw "Training failed with exit code ${LASTEXITCODE}: $($Experiment.ResultLabel)"
    }

    $Checkpoint = $null
    if (Test-Path -LiteralPath $ResultRoot -PathType Container) {
        $Checkpoint = Get-ChildItem `
            -LiteralPath $ResultRoot `
            -Recurse `
            -Filter "best-*.ckpt" `
            -File `
            | Where-Object {
                -not $ExistingCheckpointPaths.ContainsKey($_.FullName)
            } `
            | Sort-Object LastWriteTime -Descending `
            | Select-Object -First 1 -ExpandProperty FullName
    }
    if (-not $Checkpoint) {
        throw "No fresh best checkpoint found: $($Experiment.ResultLabel)"
    }

    Write-Host "Testing checkpoint: $Checkpoint"
    & $TestLauncher `
        -Checkpoint $Checkpoint `
        -PreparedDataset $PreparedDatasetPath `
        -SharedBinCachePath $SharedBinCachePath `
        -Seed $Seed `
        -BatchSize $BatchSize `
        -NumWorkers $NumWorkers `
        -PythonExe $PythonExe
    if (-not $?) {
        throw "Test failed: $($Experiment.ResultLabel)"
    }
    if ($null -ne $LASTEXITCODE -and $LASTEXITCODE -ne 0) {
        throw "Test failed with exit code ${LASTEXITCODE}: $($Experiment.ResultLabel)"
    }
}

Write-Host ""
Write-Host "Completed seed42 $SampleLabel small-sample experiment matrix."

