param(
    [string]$RawDatasetPath = "D:\Code\Datasets\MFRInstSeg",
    [string]$PreparedDatasetPath = "D:\Code\Datasets\MFRInstSeg_BrepMFR_ParamSurface_v4_seed42",
    [string]$ResultBasePath = "D:\Code\BrepMFR\results",
    [string]$PythonExe = "F:\ProgramData\anaconda3\envs\aagmspenv\python.exe",
    [int]$BatchSize = 16,
    [int]$MaxEpochs = 100,
    [int]$NumWorkers = 0,
    [double]$SamplePercent = 0.01,
    [switch]$RebuildPreparedData,
    [string]$TrainingLauncher = "",
    [string]$TestLauncher = ""
)

$ErrorActionPreference = "Stop"
$Seed = 42

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

$Experiments = @(
    [pscustomobject]@{
        Mode = "pc_aag"
        NeighborMetric = "intrinsic"
        TrimAware = $false
        Fusion = "add"
        ResultLabel = "pc_aag_param_surface_intrinsic_untrimmed_add"
    },
    [pscustomobject]@{
        Mode = "pc_aag"
        NeighborMetric = "intrinsic"
        TrimAware = $true
        Fusion = "add"
        ResultLabel = "pc_aag_param_surface_intrinsic_trim_add"
    },
    [pscustomobject]@{
        Mode = "pc_aag"
        NeighborMetric = "intrinsic"
        TrimAware = $true
        Fusion = "gated"
        ResultLabel = "pc_aag_param_surface_intrinsic_trim_gated"
    },
    [pscustomobject]@{
        Mode = "pc_aag"
        NeighborMetric = "xyz"
        TrimAware = $false
        Fusion = "add"
        ResultLabel = "pc_aag_param_surface_xyz_untrimmed_add"
    },
    [pscustomobject]@{
        Mode = "pc_aag"
        NeighborMetric = "intrinsic"
        TrimAware = $false
        Fusion = "gated"
        ResultLabel = "pc_aag_param_surface_intrinsic_untrimmed_gated"
    }
)

Write-Host "Seed: $Seed"
Write-Host "Prepared dataset: $PreparedDatasetPath"
Write-Host "Experiments: $($Experiments.Count)"

for ($Index = 0; $Index -lt $Experiments.Count; $Index++) {
    $Experiment = $Experiments[$Index]
    $ResultRoot = Join-Path $ResultBasePath (
        "MFRInstSeg_1pct_{0}_seed{1}" -f $Experiment.ResultLabel, $Seed
    )
    $CompletedTestLog = $null
    if (
        -not $RebuildPreparedData `
        -and (Test-Path -LiteralPath $ResultRoot -PathType Container)
    ) {
        $CompletedTestLog = Get-ChildItem `
            -LiteralPath $ResultRoot `
            -Recurse `
            -Filter "test_log.txt" `
            -File `
            | Sort-Object LastWriteTime -Descending `
            | Select-Object -First 1
    }
    if ($null -ne $CompletedTestLog) {
        Write-Host ""
        Write-Host "Skipping completed experiment: $($Experiment.ResultLabel)"
        Write-Host "Existing test log: $($CompletedTestLog.FullName)"
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
        RawDatasetPath = $RawDatasetPath
        PreparedDatasetPath = $PreparedDatasetPath
        SamplePercent = $SamplePercent
        Seed = $Seed
        BatchSize = $BatchSize
        MaxEpochs = $MaxEpochs
        NumWorkers = $NumWorkers
        PythonExe = $PythonExe
        ResultBasePath = $ResultBasePath
        FaceGeometryEncoder = "param_surface"
        SurfaceNeighborMetric = $Experiment.NeighborMetric
        SurfaceTrimAware = $Experiment.TrimAware
        SurfaceFusion = $Experiment.Fusion
    }
    if ($RebuildPreparedData -and $Index -eq 0) {
        $TrainingArguments.RebuildPreparedData = $true
    }

    Write-Host ""
    Write-Host "============================================================"
    Write-Host "Training: $($Experiment.ResultLabel)"
    Write-Host "Trim-aware: $($Experiment.TrimAware)"
    Write-Host "Fusion: $($Experiment.Fusion)"
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
Write-Host "Completed all seed42 parameter-surface screening experiments."

