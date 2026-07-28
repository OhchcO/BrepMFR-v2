param(
    [Parameter(Mandatory = $true)]
    [ValidateRange(0, 2147483647)]
    [int]$Seed,
    [string]$RawDatasetPath = "D:\Code\Datasets\MFRInstSeg",
    [string]$PreparedDatasetBase = "D:\Code\Datasets\MFRInstSeg_BrepMFR_SPEPCAAG",
    [string]$ResultBasePath = "D:\Code\BrepMFR\results",
    [string]$PythonExe = "G:\ProgramData\anaconda3\envs\aagmspenv\python.exe",
    [string]$TrainingLauncher = "",
    [string]$TestLauncher = ""
)

$ErrorActionPreference = "Stop"

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

$PreparedDatasetPath = "${PreparedDatasetBase}_seed${Seed}"
$Experiments = @(
    [pscustomobject]@{
        Mode = "baseline"
        Mapping = $null
        ResultLabel = "baseline"
    },
    [pscustomobject]@{
        Mode = "dynamic_no_loss"
        Mapping = "positive_only"
        ResultLabel = "dynamic_no_loss_positive"
    },
    [pscustomobject]@{
        Mode = "instance"
        Mapping = "positive_only"
        ResultLabel = "instance_positive"
    }
)

Write-Host "Seed: $Seed"
Write-Host "Prepared dataset: $PreparedDatasetPath"

foreach ($Experiment in $Experiments) {
    $StartedAt = Get-Date
    $TrainingArguments = @{
        Mode = $Experiment.Mode
        RawDatasetPath = $RawDatasetPath
        PreparedDatasetPath = $PreparedDatasetPath
        SamplePercent = 0.01
        Seed = $Seed
        BatchSize = 16
        MaxEpochs = 100
        NumWorkers = 0
        PythonExe = $PythonExe
    }
    if ($null -ne $Experiment.Mapping) {
        $TrainingArguments.InstanceRelationMapping = $Experiment.Mapping
    }

    Write-Host ""
    Write-Host "Training: $($Experiment.ResultLabel)"
    & $TrainingLauncher @TrainingArguments
    if (-not $?) {
        throw "Training failed: $($Experiment.ResultLabel)"
    }
    if ($null -ne $LASTEXITCODE -and $LASTEXITCODE -ne 0) {
        throw "Training failed with exit code ${LASTEXITCODE}: $($Experiment.ResultLabel)"
    }

    $ResultRoot = Join-Path $ResultBasePath (
        "MFRInstSeg_1pct_{0}_seed{1}" -f $Experiment.ResultLabel, $Seed
    )
    $FreshnessBoundary = $StartedAt.AddSeconds(-2)
    $Checkpoint = $null
    if (Test-Path -LiteralPath $ResultRoot -PathType Container) {
        $Checkpoint = Get-ChildItem `
            -LiteralPath $ResultRoot `
            -Recurse `
            -Filter "best-*.ckpt" `
            -File `
            | Where-Object { $_.LastWriteTime -ge $FreshnessBoundary } `
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
        -BatchSize 16 `
        -NumWorkers 0 `
        -PythonExe $PythonExe
    if (-not $?) {
        throw "Test failed: $($Experiment.ResultLabel)"
    }
    if ($null -ne $LASTEXITCODE -and $LASTEXITCODE -ne 0) {
        throw "Test failed with exit code ${LASTEXITCODE}: $($Experiment.ResultLabel)"
    }
}

Write-Host ""
Write-Host "Completed all core experiments for seed $Seed."
