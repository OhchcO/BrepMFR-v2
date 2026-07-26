param(
    [ValidateSet("baseline", "oracle", "dynamic", "dynamic_affinity")]
    [string]$Mode = "baseline",
    [switch]$All,
    [string]$DatasetPath = "D:\Code\Datasets\MFRInstSeg_BrepMFR",
    [string]$PythonExe = "G:\ProgramData\anaconda3\envs\aagmspenv\python.exe",
    [int]$BatchSize = 16,
    [int]$MaxEpochs = 1000,
    [int]$Seed = 2026,
    [int]$NumWorkers = 0,
    [double]$InstanceBiasScale = 1.0,
    [int]$AffinityDim = 64,
    [double]$AffinityTemperature = 0.2,
    [double]$AffinityLossWeight = 0.1,
    [int]$AffinityWarmupEpochs = 10
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Segmentation = Join-Path $ProjectRoot "segmentation.py"

if (-not (Test-Path -LiteralPath $PythonExe -PathType Leaf)) {
    throw "Python executable not found: $PythonExe"
}
if (-not (Test-Path -LiteralPath $Segmentation -PathType Leaf)) {
    throw "Training entry not found: $Segmentation"
}
foreach ($Manifest in @("train.txt", "val.txt")) {
    $ManifestPath = Join-Path $DatasetPath $Manifest
    if (-not (Test-Path -LiteralPath $ManifestPath -PathType Leaf)) {
        throw "Dataset manifest not found: $ManifestPath"
    }
}

if ($All) {
    $Modes = @("baseline", "oracle", "dynamic", "dynamic_affinity")
}
else {
    $Modes = @($Mode)
}

Push-Location $ProjectRoot
try {
    foreach ($CurrentMode in $Modes) {
        $ExperimentName = "MFRInstSeg_1pct_${CurrentMode}_seed${Seed}"
        Write-Host ""
        Write-Host "============================================================"
        Write-Host "Starting experiment: $CurrentMode"
        Write-Host "Result name: $ExperimentName"
        Write-Host "============================================================"

        $Arguments = @(
            $Segmentation,
            "train",
            "--dataset_path", $DatasetPath,
            "--num_classes", "25",
            "--batch_size", "$BatchSize",
            "--num_workers", "$NumWorkers",
            "--max_epochs", "$MaxEpochs",
            "--seed", "$Seed",
            "--experiment_name", $ExperimentName,
            "--experiment_mode", $CurrentMode,
            "--instance_bias_scale", "$InstanceBiasScale",
            "--affinity_dim", "$AffinityDim",
            "--affinity_temperature", "$AffinityTemperature",
            "--affinity_loss_weight", "$AffinityLossWeight",
            "--affinity_warmup_epochs", "$AffinityWarmupEpochs"
        )

        & $PythonExe @Arguments
        if ($LASTEXITCODE -ne 0) {
            throw "Experiment '$CurrentMode' failed with exit code $LASTEXITCODE"
        }
    }
}
finally {
    Pop-Location
}
