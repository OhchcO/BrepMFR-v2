param(
    [string]$RawDatasetPath = "D:\Code\Datasets\MFRInstSeg",
    [string]$PreparedDatasetBase = "D:\Code\Datasets\MFRInstSeg_BrepMFR_SPEPCAAG",
    [string]$ResultBasePath = "D:\Code\BrepMFR\results",
    [string]$PythonExe = "G:\ProgramData\anaconda3\envs\aagmspenv\python.exe",
    [string]$TrainingLauncher = "",
    [string]$TestLauncher = ""
)

$ErrorActionPreference = "Stop"
$Runner = Join-Path $PSScriptRoot "run_mfrinstseg_core_seed.ps1"
& $Runner -Seed 2027 @PSBoundParameters

$ExperimentRunner = if ([string]::IsNullOrWhiteSpace($TrainingLauncher)) {
    Join-Path $PSScriptRoot "run_mfrinstseg_experiments.ps1"
} else {
    $TrainingLauncher
}
& $ExperimentRunner `
    -Mode affinity_only `
    -InstanceRelationMapping positive_only `
    -RawDatasetPath $RawDatasetPath `
    -PreparedDatasetPath "${PreparedDatasetBase}_seed2027" `
    -SamplePercent 0.01 `
    -Seed 2027 `
    -BatchSize 16 `
    -MaxEpochs 100 `
    -NumWorkers 0 `
    -PythonExe $PythonExe
