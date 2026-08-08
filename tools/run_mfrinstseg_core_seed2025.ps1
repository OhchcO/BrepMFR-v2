param(
    [string]$RawDatasetPath = "D:\Code\Datasets\MFRInstSeg",
    [string]$PreparedDatasetBase = "D:\Code\Datasets\MFRInstSeg_BrepMFR_SPEPCAAG10D",
    [string]$ResultBasePath = "D:\Code\BrepMFR\results",
    [string]$PythonExe = "F:\ProgramData\anaconda3\envs\aagmspenv\python.exe",
    [string]$TrainingLauncher = "",
    [string]$TestLauncher = ""
)

$ErrorActionPreference = "Stop"
$Runner = Join-Path $PSScriptRoot "run_mfrinstseg_core_seed.ps1"
& $Runner -Seed 2025 @PSBoundParameters
if (-not $?) {
    throw "Seed 2025 experiment matrix failed"
}

