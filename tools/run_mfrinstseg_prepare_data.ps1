<#
.SYNOPSIS
    Prepare MFRInstSeg shared BIN cache and per-percentage manifests.

.DESCRIPTION
    Converts raw STEP files to BIN format using a shared cache so that
    1%/5%/10% preparations reuse the same BIN pool without duplicate conversion.

    Run in order: 10% first (builds the cache), then 5% and 1% (incremental).

.PARAMETER SamplePercent
    Which percentage to prepare: 0.10, 0.05, 0.01, or 0 to run all three.

.PARAMETER Rebuild
    Rebuild the prepared manifests even if they already exist. Existing
    shared BIN files are never overwritten.

.PARAMETER NumWorkers
    Number of parallel threads for BIN conversion. Default 1 (serial).
    Recommended: 4-8 depending on CPU cores and memory.

.EXAMPLE
    .\run_mfrinstseg_prepare_data.ps1 -SamplePercent 0

.EXAMPLE
    .\run_mfrinstseg_prepare_data.ps1 -SamplePercent 0.10 -NumWorkers 4
#>

param(
    [Parameter(Mandatory = $true)]
    [double]$SamplePercent,
    [string]$RawDatasetPath = "D:\Code\Datasets\MFRInstSeg",
    [string]$PreparedDatasetBasePath = "D:\Code\Datasets\MFRInstSeg_BrepMFR_ParamSurface_v4",
    [string]$SharedBinCachePath = "D:\Code\Datasets\MFRInstSeg_BrepMFR_ParamSurface_v4_shared_seed42",
    [string]$PythonExe = "F:\ProgramData\anaconda3\envs\aagmspenv\python.exe",
    [int]$Seed = 42,
    [double[]]$SplitRatios = @(0.8, 0.1, 0.1),
    [int]$NumWorkers = 1,
    [switch]$Rebuild
)

$ErrorActionPreference = "Stop"
$PrepareScript = Join-Path $PSScriptRoot "prepare_dataset.py"

if (-not (Test-Path -LiteralPath $PrepareScript -PathType Leaf)) {
    throw "prepare_dataset.py not found: $PrepareScript"
}
if (-not (Test-Path -LiteralPath $PythonExe -PathType Leaf)) {
    throw "Python not found: $PythonExe"
}
if (-not (Test-Path -LiteralPath $RawDatasetPath -PathType Container)) {
    throw "Raw dataset not found: $RawDatasetPath"
}

$PercentMap = @(
    [pscustomobject]@{ Value = 0.10; Label = "10pct"; Suffix = "10pct_seed42" },
    [pscustomobject]@{ Value = 0.05; Label = "5pct";  Suffix = "5pct_seed42"  },
    [pscustomobject]@{ Value = 0.01; Label = "1pct";  Suffix = "seed42"       }
)

# Determine which entries to run
if ($SamplePercent -eq 0) {
    $Entries = $PercentMap
} else {
    $Matched = $PercentMap |
        Where-Object { [Math]::Abs($SamplePercent - $_.Value) -le 1e-12 }
    if ($null -eq $Matched) {
        throw "SamplePercent must be 0 (all), 0.01, 0.05, or 0.10."
    }
    $Entries = @($Matched)
}

Write-Host ""
Write-Host "============================================"
Write-Host " MFRInstSeg Data Preparation"
Write-Host "============================================"
Write-Host "Raw dataset     : $RawDatasetPath"
Write-Host "Shared BIN cache: $SharedBinCachePath"
Write-Host "Seed            : $Seed"
Write-Host "Split ratios    : $SplitRatios"
Write-Host "Num workers     : $NumWorkers"
Write-Host "Percentages     : $($Entries | ForEach-Object { $_.Label })"
Write-Host "============================================"
Write-Host ""

$SplitArgs = $SplitRatios | ForEach-Object { [string]$_ }

foreach ($Entry in $Entries) {
    $PreparedPath = "${PreparedDatasetBasePath}_$($Entry.Suffix)"

    Write-Host ""
    Write-Host "------------------------------------------------------------"
    Write-Host "Preparing $($Entry.Label) ($($Entry.Value))"
    Write-Host "Output : $PreparedPath"
    Write-Host "------------------------------------------------------------"

    $Arguments = @(
        $PrepareScript,
        "--raw-dataset", $RawDatasetPath,
        "--prepared-dataset", $PreparedPath,
        "--shared-bin-cache", $SharedBinCachePath,
        "--seed", $Seed,
        "--sample-percent", $Entry.Value,
        "--split-ratios"
    ) + $SplitArgs

    if ($NumWorkers -gt 1) {
        $Arguments += "--num-workers"
        $Arguments += $NumWorkers
    }

    if ($Rebuild) {
        $Arguments += "--rebuild"
    }

    & $PythonExe @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Preparation failed: $($Entry.Label)"
    }

    # Quick validation
    $TrainFile = Join-Path $PreparedPath "train.txt"
    $ValFile   = Join-Path $PreparedPath "val.txt"
    $TestFile  = Join-Path $PreparedPath "test.txt"
    $FailFile  = Join-Path $SharedBinCachePath "conversion_failures.json"

    $TrainCount = if (Test-Path $TrainFile) { (Get-Content $TrainFile | Measure-Object -Line).Lines } else { 0 }
    $ValCount   = if (Test-Path $ValFile)   { (Get-Content $ValFile   | Measure-Object -Line).Lines } else { 0 }
    $TestCount  = if (Test-Path $TestFile)  { (Get-Content $TestFile  | Measure-Object -Line).Lines } else { 0 }
    $FailCount  = if (Test-Path $FailFile)  { (Get-Content $FailFile  | Measure-Object -Line).Lines } else { 0 }

    Write-Host ""
    Write-Host "Results for $($Entry.Label):"
    Write-Host "  train : $TrainCount"
    Write-Host "  val   : $ValCount"
    Write-Host "  test  : $TestCount"
    Write-Host "  failures: $FailCount"
    if ($FailCount -gt 0) {
        Write-Warning "$FailCount conversion failures in $($Entry.Label). Check $FailFile"
    }
}

# Final summary
Write-Host ""
Write-Host "============================================"
Write-Host " Data Preparation Complete"
Write-Host "============================================"
if (Test-Path $SharedBinCachePath) {
    $BinDir = Join-Path $SharedBinCachePath "bin"
    $BinCount = if (Test-Path $BinDir) { (Get-ChildItem $BinDir -File -ErrorAction SilentlyContinue | Measure-Object).Count } else { 0 }
    Write-Host "Shared BIN pool: $BinCount files"
    Write-Host "Manifest       : $(Test-Path (Join-Path $SharedBinCachePath 'bin_cache_manifest.json'))"
}
$FailFile = Join-Path $SharedBinCachePath "conversion_failures.json"
if (Test-Path $FailFile) {
    $FailTotal = (Get-Content $FailFile | Measure-Object -Line).Lines
    Write-Host "Total failures : $FailTotal"
} else {
    Write-Host "Total failures : 0"
}
Write-Host "============================================"

