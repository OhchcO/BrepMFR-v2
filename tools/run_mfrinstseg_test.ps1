param(
    [Parameter(Mandatory = $true)]
    [string]$Checkpoint,
    [string]$PreparedDataset = "D:\Code\Datasets\MFRInstSeg_BrepMFR_SPEPCAAG",
    [string]$PythonExe = "G:\ProgramData\anaconda3\envs\aagmspenv\python.exe",
    [int]$Seed = 2026,
    [int]$BatchSize = 16,
    [int]$NumWorkers = 0,
    [bool]$EvaluateInstances = $true,
    [double]$AffinityThreshold = 0.5,
    [int]$BackgroundClassId = 24,
    [Nullable[double]]$InstanceBiasScaleOverride = $null,
    [switch]$Prepare,
    [switch]$RebuildPreparedData,
    [string]$RawDatasetPath = "D:\Code\Datasets\MFRInstSeg",
    [string]$ExtractorFile = "D:\Code\BrepMFR-g\data_set_line\AAGExtractor.py",
    [string]$SchemaFile = "D:\Code\BrepMFR-g\data_set_line\schema.json",
    [double]$SamplePercent = 0.01,
    [double[]]$SplitRatios = @(0.8, 0.1, 0.1),
    [ValidateSet("fixed", "area_adaptive")]
    [string]$PointSamplingMode = "area_adaptive",
    [int]$FixedPointCount = 64,
    [int]$MinPointCount = 32,
    [int]$MaxPointCount = 96,
    [double]$AreaSamplingExponent = 0.5,
    [int[]]$CandidateGrids = @(20, 32, 48, 64, 80)
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Preparer = Join-Path $PSScriptRoot "prepare_dataset.py"
$Evaluator = Join-Path $PSScriptRoot "test_mfrinstseg.py"

foreach ($RequiredFile in @(
    $PythonExe,
    $Checkpoint,
    $Evaluator
)) {
    if (-not (Test-Path -LiteralPath $RequiredFile -PathType Leaf)) {
        throw "Required file not found: $RequiredFile"
    }
}

if ($RebuildPreparedData -and -not $Prepare) {
    throw "-RebuildPreparedData requires -Prepare."
}
if (
    $null -ne $InstanceBiasScaleOverride -and (
        [double]::IsNaN($InstanceBiasScaleOverride) -or
        [double]::IsInfinity($InstanceBiasScaleOverride) -or
        $InstanceBiasScaleOverride -lt 0.0
    )
) {
    throw "-InstanceBiasScaleOverride must be finite and non-negative."
}

Push-Location $ProjectRoot
try {
    if ($Prepare) {
        foreach ($Directory in @(
            $RawDatasetPath,
            (Join-Path $RawDatasetPath "steps"),
            (Join-Path $RawDatasetPath "labels")
        )) {
            if (-not (Test-Path -LiteralPath $Directory -PathType Container)) {
                throw "Raw dataset directory not found: $Directory"
            }
        }
        foreach ($RequiredFile in @($Preparer, $ExtractorFile, $SchemaFile)) {
            if (-not (Test-Path -LiteralPath $RequiredFile -PathType Leaf)) {
                throw "Preparation input not found: $RequiredFile"
            }
        }

        Write-Host "Preparing or reusing the shared fingerprinted dataset cache..."
        $PreparationArguments = @(
            $Preparer,
            "--raw-dataset", $RawDatasetPath,
            "--prepared-dataset", $PreparedDataset,
            "--extractor-file", $ExtractorFile,
            "--schema-file", $SchemaFile,
            "--seed", "$Seed",
            "--sample-percent", "$SamplePercent",
            "--split-ratios"
        ) + @($SplitRatios | ForEach-Object { "$_" }) + @(
            "--point-sampling-mode", $PointSamplingMode,
            "--fixed-point-count", "$FixedPointCount",
            "--min-point-count", "$MinPointCount",
            "--max-point-count", "$MaxPointCount",
            "--area-sampling-exponent", "$AreaSamplingExponent",
            "--candidate-grids"
        ) + @($CandidateGrids | ForEach-Object { "$_" })
        if ($RebuildPreparedData) {
            $PreparationArguments += "--rebuild"
        }

        & $PythonExe @PreparationArguments
        if ($LASTEXITCODE -ne 0) {
            throw "Dataset preparation failed with exit code $LASTEXITCODE. Check conversion_failures.json."
        }
    }

    $PreparedTestManifest = Join-Path $PreparedDataset "test.txt"
    if (-not (Test-Path -LiteralPath $PreparedTestManifest -PathType Leaf)) {
        throw "Prepared test manifest not found: $PreparedTestManifest. Run training auto-preparation or use -Prepare."
    }

    Write-Host "Evaluating checkpoint on every entry in the prepared test.txt split..."
    $EvaluationArguments = @(
        $Evaluator,
        "--dataset-path", $PreparedDataset,
        "--checkpoint", $Checkpoint,
        "--split", "test",
        "--seed", "$Seed",
        "--batch-size", "$BatchSize",
        "--num-workers", "$NumWorkers",
        "--affinity-threshold", "$AffinityThreshold",
        "--background-class-id", "$BackgroundClassId"
    )
    if ($EvaluateInstances) {
        $EvaluationArguments += "--evaluate-instances"
    }
    if ($null -ne $InstanceBiasScaleOverride) {
        $EvaluationArguments += @(
            "--instance-bias-scale-override",
            $InstanceBiasScaleOverride.ToString(
                [System.Globalization.CultureInfo]::InvariantCulture
            )
        )
    }
    & $PythonExe @EvaluationArguments
    if ($LASTEXITCODE -ne 0) {
        throw "Test evaluation failed with exit code $LASTEXITCODE"
    }
}
finally {
    Pop-Location
}
