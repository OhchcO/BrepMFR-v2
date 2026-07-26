param(
    [Parameter(Mandatory = $true)]
    [string]$Checkpoint,
    [string]$SourceDataset = "D:\Code\Datasets\MFRInstSeg",
    [string]$BinDataset = "D:\Code\Datasets\MFRInstSeg_BrepMFR",
    [string]$PythonExe = "G:\ProgramData\anaconda3\envs\aagmspenv\python.exe",
    [string]$ExtractorFile = "D:\Code\BrepMFR-g\data_set_line\AAGExtractor.py",
    [string]$SchemaFile = "D:\Code\BrepMFR-g\data_set_line\schema.json",
    [int]$BatchSize = 16,
    [int]$NumWorkers = 0,
    [int]$TestLimit = 0,
    [switch]$SkipConversion,
    [switch]$OverwriteBins
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Converter = Join-Path $PSScriptRoot "mfrinstseg_to_bin.py"
$Evaluator = Join-Path $PSScriptRoot "test_mfrinstseg.py"
$OfficialTestManifest = Join-Path $SourceDataset "test.txt"

foreach ($RequiredFile in @(
    $PythonExe,
    $Checkpoint,
    $Converter,
    $Evaluator,
    $OfficialTestManifest
)) {
    if (-not (Test-Path -LiteralPath $RequiredFile -PathType Leaf)) {
        throw "Required file not found: $RequiredFile"
    }
}

Push-Location $ProjectRoot
try {
    if (-not $SkipConversion) {
        Write-Host "Converting the official MFRInstSeg test.txt split to BIN..."
        $ConversionArguments = @(
            $Converter,
            "--dataset-root", $SourceDataset,
            "--subset-root", $SourceDataset,
            "--output-root", $BinDataset,
            "--extractor-file", $ExtractorFile,
            "--schema-file", $SchemaFile,
            "--split", "test"
        )
        if ($TestLimit -gt 0) {
            $ConversionArguments += @("--limit", "$TestLimit")
        }
        if ($OverwriteBins) {
            $ConversionArguments += "--overwrite"
        }

        & $PythonExe @ConversionArguments
        if ($LASTEXITCODE -ne 0) {
            throw "Test BIN conversion failed with exit code $LASTEXITCODE. Check conversion_failures.json."
        }
    }

    $ConvertedTestManifest = Join-Path $BinDataset "test.txt"
    if (-not (Test-Path -LiteralPath $ConvertedTestManifest -PathType Leaf)) {
        throw "Converted test manifest not found: $ConvertedTestManifest"
    }

    Write-Host "Evaluating checkpoint on the official test split..."
    & $PythonExe $Evaluator `
        --dataset-path $BinDataset `
        --checkpoint $Checkpoint `
        --split test `
        --batch-size $BatchSize `
        --num-workers $NumWorkers
    if ($LASTEXITCODE -ne 0) {
        throw "Test evaluation failed with exit code $LASTEXITCODE"
    }
}
finally {
    Pop-Location
}
