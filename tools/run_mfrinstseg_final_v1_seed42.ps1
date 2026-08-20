param(
    [string]$RawDatasetPath = "D:\Code\Datasets\MFRInstSeg",
    [string]$ParentPreparedDatasetPath = "D:\Code\Datasets\MFRInstSeg_BrepMFR_ParamSurface_v4_10pct_seed42",
    [string]$PreparedDatasetPath = "D:\Code\Datasets\MFRInstSeg_BrepMFR_ParamSurface_v4_1pct_seed42",
    [string]$SharedBinCachePath = "D:\Code\Datasets\MFRInstSeg_BrepMFR_ParamSurface_v4_shared_seed42",
    [string]$ResultBasePath = "D:\Code\BrepMFR\results\final_v1",
    [string]$PythonExe = "F:\ProgramData\anaconda3\envs\aagmspenv\python.exe",
    [int]$BatchSize = 16,
    [int]$MaxEpochs = 100,
    [int]$NumWorkers = 0,
    [switch]$RebuildSubset,
    [switch]$PrepareOnly,
    [string]$SubsetBuilder = "",
    [string]$TrainingLauncher = "",
    [string]$TestLauncher = ""
)

$ErrorActionPreference = "Stop"
$Seed = 42
$SamplePercent = 0.01

if ([string]::IsNullOrWhiteSpace($SubsetBuilder)) {
    $SubsetBuilder = Join-Path $PSScriptRoot "derive_nested_mfrinstseg_subset.py"
}
if ([string]::IsNullOrWhiteSpace($TrainingLauncher)) {
    $TrainingLauncher = Join-Path $PSScriptRoot "run_mfrinstseg_experiments.ps1"
}
if ([string]::IsNullOrWhiteSpace($TestLauncher)) {
    $TestLauncher = Join-Path $PSScriptRoot "run_mfrinstseg_test.ps1"
}

foreach ($RequiredFile in @($PythonExe, $SubsetBuilder, $TrainingLauncher, $TestLauncher)) {
    if (-not (Test-Path -LiteralPath $RequiredFile -PathType Leaf)) {
        throw "Required file not found: $RequiredFile"
    }
}
foreach ($RequiredDirectory in @(
    $RawDatasetPath,
    $ParentPreparedDatasetPath,
    $SharedBinCachePath,
    (Join-Path $SharedBinCachePath "bin")
)) {
    if (-not (Test-Path -LiteralPath $RequiredDirectory -PathType Container)) {
        throw "Required directory not found: $RequiredDirectory"
    }
}

$SubsetArguments = @(
    $SubsetBuilder,
    "--raw-dataset", $RawDatasetPath,
    "--parent-prepared-dataset", $ParentPreparedDatasetPath,
    "--output-prepared-dataset", $PreparedDatasetPath,
    "--shared-bin-cache", $SharedBinCachePath,
    "--sample-percent", "$SamplePercent",
    "--seed", "$Seed"
)
if ($RebuildSubset) {
    $SubsetArguments += "--rebuild"
}

Write-Host "============================================================"
Write-Host "Preparing nested Final v1 subset"
Write-Host "Parent:  $ParentPreparedDatasetPath"
Write-Host "Output:  $PreparedDatasetPath"
Write-Host "BIN root: $SharedBinCachePath"
Write-Host "Seed/sample: $Seed / $SamplePercent"
Write-Host "============================================================"

& $PythonExe @SubsetArguments
if ($LASTEXITCODE -ne 0) {
    throw "Nested subset preparation failed with exit code $LASTEXITCODE"
}

foreach ($ManifestName in @("train.txt", "val.txt", "test.txt", "preparation_manifest.json")) {
    $ManifestPath = Join-Path $PreparedDatasetPath $ManifestName
    if (-not (Test-Path -LiteralPath $ManifestPath -PathType Leaf)) {
        throw "Prepared dataset file not found after derivation: $ManifestPath"
    }
}

if ($PrepareOnly) {
    Write-Host "Nested 1% dataset is ready. PrepareOnly requested; training was not started."
    return
}

$Experiments = @(
    [pscustomobject]@{
        Mode = "topo_aag"
        Mapping = "signed"
        ResultLabel = "topo_aag"
        FaceGeometryEncoder = "mini_point"
        SurfaceNeighborMetric = "intrinsic"
        SurfaceTrimAware = $false
        SurfaceFusion = "add"
        Description = "AAG-only internal baseline"
    },
    [pscustomobject]@{
        Mode = "uv_aag"
        Mapping = "signed"
        ResultLabel = "uv_aag"
        FaceGeometryEncoder = "mini_point"
        SurfaceNeighborMetric = "intrinsic"
        SurfaceTrimAware = $false
        SurfaceFusion = "add"
        Description = "Original UV+AAG baseline"
    },
    [pscustomobject]@{
        Mode = "pc_aag"
        Mapping = "signed"
        ResultLabel = "pc_aag_param_surface_intrinsic_untrimmed_gated"
        FaceGeometryEncoder = "param_surface"
        SurfaceNeighborMetric = "intrinsic"
        SurfaceTrimAware = $false
        SurfaceFusion = "gated"
        Description = "Parametric surface+AAG"
    },
    [pscustomobject]@{
        Mode = "cyl_aag"
        Mapping = "signed"
        ResultLabel = "cyl_aag"
        FaceGeometryEncoder = "mini_point"
        SurfaceNeighborMetric = "intrinsic"
        SurfaceTrimAware = $false
        SurfaceFusion = "add"
        Description = "Cylindrical position+AAG"
    },
    [pscustomobject]@{
        Mode = "spe_pcaag"
        Mapping = "signed"
        ResultLabel = "spe_pcaag_param_surface_intrinsic_untrimmed_gated"
        FaceGeometryEncoder = "param_surface"
        SurfaceNeighborMetric = "intrinsic"
        SurfaceTrimAware = $false
        SurfaceFusion = "gated"
        Description = "Parametric surface+cylindrical position+AAG"
    },
    [pscustomobject]@{
        Mode = "spe_affinity_only"
        Mapping = "positive_only"
        ResultLabel = "spe_affinity_only_positive_param_surface_intrinsic_untrimmed_gated"
        FaceGeometryEncoder = "param_surface"
        SurfaceNeighborMetric = "intrinsic"
        SurfaceTrimAware = $false
        SurfaceFusion = "gated"
        Description = "Complete input+affinity supervision"
    },
    [pscustomobject]@{
        Mode = "spe_instance"
        Mapping = "positive_only"
        ResultLabel = "spe_instance_positive_param_surface_intrinsic_untrimmed_gated"
        FaceGeometryEncoder = "param_surface"
        SurfaceNeighborMetric = "intrinsic"
        SurfaceTrimAware = $false
        SurfaceFusion = "gated"
        Description = "Complete input+IAAB (Ours)"
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

Write-Host ""
Write-Host "Final v1 experiment matrix: $($Experiments.Count) runs"
Write-Host "Results: $ResultBasePath"

for ($Index = 0; $Index -lt $Experiments.Count; $Index++) {
    $Experiment = $Experiments[$Index]
    $ResultRoot = Join-Path $ResultBasePath (
        "MFRInstSeg_1pct_{0}_seed42" -f $Experiment.ResultLabel
    )
    $CompletedLog = Get-LatestNormalTestLog -ResultRoot $ResultRoot
    if ($null -ne $CompletedLog) {
        Write-Host ""
        Write-Host "Skipping completed experiment [$($Index + 1)/7]: $($Experiment.Description)"
        Write-Host "Existing test log: $($CompletedLog.FullName)"
        continue
    }

    $ExistingCheckpointPaths = @{}
    if (Test-Path -LiteralPath $ResultRoot -PathType Container) {
        Get-ChildItem `
            -LiteralPath $ResultRoot `
            -Recurse `
            -Filter "best-mIoU-*.ckpt" `
            -File `
            | ForEach-Object {
                $ExistingCheckpointPaths[$_.FullName] = $true
            }
    }

    Write-Host ""
    Write-Host "============================================================"
    Write-Host "Training [$($Index + 1)/7]: $($Experiment.Description)"
    Write-Host "Mode/mapping: $($Experiment.Mode) / $($Experiment.Mapping)"
    Write-Host "============================================================"

    $TrainingArguments = @{
        Mode = $Experiment.Mode
        NoAutoPrepare = $true
        DatasetPath = $PreparedDatasetPath
        PreparedDatasetPath = $PreparedDatasetPath
        SharedBinCachePath = $SharedBinCachePath
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
        InstanceRelationMapping = $Experiment.Mapping
    }
    & $TrainingLauncher @TrainingArguments
    if (-not $?) {
        throw "Training failed: $($Experiment.Description)"
    }
    if ($null -ne $LASTEXITCODE -and $LASTEXITCODE -ne 0) {
        throw "Training failed with exit code ${LASTEXITCODE}: $($Experiment.Description)"
    }

    $Checkpoint = $null
    if (Test-Path -LiteralPath $ResultRoot -PathType Container) {
        $Checkpoint = Get-ChildItem `
            -LiteralPath $ResultRoot `
            -Recurse `
            -Filter "best-mIoU-*.ckpt" `
            -File `
            | Where-Object {
                -not $ExistingCheckpointPaths.ContainsKey($_.FullName)
            } `
            | Sort-Object LastWriteTime -Descending `
            | Select-Object -First 1 -ExpandProperty FullName
    }
    if (-not $Checkpoint) {
        throw "No fresh best-mIoU checkpoint found: $($Experiment.Description)"
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
        throw "Test failed: $($Experiment.Description)"
    }
    if ($null -ne $LASTEXITCODE -and $LASTEXITCODE -ne 0) {
        throw "Test failed with exit code ${LASTEXITCODE}: $($Experiment.Description)"
    }
}

Write-Host ""
Write-Host "Completed all seven Seed 42 Final v1 experiments."
