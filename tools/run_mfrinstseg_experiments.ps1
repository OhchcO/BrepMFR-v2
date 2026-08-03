param(
    [ValidateSet("baseline", "topo_aag", "uv_aag", "pc_aag", "cyl_aag", "uv_pc_aag", "dynamic_no_loss", "affinity_only", "instance", "oracle", "cyl", "cyl_affinity", "point", "spe_pcaag", "spe_instance", "full")]
    [string]$Mode = "baseline",
    [switch]$All,
    [bool]$AutoPrepare = $true,
    [switch]$RebuildPreparedData,
    [string]$DatasetPath = "D:\Code\Datasets\MFRInstSeg_BrepMFR",
    [string]$RawDatasetPath = "D:\Code\Datasets\MFRInstSeg",
    [string]$PreparedDatasetPath = "D:\Code\Datasets\MFRInstSeg_BrepMFR_SPEPCAAG",
    [string]$ExtractorFile = "D:\Code\BrepMFR-g\data_set_line\AAGExtractor.py",
    [string]$SchemaFile = "D:\Code\BrepMFR-g\data_set_line\schema.json",
    [string]$PythonExe = "G:\ProgramData\anaconda3\envs\aagmspenv\python.exe",
    [int]$BatchSize = 16,
    [int]$MaxEpochs = 100,
    [int]$Seed = 2026,
    [int]$NumWorkers = 0,
    [double]$SamplePercent = 0.01,
    [double[]]$SplitRatios = @(0.8, 0.1, 0.1),
    [ValidateSet("fixed", "area_adaptive")]
    [string]$PointSamplingMode = "area_adaptive",
    [int]$FixedPointCount = 64,
    [int]$MinPointCount = 32,
    [int]$MaxPointCount = 96,
    [double]$AreaSamplingExponent = 0.5,
    [int[]]$CandidateGrids = @(20, 32, 48, 64, 80),
    [double]$PointcloudScale = 1.0,
    [double]$CylindricalScale = 1.0,
    [ValidateSet("mini_point", "param_surface")]
    [string]$FaceGeometryEncoder = "mini_point",
    [ValidateSet("xyz", "uv", "intrinsic")]
    [string]$SurfaceNeighborMetric = "intrinsic",
    [bool]$SurfaceTrimAware = $false,
    [ValidateSet("add", "gated")]
    [string]$SurfaceFusion = "add",
    [int]$AffinityDim = 64,
    [double]$AffinityTemperature = 0.2,
    [double]$AffinityLossWeight = 0.1,
    [int]$AffinityWarmupEpochs = 10,
    [ValidateSet("signed", "positive_only")]
    [string]$InstanceRelationMapping = "signed",
    [int]$InstanceBiasLastLayers = 2,
    [double]$InstanceBiasConfidencePower = 1.0,
    [double]$InstanceBiasGateInit = 0.0,
    [double]$InstanceBiasMaxGateScale = 1.0,
    [int]$InstanceBiasDelayEpochs = 10,
    [int]$InstanceBiasRampEpochs = 10,
    [string]$ResultBasePath = "D:\Code\BrepMFR\results"
)

$ErrorActionPreference = "Stop"
$env:CUBLAS_WORKSPACE_CONFIG = ":4096:8"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Segmentation = Join-Path $ProjectRoot "segmentation.py"

if (-not (Test-Path -LiteralPath $PythonExe -PathType Leaf)) {
    throw "Python executable not found: $PythonExe"
}
if (-not (Test-Path -LiteralPath $Segmentation -PathType Leaf)) {
    throw "Training entry not found: $Segmentation"
}

if ($AutoPrepare) {
    foreach ($Directory in @($RawDatasetPath, (Join-Path $RawDatasetPath "steps"), (Join-Path $RawDatasetPath "labels"))) {
        if (-not (Test-Path -LiteralPath $Directory -PathType Container)) {
            throw "Raw dataset directory not found: $Directory"
        }
    }
    foreach ($RequiredFile in @($ExtractorFile, $SchemaFile)) {
        if (-not (Test-Path -LiteralPath $RequiredFile -PathType Leaf)) {
            throw "Preparation input not found: $RequiredFile"
        }
    }
}
else {
    foreach ($Manifest in @("train.txt", "val.txt")) {
        $ManifestPath = Join-Path $DatasetPath $Manifest
        if (-not (Test-Path -LiteralPath $ManifestPath -PathType Leaf)) {
            throw "Dataset manifest not found: $ManifestPath"
        }
    }
}

if ($All) {
    $Modes = @("baseline", "topo_aag", "uv_aag", "pc_aag", "cyl_aag", "uv_pc_aag", "dynamic_no_loss", "affinity_only", "instance", "oracle", "cyl", "cyl_affinity", "point", "spe_pcaag", "spe_instance", "full")
}
else {
    $Modes = @($Mode)
}

$CommonArguments = @(
    $Segmentation,
    "train",
    "--num_classes", "25",
    "--batch_size", "$BatchSize",
    "--num_workers", "$NumWorkers",
    "--max_epochs", "$MaxEpochs",
    "--seed", "$Seed",
    "--results_root", $ResultBasePath,
    "--sample_percent", "$SamplePercent",
    "--split_ratios"
) + @($SplitRatios | ForEach-Object { "$_" }) + @(
    "--point_sampling_mode", $PointSamplingMode,
    "--fixed_point_count", "$FixedPointCount",
    "--min_point_count", "$MinPointCount",
    "--max_point_count", "$MaxPointCount",
    "--area_sampling_exponent", "$AreaSamplingExponent",
    "--candidate_grids"
) + @($CandidateGrids | ForEach-Object { "$_" }) + @(
    "--pointcloud_scale", "$PointcloudScale",
    "--cylindrical_scale", "$CylindricalScale",
    "--face_geometry_encoder", $FaceGeometryEncoder,
    "--surface_neighbor_metric", $SurfaceNeighborMetric,
    "--surface_fusion", $SurfaceFusion,
    "--affinity_dim", "$AffinityDim",
    "--affinity_temperature", "$AffinityTemperature",
    "--affinity_loss_weight", "$AffinityLossWeight",
    "--affinity_warmup_epochs", "$AffinityWarmupEpochs",
    "--instance_relation_mapping", $InstanceRelationMapping,
    "--instance_bias_last_layers", "$InstanceBiasLastLayers",
    "--instance_bias_confidence_power", "$InstanceBiasConfidencePower",
    "--instance_bias_gate_init", "$InstanceBiasGateInit",
    "--instance_bias_max_gate_scale", "$InstanceBiasMaxGateScale",
    "--instance_bias_delay_epochs", "$InstanceBiasDelayEpochs",
    "--instance_bias_ramp_epochs", "$InstanceBiasRampEpochs"
)

if ($SurfaceTrimAware) {
    $CommonArguments += "--surface_trim_aware"
}
else {
    $CommonArguments += "--no_surface_trim_aware"
}

if ($AutoPrepare) {
    $CommonArguments += @(
        "--auto_prepare",
        "--raw_dataset_path", $RawDatasetPath,
        "--prepared_dataset_path", $PreparedDatasetPath,
        "--extractor_file", $ExtractorFile,
        "--schema_file", $SchemaFile
    )
}
else {
    $CommonArguments += @("--dataset_path", $DatasetPath)
}

Push-Location $ProjectRoot
try {
    for ($Index = 0; $Index -lt $Modes.Count; $Index++) {
        $CurrentMode = $Modes[$Index]
        $ExperimentLabel = if ($InstanceRelationMapping -eq "positive_only") {
            "${CurrentMode}_positive"
        }
        else {
            $CurrentMode
        }
        if ($FaceGeometryEncoder -eq "param_surface") {
            $TrimLabel = if ($SurfaceTrimAware) { "trim" } else { "untrimmed" }
            $ExperimentLabel = "${ExperimentLabel}_param_surface_${SurfaceNeighborMetric}_${TrimLabel}_${SurfaceFusion}"
        }
        $ExperimentName = "MFRInstSeg_1pct_${ExperimentLabel}_seed${Seed}"
        $Arguments = @($CommonArguments) + @(
            "--experiment_name", $ExperimentName,
            "--experiment_alias", $CurrentMode
        )
        # Rebuild applies only to the first formal run. Every later alias
        # requests the identical cache fingerprint and reuses that result.
        if ($AutoPrepare -and $RebuildPreparedData -and $Index -eq 0) {
            $Arguments += "--rebuild_prepared_data"
        }

        Write-Host ""
        Write-Host "============================================================"
        Write-Host "Starting experiment: $CurrentMode"
        Write-Host "Prepared dataset: $PreparedDatasetPath"
        Write-Host "Seed/sample: $Seed / $SamplePercent"
        Write-Host "Random rotation: disabled"
        Write-Host "Result name: $ExperimentName"
        Write-Host "============================================================"

        & $PythonExe @Arguments
        if ($LASTEXITCODE -ne 0) {
            throw "Experiment '$CurrentMode' failed with exit code $LASTEXITCODE"
        }
    }
}
finally {
    Pop-Location
}
