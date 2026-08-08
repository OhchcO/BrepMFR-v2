param(
    [ValidateSet("baseline", "topo_aag", "uv_aag", "pc_aag", "cyl_aag", "uv_pc_aag", "dynamic_no_loss", "pc_dynamic_no_loss", "affinity_only", "pc_affinity_only", "instance", "pc_instance", "oracle", "pc_oracle", "cyl", "cyl_affinity", "point", "spe_pcaag", "spe_instance", "full")]
    [string]$Mode = "baseline",
    [switch]$All,
    [switch]$AutoPrepare,
    [switch]$NoAutoPrepare,
    [switch]$RebuildPreparedData,
    [string]$DatasetPath = "D:\Code\Datasets\MFRInstSeg_BrepMFR",
    [string]$RawDatasetPath = "D:\Code\Datasets\MFRInstSeg",
    [string]$PreparedDatasetPath = "D:\Code\Datasets\MFRInstSeg_BrepMFR_SPEPCAAG",
    [string]$SharedBinCachePath = "",
    [string[]]$DonorPreparedDatasetPaths = @(),
    [string]$ExtractorFile = "",
    [string]$SchemaFile = "",
    [string]$PythonExe = "F:\ProgramData\anaconda3\envs\aagmspenv\python.exe",
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
# Resolve auto-prepare: default on, -NoAutoPrepare to disable
$EffectiveAutoPrepare = if ($NoAutoPrepare) { $false } else { (-not $AutoPrepare.IsPresent) -or $AutoPrepare }
$env:CUBLAS_WORKSPACE_CONFIG = ":4096:8"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Segmentation = Join-Path $ProjectRoot "segmentation.py"
if ([string]::IsNullOrWhiteSpace($ExtractorFile)) {
    $ExtractorFile = Join-Path $PSScriptRoot "AAGExtractor.py"
}
if ([string]::IsNullOrWhiteSpace($SchemaFile)) {
    $SchemaFile = Join-Path $PSScriptRoot "schema.json"
}

function Get-SamplePercentLabel {
    param([double]$Value)
    if (
        [double]::IsNaN($Value) -or
        [double]::IsInfinity($Value) -or
        $Value -le 0.0 -or
        $Value -gt 1.0
    ) {
        throw "SamplePercent must be finite and in the interval (0, 1]."
    }
    $PercentValue = 100.0 * $Value
    $PercentText = $PercentValue.ToString(
        "0.############",
        [System.Globalization.CultureInfo]::InvariantCulture
    ).Replace(".", "p")
    return "${PercentText}pct"
}

$SampleLabel = Get-SamplePercentLabel -Value $SamplePercent

if (-not (Test-Path -LiteralPath $PythonExe -PathType Leaf)) {
    throw "Python executable not found: $PythonExe"
}
if (-not (Test-Path -LiteralPath $Segmentation -PathType Leaf)) {
    throw "Training entry not found: $Segmentation"
}

if ($EffectiveAutoPrepare) {
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
    $Modes = @("baseline", "topo_aag", "uv_aag", "pc_aag", "cyl_aag", "uv_pc_aag", "dynamic_no_loss", "pc_dynamic_no_loss", "affinity_only", "pc_affinity_only", "instance", "pc_instance", "oracle", "pc_oracle", "cyl", "cyl_affinity", "point", "spe_pcaag", "spe_instance", "full")
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

if ($EffectiveAutoPrepare) {
    $CommonArguments += @(
        "--auto_prepare",
        "--raw_dataset_path", $RawDatasetPath,
        "--prepared_dataset_path", $PreparedDatasetPath,
        "--extractor_file", $ExtractorFile,
        "--schema_file", $SchemaFile
    )
    if (-not [string]::IsNullOrWhiteSpace($SharedBinCachePath)) {
        $CommonArguments += @(
            "--shared_bin_cache_path", $SharedBinCachePath
        )
    }
    if ($DonorPreparedDatasetPaths.Count -gt 0) {
        $CommonArguments += "--donor_prepared_dataset_paths"
        $CommonArguments += $DonorPreparedDatasetPaths
    }
}
else {
    $CommonArguments += @("--dataset_path", $DatasetPath)
    if (-not [string]::IsNullOrWhiteSpace($SharedBinCachePath)) {
        $CommonArguments += @("--bin_root", $SharedBinCachePath)
    }
    elseif (-not [string]::IsNullOrWhiteSpace($PreparedDatasetPath)) {
        $CommonArguments += @("--bin_root", $PreparedDatasetPath)
    }
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
        $ExperimentName = "MFRInstSeg_${SampleLabel}_${ExperimentLabel}_seed${Seed}"
        $Arguments = @($CommonArguments) + @(
            "--experiment_name", $ExperimentName,
            "--experiment_alias", $CurrentMode
        )
        # Rebuild applies only to the first formal run. Every later alias
        # requests the identical cache fingerprint and reuses that result.
        if ($EffectiveAutoPrepare -and $RebuildPreparedData -and $Index -eq 0) {
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

