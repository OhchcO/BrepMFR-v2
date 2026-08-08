param(
    [string]$RawDatasetPath = "D:\Code\Datasets\MFRInstSeg",
    [string]$PreparedDatasetPath = "D:\Code\Datasets\MFRInstSeg_BrepMFR_ParamSurface_v4_seed42",
    [string]$ResultBasePath = "D:\Code\BrepMFR\results",
    [string]$PythonExe = "F:\ProgramData\anaconda3\envs\aagmspenv\python.exe",
    [int]$BatchSize = 16,
    [int]$MaxEpochs = 100,
    [int]$NumWorkers = 0,
    [double]$SamplePercent = 0.01,
    [switch]$RebuildPreparedData,
    [string]$TrainingLauncher = "",
    [string]$TestLauncher = ""
)

$ErrorActionPreference = "Stop"
$Seed = 42
if ([Math]::Abs($SamplePercent - 0.01) -gt 1e-12) {
    throw "This 1pct experiment runner requires exactly SamplePercent=0.01."
}

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

$Experiments = @(
    [pscustomobject]@{
        Mode = "pc_aag"
        Mapping = "signed"
        ResultLabel = "pc_aag_param_surface_intrinsic_untrimmed_gated"
        TestBiasOff = $false
    },
    [pscustomobject]@{
        Mode = "pc_affinity_only"
        Mapping = "positive_only"
        ResultLabel = "pc_affinity_only_positive_param_surface_intrinsic_untrimmed_gated"
        TestBiasOff = $false
    },
    [pscustomobject]@{
        Mode = "pc_dynamic_no_loss"
        Mapping = "positive_only"
        ResultLabel = "pc_dynamic_no_loss_positive_param_surface_intrinsic_untrimmed_gated"
        TestBiasOff = $false
    },
    [pscustomobject]@{
        Mode = "pc_instance"
        Mapping = "positive_only"
        ResultLabel = "pc_instance_positive_param_surface_intrinsic_untrimmed_gated"
        TestBiasOff = $true
    },
    [pscustomobject]@{
        Mode = "pc_oracle"
        Mapping = "positive_only"
        ResultLabel = "pc_oracle_positive_param_surface_intrinsic_untrimmed_gated"
        TestBiasOff = $false
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
    $Logs = Get-ChildItem `
        -LiteralPath $ResultRoot `
        -Recurse `
        -Filter "test_log.txt" `
        -File
    $Logs = $Logs | Where-Object {
        $_.FullName -notmatch '[\\/]test_bias_scale_0[\\/]'
    }
    return $Logs | Sort-Object LastWriteTime -Descending | Select-Object -First 1
}

function Get-PairedBiasOffLog {
    param(
        [Parameter(Mandatory = $true)]
        [System.IO.FileInfo]$NormalLog
    )

    $BiasOffPath = Join-Path `
        $NormalLog.DirectoryName `
        "test_bias_scale_0\test_log.txt"
    if (Test-Path -LiteralPath $BiasOffPath -PathType Leaf) {
        return Get-Item -LiteralPath $BiasOffPath
    }
    return $null
}

Write-Host "Seed: $Seed"
Write-Host "Prepared dataset: $PreparedDatasetPath"
Write-Host "Experiments: $($Experiments.Count)"

for ($Index = 0; $Index -lt $Experiments.Count; $Index++) {
    $Experiment = $Experiments[$Index]
    $ResultRoot = Join-Path $ResultBasePath (
        "MFRInstSeg_1pct_{0}_seed{1}" -f $Experiment.ResultLabel, $Seed
    )
    $NormalLog = $null
    $BiasOffLog = $null
    if (-not $RebuildPreparedData) {
        $NormalLog = Get-LatestNormalTestLog -ResultRoot $ResultRoot
        if ($Experiment.TestBiasOff -and $null -ne $NormalLog) {
            $BiasOffLog = Get-PairedBiasOffLog -NormalLog $NormalLog
        }
    }
    $AllRequiredTestsExist = (
        $null -ne $NormalLog `
        -and (-not $Experiment.TestBiasOff -or $null -ne $BiasOffLog)
    )
    if ($AllRequiredTestsExist) {
        Write-Host ""
        Write-Host "Skipping completed experiment: $($Experiment.ResultLabel)"
        Write-Host "Existing test log: $($NormalLog.FullName)"
        if ($Experiment.TestBiasOff) {
            Write-Host "Existing bias-off log: $($BiasOffLog.FullName)"
        }
        continue
    }

    $Checkpoint = $null
    if ($null -ne $NormalLog) {
        $Checkpoint = Get-ChildItem `
            -LiteralPath $NormalLog.DirectoryName `
            -Filter "best-*.ckpt" `
            -File `
            | Sort-Object LastWriteTime -Descending `
            | Select-Object -First 1 -ExpandProperty FullName
        if (-not $Checkpoint) {
            throw "Completed test has no checkpoint: $($Experiment.ResultLabel)"
        }
    }
    else {
        $ExistingCheckpointPaths = @{}
        if (Test-Path -LiteralPath $ResultRoot -PathType Container) {
            Get-ChildItem `
                -LiteralPath $ResultRoot `
                -Recurse `
                -Filter "best-*.ckpt" `
                -File `
                | ForEach-Object {
                    $ExistingCheckpointPaths[$_.FullName] = $true
                }
        }

        $TrainingArguments = @{
            Mode = $Experiment.Mode
            InstanceRelationMapping = $Experiment.Mapping
            RawDatasetPath = $RawDatasetPath
            PreparedDatasetPath = $PreparedDatasetPath
            SamplePercent = $SamplePercent
            Seed = $Seed
            BatchSize = $BatchSize
            MaxEpochs = $MaxEpochs
            NumWorkers = $NumWorkers
            PythonExe = $PythonExe
            ResultBasePath = $ResultBasePath
            FaceGeometryEncoder = "param_surface"
            SurfaceNeighborMetric = "intrinsic"
            SurfaceTrimAware = $false
            SurfaceFusion = "gated"
        }
        if ($RebuildPreparedData -and $Index -eq 0) {
            $TrainingArguments.RebuildPreparedData = $true
        }

        Write-Host ""
        Write-Host "============================================================"
        Write-Host "Training: $($Experiment.ResultLabel)"
        Write-Host "Mode/mapping: $($Experiment.Mode) / $($Experiment.Mapping)"
        Write-Host "Geometry: param_surface / intrinsic / untrimmed / gated"
        Write-Host "============================================================"

        & $TrainingLauncher @TrainingArguments
        if (-not $?) {
            throw "Training failed: $($Experiment.ResultLabel)"
        }
        if ($null -ne $LASTEXITCODE -and $LASTEXITCODE -ne 0) {
            throw "Training failed with exit code ${LASTEXITCODE}: $($Experiment.ResultLabel)"
        }

        if (Test-Path -LiteralPath $ResultRoot -PathType Container) {
            $Checkpoint = Get-ChildItem `
                -LiteralPath $ResultRoot `
                -Recurse `
                -Filter "best-*.ckpt" `
                -File `
                | Where-Object {
                    -not $ExistingCheckpointPaths.ContainsKey($_.FullName)
                } `
                | Sort-Object LastWriteTime -Descending `
                | Select-Object -First 1 -ExpandProperty FullName
        }
        if (-not $Checkpoint) {
            throw "No fresh best checkpoint found: $($Experiment.ResultLabel)"
        }
    }

    if ($null -eq $NormalLog) {
        Write-Host "Testing checkpoint: $Checkpoint"
        & $TestLauncher `
            -Checkpoint $Checkpoint `
            -PreparedDataset $PreparedDatasetPath `
            -Seed $Seed `
            -BatchSize $BatchSize `
            -NumWorkers $NumWorkers `
            -PythonExe $PythonExe
        if (-not $?) {
            throw "Test failed: $($Experiment.ResultLabel)"
        }
        if ($null -ne $LASTEXITCODE -and $LASTEXITCODE -ne 0) {
            throw "Test failed with exit code ${LASTEXITCODE}: $($Experiment.ResultLabel)"
        }
    }

    if ($Experiment.TestBiasOff -and $null -eq $BiasOffLog) {
        Write-Host "Testing checkpoint with instance bias disabled: $Checkpoint"
        & $TestLauncher `
            -Checkpoint $Checkpoint `
            -PreparedDataset $PreparedDatasetPath `
            -Seed $Seed `
            -BatchSize $BatchSize `
            -NumWorkers $NumWorkers `
            -PythonExe $PythonExe `
            -InstanceBiasScaleOverride 0
        if (-not $?) {
            throw "Bias-off test failed: $($Experiment.ResultLabel)"
        }
        if ($null -ne $LASTEXITCODE -and $LASTEXITCODE -ne 0) {
            throw "Bias-off test failed with exit code ${LASTEXITCODE}: $($Experiment.ResultLabel)"
        }
    }
}

Write-Host ""
Write-Host "Completed seed42 parameter-surface instance-combination experiments."

