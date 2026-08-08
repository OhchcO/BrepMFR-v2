import os
import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNNER = PROJECT_ROOT / "tools" / "run_mfrinstseg_param_surface_instance_seed42.ps1"
EXPECTED_MATRIX = [
    "pc_aag",
    "pc_affinity_only",
    "pc_dynamic_no_loss",
    "pc_instance",
    "pc_oracle",
]
EXPECTED_LABELS = [
    "pc_aag_param_surface_intrinsic_untrimmed_gated",
    "pc_affinity_only_positive_param_surface_intrinsic_untrimmed_gated",
    "pc_dynamic_no_loss_positive_param_surface_intrinsic_untrimmed_gated",
    "pc_instance_positive_param_surface_intrinsic_untrimmed_gated",
    "pc_oracle_positive_param_surface_intrinsic_untrimmed_gated",
]


def _fake_launchers(tmp_path):
    events = tmp_path / "events.txt"
    fake_train = tmp_path / "fake-train.ps1"
    fake_test = tmp_path / "fake-test.ps1"
    fake_train.write_text(
        r'''
param(
    [string]$Mode,
    [string]$InstanceRelationMapping,
    [string]$RawDatasetPath,
    [string]$PreparedDatasetPath,
    [double]$SamplePercent,
    [int]$Seed,
    [int]$BatchSize,
    [int]$MaxEpochs,
    [int]$NumWorkers,
    [string]$PythonExe,
    [string]$ResultBasePath,
    [string]$FaceGeometryEncoder,
    [string]$SurfaceNeighborMetric,
    [bool]$SurfaceTrimAware,
    [string]$SurfaceFusion,
    [switch]$RebuildPreparedData
)
$trim = if ($SurfaceTrimAware) { "trim" } else { "untrimmed" }
$mapping = if ($InstanceRelationMapping -eq "positive_only") {
    "positive"
} else {
    $InstanceRelationMapping
}
$label = "${Mode}_${mapping}_param_surface_${SurfaceNeighborMetric}_${trim}_${SurfaceFusion}"
if ($Mode -eq "pc_aag") {
    $label = "pc_aag_param_surface_${SurfaceNeighborMetric}_${trim}_${SurfaceFusion}"
}
$root = Join-Path $env:MFR_PARAM_INSTANCE_RESULT_BASE "MFRInstSeg_1pct_${label}_seed${Seed}"
$run = Join-Path $root "fake-run"
New-Item -ItemType Directory -Path $run -Force | Out-Null
$checkpoint = Join-Path $run "best-fake.ckpt"
[System.IO.File]::WriteAllText($checkpoint, "checkpoint")
[System.IO.File]::AppendAllText(
    $env:MFR_PARAM_INSTANCE_EVENTS,
    "train|$Mode|$InstanceRelationMapping|$FaceGeometryEncoder|$SurfaceNeighborMetric|$SurfaceTrimAware|$SurfaceFusion|$Seed|$checkpoint`n"
)
'''.strip()
        + "\n",
        encoding="utf-8",
    )
    fake_test.write_text(
        r'''
param(
    [string]$Checkpoint,
    [string]$PreparedDataset,
    [int]$Seed,
    [int]$BatchSize,
    [int]$NumWorkers,
    [string]$PythonExe,
    [Nullable[double]]$InstanceBiasScaleOverride
)
$override = if ($null -eq $InstanceBiasScaleOverride) {
    "default"
} else {
    "$InstanceBiasScaleOverride"
}
[System.IO.File]::AppendAllText(
    $env:MFR_PARAM_INSTANCE_EVENTS,
    "test|$Seed|$PreparedDataset|$Checkpoint|$override`n"
)
'''.strip()
        + "\n",
        encoding="utf-8",
    )
    return fake_train, fake_test, events


def test_seed42_runner_executes_controlled_matrix_and_instance_bias_ablation(tmp_path):
    fake_train, fake_test, events = _fake_launchers(tmp_path)
    result_base = tmp_path / "results"
    prepared = tmp_path / "prepared-v4-seed42"
    old_checkpoints = []
    for label in EXPECTED_LABELS:
        old_run = result_base / f"MFRInstSeg_1pct_{label}_seed42" / "old-run"
        old_run.mkdir(parents=True)
        old_checkpoint = old_run / "best-old.ckpt"
        old_checkpoint.write_bytes(b"old")
        old_checkpoints.append(str(old_checkpoint))
    environment = os.environ.copy()
    environment["MFR_PARAM_INSTANCE_RESULT_BASE"] = str(result_base)
    environment["MFR_PARAM_INSTANCE_EVENTS"] = str(events)

    completed = subprocess.run(
        [
            "powershell.exe",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(RUNNER),
            "-RawDatasetPath",
            str(tmp_path / "raw"),
            "-PreparedDatasetPath",
            str(prepared),
            "-ResultBasePath",
            str(result_base),
            "-TrainingLauncher",
            str(fake_train),
            "-TestLauncher",
            str(fake_test),
            "-PythonExe",
            "fake-python.exe",
        ],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    lines = events.read_text(encoding="utf-8").splitlines()
    train_lines = [line.split("|") for line in lines if line.startswith("train|")]
    test_lines = [line.split("|") for line in lines if line.startswith("test|")]

    assert [line[1] for line in train_lines] == EXPECTED_MATRIX
    assert [line[2] for line in train_lines] == [
        "signed",
        "positive_only",
        "positive_only",
        "positive_only",
        "positive_only",
    ]
    assert all(line[3] == "param_surface" for line in train_lines)
    assert all(line[4] == "intrinsic" for line in train_lines)
    assert all(line[5] == "False" for line in train_lines)
    assert all(line[6] == "gated" for line in train_lines)
    assert all(line[7] == "42" for line in train_lines)
    assert len(test_lines) == 6
    assert [line[4] for line in test_lines].count("default") == 5
    assert [line[4] for line in test_lines].count("0") == 1
    default_test_checkpoints = {
        line[3] for line in test_lines if line[4] == "default"
    }
    assert default_test_checkpoints == {line[8] for line in train_lines}
    assert default_test_checkpoints.isdisjoint(old_checkpoints)
    bias_off_line = next(line for line in test_lines if line[4] == "0")
    assert bias_off_line[3] == train_lines[3][8]
    assert all(line[2] == str(prepared) for line in test_lines)


def test_seed42_runner_pairs_bias_ablation_with_latest_normal_run(tmp_path):
    fake_train, fake_test, events = _fake_launchers(tmp_path)
    result_base = tmp_path / "results"
    prepared = tmp_path / "prepared-v4-seed42"
    instance_checkpoint = None
    for label in EXPECTED_LABELS:
        run = result_base / f"MFRInstSeg_1pct_{label}_seed42" / "fake-run"
        run.mkdir(parents=True)
        (run / "test_log.txt").write_text("complete\n", encoding="utf-8")
        checkpoint = run / "best-fake.ckpt"
        checkpoint.write_bytes(b"checkpoint")
        if label.startswith("pc_instance_positive"):
            instance_checkpoint = checkpoint

            older_run = run.parent / "older-run"
            older_bias = older_run / "test_bias_scale_0"
            older_bias.mkdir(parents=True)
            (older_run / "test_log.txt").write_text("older\n", encoding="utf-8")
            (older_run / "best-older.ckpt").write_bytes(b"older")
            (older_bias / "test_log.txt").write_text(
                "older bias off\n", encoding="utf-8"
            )

            # The regular log selects fake-run as the latest completed run. An
            # older bias-off log must not be paired with it to skip evaluation.
            os.utime(older_run / "test_log.txt", (1, 1))
            os.utime(run / "test_log.txt", (2, 2))

    environment = os.environ.copy()
    environment["MFR_PARAM_INSTANCE_RESULT_BASE"] = str(result_base)
    environment["MFR_PARAM_INSTANCE_EVENTS"] = str(events)
    completed = subprocess.run(
        [
            "powershell.exe",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(RUNNER),
            "-RawDatasetPath",
            str(tmp_path / "raw"),
            "-PreparedDatasetPath",
            str(prepared),
            "-ResultBasePath",
            str(result_base),
            "-TrainingLauncher",
            str(fake_train),
            "-TestLauncher",
            str(fake_test),
            "-PythonExe",
            "fake-python.exe",
        ],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert events.read_text(encoding="utf-8").splitlines() == [
        f"test|42|{prepared}|{instance_checkpoint}|0"
    ]


def test_seed42_runner_rejects_non_one_percent_identity(tmp_path):
    fake_train, fake_test, events = _fake_launchers(tmp_path)
    completed = subprocess.run(
        [
            "powershell.exe",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(RUNNER),
            "-SamplePercent",
            "0.02",
            "-ResultBasePath",
            str(tmp_path / "results"),
            "-TrainingLauncher",
            str(fake_train),
            "-TestLauncher",
            str(fake_test),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert "exactly SamplePercent=0.01" in (completed.stdout + completed.stderr)
    assert not events.exists()
