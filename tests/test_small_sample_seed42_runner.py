import os
import subprocess
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNNER = PROJECT_ROOT / "tools" / "run_mfrinstseg_small_sample_seed42.ps1"
RESULT_LABELS = [
    "baseline",
    "pc_aag_param_surface_intrinsic_untrimmed_gated",
    "affinity_only_positive",
    "instance_positive",
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
    [string]$SharedBinCachePath,
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
$labels = @{
    baseline = "baseline"
    pc_aag = "pc_aag_param_surface_intrinsic_untrimmed_gated"
    affinity_only = "affinity_only_positive"
    instance = "instance_positive"
}
$rootName = "MFRInstSeg_$($env:MFR_SMALL_SAMPLE_LABEL)_$($labels[$Mode])_seed${Seed}"
$root = Join-Path $env:MFR_SMALL_SAMPLE_RESULT_BASE $rootName
$run = Join-Path $root "fake-run"
New-Item -ItemType Directory -Path $run -Force | Out-Null
$checkpoint = Join-Path $run "best-fake.ckpt"
[System.IO.File]::WriteAllText($checkpoint, "checkpoint")
[System.IO.File]::AppendAllText(
    $env:MFR_SMALL_SAMPLE_EVENTS,
    "train|$Mode|$InstanceRelationMapping|$SamplePercent|$FaceGeometryEncoder|$SurfaceNeighborMetric|$SurfaceTrimAware|$SurfaceFusion|$Seed|$PreparedDatasetPath|$SharedBinCachePath|$($RebuildPreparedData.IsPresent)|$checkpoint`n"
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
    [string]$SharedBinCachePath,
    [int]$Seed,
    [int]$BatchSize,
    [int]$NumWorkers,
    [string]$PythonExe
)
[System.IO.File]::AppendAllText(
    $env:MFR_SMALL_SAMPLE_EVENTS,
    "test|$Seed|$PreparedDataset|$SharedBinCachePath|$Checkpoint`n"
)
'''.strip()
        + "\n",
        encoding="utf-8",
    )
    return fake_train, fake_test, events


def _invoke_runner(
    tmp_path,
    sample_percent,
    sample_label,
    *,
    rebuild=False,
):
    fake_train, fake_test, events = _fake_launchers(tmp_path)
    result_base = tmp_path / "results"
    prepared_base = tmp_path / "MFRInstSeg_BrepMFR_ParamSurface_v4"
    shared = tmp_path / "MFRInstSeg_BrepMFR_ParamSurface_v4_shared_seed42"
    environment = os.environ.copy()
    environment["MFR_SMALL_SAMPLE_EVENTS"] = str(events)
    environment["MFR_SMALL_SAMPLE_RESULT_BASE"] = str(result_base)
    environment["MFR_SMALL_SAMPLE_LABEL"] = sample_label
    arguments = [
        "powershell.exe",
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        str(RUNNER),
        "-SamplePercent",
        str(sample_percent),
        "-RawDatasetPath",
        str(tmp_path / "raw"),
        "-PreparedDatasetBasePath",
        str(prepared_base),
        "-ResultBasePath",
        str(result_base),
        "-SharedBinCachePath",
        str(shared),
        "-TrainingLauncher",
        str(fake_train),
        "-TestLauncher",
        str(fake_test),
        "-PythonExe",
        "fake-python.exe",
    ]
    if rebuild:
        arguments.append("-RebuildPreparedData")
    completed = subprocess.run(
        arguments,
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    return completed, events, result_base, prepared_base, shared


def _read_events(events):
    rows = []
    for line in events.read_text(encoding="utf-8").splitlines():
        fields = line.split("|")
        if fields[0] == "train":
            rows.append(
                {
                    "kind": "train",
                    "mode": fields[1],
                    "mapping": fields[2],
                    "sample": fields[3],
                    "encoder": fields[4],
                    "metric": fields[5],
                    "trim": fields[6],
                    "fusion": fields[7],
                    "seed": fields[8],
                    "prepared": fields[9],
                    "shared": fields[10],
                    "rebuild": fields[11],
                    "checkpoint": fields[12],
                }
            )
        else:
            rows.append(
                {
                    "kind": "test",
                    "seed": fields[1],
                    "prepared": fields[2],
                    "shared": fields[3],
                    "checkpoint": fields[4],
                }
            )
    return rows


@pytest.mark.parametrize(
    ("sample_percent", "sample_label", "cache_suffix"),
    [
        (0.01, "1pct", "seed42"),
        (0.05, "5pct", "5pct_seed42"),
        (0.10, "10pct", "10pct_seed42"),
    ],
)
def test_runner_executes_percentage_matrix(
    tmp_path,
    sample_percent,
    sample_label,
    cache_suffix,
):
    result_base = tmp_path / "results"
    old_checkpoints = []
    for label in RESULT_LABELS:
        old_run = result_base / f"MFRInstSeg_{sample_label}_{label}_seed42" / "old-run"
        old_run.mkdir(parents=True)
        old_checkpoint = old_run / "best-old.ckpt"
        old_checkpoint.write_bytes(b"old")
        old_checkpoints.append(str(old_checkpoint))

    completed, events, _, prepared_base, shared = _invoke_runner(
        tmp_path,
        sample_percent,
        sample_label,
        rebuild=True,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    rows = _read_events(events)
    trains = [row for row in rows if row["kind"] == "train"]
    tests = [row for row in rows if row["kind"] == "test"]
    assert [row["mode"] for row in trains] == [
        "baseline",
        "pc_aag",
        "affinity_only",
        "instance",
    ]
    assert [row["mapping"] for row in trains] == [
        "signed",
        "signed",
        "positive_only",
        "positive_only",
    ]
    assert [row["encoder"] for row in trains] == [
        "mini_point",
        "param_surface",
        "mini_point",
        "mini_point",
    ]
    assert trains[1]["metric"] == "intrinsic"
    assert trains[1]["trim"] == "False"
    assert trains[1]["fusion"] == "gated"
    assert all(row["sample"] == str(sample_percent) for row in trains)
    assert all(row["seed"] == "42" for row in trains)
    expected_prepared = f"{prepared_base}_{cache_suffix}"
    assert all(row["prepared"] == expected_prepared for row in trains)
    assert all(row["shared"] == str(shared) for row in trains)
    assert [row["rebuild"] for row in trains] == [
        "True",
        "False",
        "False",
        "False",
    ]
    assert len(tests) == 4
    tested_checkpoints = {row["checkpoint"] for row in tests}
    assert tested_checkpoints == {row["checkpoint"] for row in trains}
    assert tested_checkpoints.isdisjoint(old_checkpoints)
    assert all(row["prepared"] == expected_prepared for row in tests)
    assert all(row["shared"] == str(shared) for row in tests)


@pytest.mark.parametrize("sample_percent", [0, 0.02, 1.01])
def test_runner_rejects_unsupported_percentage_before_launch(
    tmp_path,
    sample_percent,
):
    completed, events, _, _, _ = _invoke_runner(
        tmp_path,
        sample_percent,
        "unsupported",
    )

    assert completed.returncode != 0
    assert "SamplePercent must be one of 0.01, 0.05, or 0.10" in (
        completed.stdout + completed.stderr
    )
    assert not events.exists()


def test_runner_skips_four_completed_results(tmp_path):
    result_base = tmp_path / "results"
    for label in RESULT_LABELS:
        run = result_base / f"MFRInstSeg_10pct_{label}_seed42" / "fake-run"
        run.mkdir(parents=True)
        (run / "best-fake.ckpt").write_bytes(b"checkpoint")
        (run / "test_log.txt").write_text("complete\n", encoding="utf-8")

    completed, events, _, _, _ = _invoke_runner(
        tmp_path,
        0.10,
        "10pct",
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert not events.exists()
