import os
import subprocess
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CORE_RUNNER = PROJECT_ROOT / "tools" / "run_mfrinstseg_core_seed.ps1"


def _fake_launchers(tmp_path):
    events = tmp_path / "events.txt"
    fake_train = tmp_path / "fake-train.ps1"
    fake_test = tmp_path / "fake-test.ps1"
    fake_train.write_text(
        r"""
param(
    [string]$Mode,
    [string]$InstanceRelationMapping = "signed",
    [string]$RawDatasetPath,
    [string]$PreparedDatasetPath,
    [double]$SamplePercent,
    [int]$Seed,
    [int]$BatchSize,
    [int]$MaxEpochs,
    [int]$NumWorkers,
    [string]$PythonExe
)
$label = if ($InstanceRelationMapping -eq "positive_only") {
    "${Mode}_positive"
} else {
    $Mode
}
$root = Join-Path $env:MFR_CORE_RESULT_BASE "MFRInstSeg_1pct_${label}_seed${Seed}"
$run = Join-Path $root "fake-run"
New-Item -ItemType Directory -Path $run -Force | Out-Null
$checkpoint = Join-Path $run "best-fake.ckpt"
[System.IO.File]::WriteAllText($checkpoint, "checkpoint")
[System.IO.File]::AppendAllText(
    $env:MFR_CORE_EVENTS,
    "train|$Mode|$InstanceRelationMapping|$Seed|$PreparedDatasetPath|$checkpoint`n"
)
""".strip()
        + "\n",
        encoding="utf-8",
    )
    fake_test.write_text(
        r"""
param(
    [string]$Checkpoint,
    [string]$PreparedDataset,
    [int]$Seed,
    [int]$BatchSize,
    [int]$NumWorkers,
    [string]$PythonExe
)
[System.IO.File]::AppendAllText(
    $env:MFR_CORE_EVENTS,
    "test|$Seed|$PreparedDataset|$Checkpoint`n"
)
""".strip()
        + "\n",
        encoding="utf-8",
    )
    return fake_train, fake_test, events


def _run_core(tmp_path, seed=42, training_launcher=None):
    fake_train, fake_test, events = _fake_launchers(tmp_path)
    if training_launcher is not None:
        fake_train = training_launcher
    result_base = tmp_path / "results"
    prepared_base = tmp_path / "prepared"
    environment = os.environ.copy()
    environment["MFR_CORE_RESULT_BASE"] = str(result_base)
    environment["MFR_CORE_EVENTS"] = str(events)
    completed = subprocess.run(
        [
            "powershell.exe",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(CORE_RUNNER),
            "-Seed",
            str(seed),
            "-RawDatasetPath",
            str(tmp_path / "raw"),
            "-PreparedDatasetBase",
            str(prepared_base),
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
    return completed, events, prepared_base, result_base


def test_core_runner_trains_and_tests_three_experiments_in_order(tmp_path):
    completed, events, prepared_base, _ = _run_core(tmp_path, seed=42)

    assert completed.returncode == 0, completed.stderr
    lines = events.read_text(encoding="utf-8").splitlines()
    prepared = str(prepared_base) + "_seed42"
    assert [line.split("|")[:4] for line in lines[::2]] == [
        ["train", "baseline", "signed", "42"],
        ["train", "dynamic_no_loss", "positive_only", "42"],
        ["train", "instance", "positive_only", "42"],
    ]
    assert all(line.split("|")[4] == prepared for line in lines[::2])
    assert [line.split("|")[0] for line in lines] == [
        "train",
        "test",
        "train",
        "test",
        "train",
        "test",
    ]
    assert all(line.split("|")[2] == prepared for line in lines[1::2])


def test_core_runner_rejects_stale_checkpoint_after_noop_training(tmp_path):
    noop = tmp_path / "noop-train.ps1"
    noop.write_text(
        r"""
param(
    [string]$Mode,
    [string]$InstanceRelationMapping = "signed",
    [string]$RawDatasetPath,
    [string]$PreparedDatasetPath,
    [double]$SamplePercent,
    [int]$Seed,
    [int]$BatchSize,
    [int]$MaxEpochs,
    [int]$NumWorkers,
    [string]$PythonExe
)
""".strip()
        + "\n",
        encoding="utf-8",
    )
    result_base = tmp_path / "results"
    stale = (
        result_base
        / "MFRInstSeg_1pct_baseline_seed42"
        / "old-run"
        / "best-old.ckpt"
    )
    stale.parent.mkdir(parents=True)
    stale.write_bytes(b"old")
    os.utime(stale, (1_600_000_000, 1_600_000_000))

    completed, events, _, _ = _run_core(
        tmp_path,
        seed=42,
        training_launcher=noop,
    )

    assert completed.returncode != 0
    assert "No fresh best checkpoint found: baseline" in (
        completed.stdout + completed.stderr
    )
    assert not events.exists()
