# Multi-Seed Core Experiment Runner Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add three directly runnable Seed 42, 2025, and 2027 PowerShell entry points that sequentially train and test Baseline, Dynamic Positive, and Instance Positive.

**Architecture:** A shared PowerShell runner owns experiment ordering, seed-specific cache derivation, fresh-checkpoint selection, and failure propagation. Three thin wrappers fix the declared seeds while forwarding optional path overrides, allowing behavior to be tested with temporary fake launchers without executing conversion, training, or evaluation.

**Tech Stack:** PowerShell 5.1, Python 3, pytest.

## Global Constraints

- Never execute real dataset conversion, training, or evaluation while implementing or testing.
- The experiment order is `baseline`, `dynamic_no_loss + positive_only`, then `instance + positive_only`.
- Use `SamplePercent=0.01`, `BatchSize=16`, `MaxEpochs=100`, and `NumWorkers=0`.
- Derive the prepared path as `<PreparedDatasetBase>_seed<Seed>`.
- Never pass `-RebuildPreparedData`.
- Test only checkpoints created during the current training invocation.
- Do not add Bias OFF evaluation to these scripts.
- Do not modify the existing training or test launchers.

---

### Task 1: Shared seeded experiment runner

**Files:**
- Create: `tests/test_multiseed_core_runner.py`
- Create: `tools/run_mfrinstseg_core_seed.ps1`

**Interfaces:**
- Consumes: `tools/run_mfrinstseg_experiments.ps1` and `tools/run_mfrinstseg_test.ps1`.
- Produces: `run_mfrinstseg_core_seed.ps1 -Seed <int>` with optional `RawDatasetPath`, `PreparedDatasetBase`, `ResultBasePath`, `PythonExe`, `TrainingLauncher`, and `TestLauncher`.

- [ ] **Step 1: Write shared test fixtures and the failing sequence test**

Create `tests/test_multiseed_core_runner.py` with:

```python
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
""".strip() + "\n",
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
""".strip() + "\n",
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
        "train", "test", "train", "test", "train", "test"
    ]
    assert all(line.split("|")[2] == prepared for line in lines[1::2])
```

- [ ] **Step 2: Run the sequence test and confirm RED**

Run:

```powershell
& "G:\ProgramData\anaconda3\envs\aagmspenv\python.exe" -m pytest `
  tests/test_multiseed_core_runner.py::test_core_runner_trains_and_tests_three_experiments_in_order -q
```

Expected: FAIL because `tools/run_mfrinstseg_core_seed.ps1` does not exist.

- [ ] **Step 3: Implement the shared runner**

Create `tools/run_mfrinstseg_core_seed.ps1` with these behaviors:

```powershell
param(
    [Parameter(Mandatory = $true)]
    [ValidateRange(0, [int]::MaxValue)]
    [int]$Seed,
    [string]$RawDatasetPath = "D:\Code\Datasets\MFRInstSeg",
    [string]$PreparedDatasetBase = "D:\Code\Datasets\MFRInstSeg_BrepMFR_SPEPCAAG",
    [string]$ResultBasePath = "D:\Code\BrepMFR\results",
    [string]$PythonExe = "G:\ProgramData\anaconda3\envs\aagmspenv\python.exe",
    [string]$TrainingLauncher = "",
    [string]$TestLauncher = ""
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
if ([string]::IsNullOrWhiteSpace($TrainingLauncher)) {
    $TrainingLauncher = Join-Path $PSScriptRoot "run_mfrinstseg_experiments.ps1"
}
if ([string]::IsNullOrWhiteSpace($TestLauncher)) {
    $TestLauncher = Join-Path $PSScriptRoot "run_mfrinstseg_test.ps1"
}
foreach ($launcher in @($TrainingLauncher, $TestLauncher)) {
    if (-not (Test-Path -LiteralPath $launcher -PathType Leaf)) {
        throw "Launcher not found: $launcher"
    }
}

$PreparedDatasetPath = "${PreparedDatasetBase}_seed${Seed}"
$Experiments = @(
    [pscustomobject]@{
        Mode = "baseline"
        Mapping = $null
        ResultLabel = "baseline"
    },
    [pscustomobject]@{
        Mode = "dynamic_no_loss"
        Mapping = "positive_only"
        ResultLabel = "dynamic_no_loss_positive"
    },
    [pscustomobject]@{
        Mode = "instance"
        Mapping = "positive_only"
        ResultLabel = "instance_positive"
    }
)

foreach ($Experiment in $Experiments) {
    $StartedAt = Get-Date
    $TrainingArguments = @{
        Mode = $Experiment.Mode
        RawDatasetPath = $RawDatasetPath
        PreparedDatasetPath = $PreparedDatasetPath
        SamplePercent = 0.01
        Seed = $Seed
        BatchSize = 16
        MaxEpochs = 100
        NumWorkers = 0
        PythonExe = $PythonExe
    }
    if ($null -ne $Experiment.Mapping) {
        $TrainingArguments.InstanceRelationMapping = $Experiment.Mapping
    }

    & $TrainingLauncher @TrainingArguments
    if (-not $?) {
        throw "Training failed: $($Experiment.ResultLabel)"
    }
    if ($null -ne $LASTEXITCODE -and $LASTEXITCODE -ne 0) {
        throw "Training failed with exit code ${LASTEXITCODE}: $($Experiment.ResultLabel)"
    }

    $ResultRoot = Join-Path $ResultBasePath (
        "MFRInstSeg_1pct_{0}_seed{1}" -f $Experiment.ResultLabel, $Seed
    )
    $FreshnessBoundary = $StartedAt.AddSeconds(-2)
    $Checkpoint = $null
    if (Test-Path -LiteralPath $ResultRoot -PathType Container) {
        $Checkpoint = Get-ChildItem -LiteralPath $ResultRoot -Recurse -Filter "best-*.ckpt" -File `
            | Where-Object { $_.LastWriteTime -ge $FreshnessBoundary } `
            | Sort-Object LastWriteTime -Descending `
            | Select-Object -First 1 -ExpandProperty FullName
    }
    if (-not $Checkpoint) {
        throw "No fresh best checkpoint found: $($Experiment.ResultLabel)"
    }

    & $TestLauncher `
        -Checkpoint $Checkpoint `
        -PreparedDataset $PreparedDatasetPath `
        -Seed $Seed `
        -BatchSize 16 `
        -NumWorkers 0 `
        -PythonExe $PythonExe
    if (-not $?) {
        throw "Test failed: $($Experiment.ResultLabel)"
    }
    if ($null -ne $LASTEXITCODE -and $LASTEXITCODE -ne 0) {
        throw "Test failed with exit code ${LASTEXITCODE}: $($Experiment.ResultLabel)"
    }
}
```

- [ ] **Step 4: Run the sequence test and confirm GREEN**

Run the Step 2 command again.

Expected: `1 passed`.

- [ ] **Step 5: Add the stale-checkpoint failing test**

Append:

```python
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
""".strip() + "\n",
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
```

- [ ] **Step 6: Run both shared-runner tests**

Run:

```powershell
& "G:\ProgramData\anaconda3\envs\aagmspenv\python.exe" -m pytest `
  tests/test_multiseed_core_runner.py -q
```

Expected: `2 passed`.

- [ ] **Step 7: Commit the shared runner**

```powershell
git add tools/run_mfrinstseg_core_seed.ps1 tests/test_multiseed_core_runner.py
git commit -m "feat: add shared seeded core experiment runner"
```

---

### Task 2: Fixed Seed 42, 2025, and 2027 entry points

**Files:**
- Modify: `tests/test_multiseed_core_runner.py`
- Create: `tools/run_mfrinstseg_core_seed42.ps1`
- Create: `tools/run_mfrinstseg_core_seed2025.ps1`
- Create: `tools/run_mfrinstseg_core_seed2027.ps1`

**Interfaces:**
- Consumes: `tools/run_mfrinstseg_core_seed.ps1`.
- Produces: three directly runnable fixed-seed scripts that accept and forward optional path and launcher overrides.

- [ ] **Step 1: Write the failing wrapper test**

Append:

```python
@pytest.mark.parametrize(
    "filename, expected_seed",
    [
        ("run_mfrinstseg_core_seed42.ps1", 42),
        ("run_mfrinstseg_core_seed2025.ps1", 2025),
        ("run_mfrinstseg_core_seed2027.ps1", 2027),
    ],
)
def test_fixed_seed_entrypoint_forwards_declared_seed(
    tmp_path,
    filename,
    expected_seed,
):
    fake_train, fake_test, events = _fake_launchers(tmp_path)
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
            str(PROJECT_ROOT / "tools" / filename),
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

    assert completed.returncode == 0, completed.stderr
    train_lines = events.read_text(encoding="utf-8").splitlines()[::2]
    assert [int(line.split("|")[3]) for line in train_lines] == [
        expected_seed,
        expected_seed,
        expected_seed,
    ]
    assert all(
        line.split("|")[4] == str(prepared_base) + "_seed" + str(expected_seed)
        for line in train_lines
    )
```

- [ ] **Step 2: Run the wrapper test and confirm RED**

Run:

```powershell
& "G:\ProgramData\anaconda3\envs\aagmspenv\python.exe" -m pytest `
  tests/test_multiseed_core_runner.py::test_fixed_seed_entrypoint_forwards_declared_seed -q
```

Expected: three failures because the wrappers do not exist.

- [ ] **Step 3: Create the three wrappers**

Each wrapper contains the same forwarding parameters:

```powershell
param(
    [string]$RawDatasetPath = "D:\Code\Datasets\MFRInstSeg",
    [string]$PreparedDatasetBase = "D:\Code\Datasets\MFRInstSeg_BrepMFR_SPEPCAAG",
    [string]$ResultBasePath = "D:\Code\BrepMFR\results",
    [string]$PythonExe = "G:\ProgramData\anaconda3\envs\aagmspenv\python.exe",
    [string]$TrainingLauncher = "",
    [string]$TestLauncher = ""
)

$ErrorActionPreference = "Stop"
$Runner = Join-Path $PSScriptRoot "run_mfrinstseg_core_seed.ps1"
& $Runner -Seed FIXED_SEED @PSBoundParameters
```

Create the files with `FIXED_SEED` replaced literally:

- `run_mfrinstseg_core_seed42.ps1`: `42`
- `run_mfrinstseg_core_seed2025.ps1`: `2025`
- `run_mfrinstseg_core_seed2027.ps1`: `2027`

- [ ] **Step 4: Run all multi-seed runner tests**

Run:

```powershell
& "G:\ProgramData\anaconda3\envs\aagmspenv\python.exe" -m pytest `
  tests/test_multiseed_core_runner.py -q
```

Expected: `5 passed`.

- [ ] **Step 5: Run related experiment-launcher regression tests**

Run:

```powershell
& "G:\ProgramData\anaconda3\envs\aagmspenv\python.exe" -m pytest `
  tests/test_multiseed_core_runner.py `
  tests/test_experiment_configuration.py `
  tests/test_experiment_commands.py -q
```

Expected: all selected tests pass.

- [ ] **Step 6: Verify and commit**

```powershell
git diff --check
git status --short
git add `
  tools/run_mfrinstseg_core_seed42.ps1 `
  tools/run_mfrinstseg_core_seed2025.ps1 `
  tools/run_mfrinstseg_core_seed2027.ps1 `
  tests/test_multiseed_core_runner.py
git commit -m "feat: add fixed multi-seed core experiment launchers"
```
