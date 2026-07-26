# MFRInstSeg Manual Experiment Launch Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the missing `dynamic_no_loss` and `oracle` experiment switches and provide one copy-ready TXT template for manually training and testing each MFRInstSeg ablation.

**Architecture:** The existing experiment-alias mapping remains the single source of truth for model behavior, while the existing PowerShell launcher exposes the aliases as `-Mode` values. A standalone TXT file supplies a one-variable PowerShell workflow that reuses the shared BIN cache and locates the latest best checkpoint for the selected mode.

**Tech Stack:** Python 3, pytest, PowerShell 5+, PyTorch Lightning command-line entry points.

## Global Constraints

- Do not execute training, full test evaluation, or STEP-to-BIN conversion.
- Keep manual one-group-at-a-time execution; do not automatically chain experiments.
- Fix `SamplePercent=0.01`, `Seed=2026`, `NumWorkers=0`, and `MaxEpochs=100` in the copy-ready example.
- Reuse `D:\Code\Datasets\MFRInstSeg_BrepMFR_SPEPCAAG` without `-RebuildPreparedData`.
- Preserve the user's existing local change that sets the launcher default `MaxEpochs` to 100.
- Oracle is an upper-bound experiment requiring GT instance annotations at evaluation.

---

### Task 1: Expose the missing experiment aliases

**Files:**
- Modify: `tests/test_experiment_configuration.py`
- Modify: `segmentation.py`
- Modify: `tools/run_mfrinstseg_experiments.ps1`

**Interfaces:**
- Consumes: `resolve_experiment_alias(alias: str) -> ExperimentSwitches`
- Produces: `dynamic_no_loss` and `oracle` accepted by both `--experiment_alias` and PowerShell `-Mode`

- [ ] **Step 1: Extend the alias-mapping test first**

Add these cases to `test_alias_mapping`:

```python
("dynamic_no_loss", "dynamic", False, False, 1.0),
("oracle", "oracle", False, False, 1.0),
```

Also add a test that reads `tools/run_mfrinstseg_experiments.ps1` and asserts both names appear in its first `ValidateSet`.

- [ ] **Step 2: Run the focused test and verify RED**

Run:

```powershell
G:\ProgramData\anaconda3\envs\aagmspenv\python.exe -m pytest tests/test_experiment_configuration.py -k "alias_mapping or launcher_accepts" -v
```

Expected: failure because the Python alias table and PowerShell whitelist do not yet contain the new names. If pytest is unavailable in `aagmspenv`, use the repository Python that has pytest for this import-safe configuration test.

- [ ] **Step 3: Implement the minimal alias mappings**

Add both names to `EXPERIMENT_ALIASES` and `_EXPERIMENT_SWITCHES`:

```python
"dynamic_no_loss": ExperimentSwitches("dynamic", False, False, 1.0),
"oracle": ExperimentSwitches("oracle", False, False, 1.0),
```

Add both values to the PowerShell `Mode` `ValidateSet`. Extend the `-All` list in the documented experimental order without changing the manual `-Mode` behavior.

- [ ] **Step 4: Run the focused test and verify GREEN**

Repeat the command from Step 2 and confirm all selected cases pass.

- [ ] **Step 5: Commit Task 1**

```powershell
git add segmentation.py tools/run_mfrinstseg_experiments.ps1 tests/test_experiment_configuration.py
git commit -m "feat: expose missing instance-bias experiments"
```

### Task 2: Add the copy-ready train/test command file

**Files:**
- Create: `tests/test_experiment_commands.py`
- Modify: `tools/实验.txt`

**Interfaces:**
- Consumes: one editable PowerShell variable `$Mode`
- Produces: manual training command, latest-best-checkpoint resolution, and manual full-test command

- [ ] **Step 1: Write the command-file contract test**

Create `tests/test_experiment_commands.py` to require:

```python
import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
COMMAND_FILE = PROJECT_ROOT / "tools" / "实验.txt"


def test_experiment_file_has_one_copyable_train_and_test_block_per_mode():
    text = COMMAND_FILE.read_text(encoding="utf-8")
    expected_modes = [
        "baseline",
        "dynamic_no_loss",
        "affinity_only",
        "instance",
        "oracle",
        "cyl",
        "point",
        "spe_pcaag",
        "full",
    ]
    actual_modes = re.findall(r"(?m)^\s+-Mode ([a-z_]+) `?$", text)

    assert actual_modes == expected_modes
    assert text.count("run_mfrinstseg_test.ps1") == len(expected_modes)
    assert text.count("Get-ChildItem -LiteralPath") == len(expected_modes)
    assert "最优best模型的完整路径" not in text
```

- [ ] **Step 2: Run the new test and verify RED**

Run:

```powershell
python -m pytest tests/test_experiment_commands.py -v
```

Expected: failure because the existing `tools/实验.txt` covers only part of the experiment matrix.

- [ ] **Step 3: Write the TXT command template**

Update the user-specified UTF-8 TXT with:

- the recommended mode sequence;
- one self-contained copyable block per experiment Mode;
- the existing training launcher with the fixed shared paths and parameters in every block;
- checkpoint lookup beneath the corresponding `results\MFRInstSeg_1pct_<mode>_seed2026`;
- a guard that throws if no `best-*.ckpt` exists;
- the existing test launcher using `$Checkpoint`;
- a note that Oracle is a GT-only upper bound.

Each block's lookup follows this form with its literal Mode:

```powershell
$ResultRoot = "D:\Code\BrepMFR\results\MFRInstSeg_1pct_dynamic_no_loss_seed2026"
$Checkpoint = Get-ChildItem -LiteralPath $ResultRoot -Recurse -Filter "best-*.ckpt" |
  Sort-Object LastWriteTime -Descending |
  Select-Object -First 1 -ExpandProperty FullName
if (-not $Checkpoint) {
  throw "No best checkpoint found for dynamic_no_loss"
}
```

- [ ] **Step 4: Verify tests and PowerShell syntax without execution**

Run:

```powershell
python -m pytest tests/test_experiment_commands.py -v
```

Extract only PowerShell code from the TXT into a temporary `.ps1`, then parse it with:

```powershell
$Errors = $null
[System.Management.Automation.Language.Parser]::ParseFile(
  $TemporaryScript,
  [ref]$null,
  [ref]$Errors
) | Out-Null
if ($Errors.Count -ne 0) { throw ($Errors | Out-String) }
```

Do not invoke either launcher.

- [ ] **Step 5: Run final focused verification**

Run the alias configuration tests, command-file tests, Python compilation, `git diff --check`, and the PowerShell parser check. Confirm zero failures.

- [ ] **Step 6: Commit Task 2**

```powershell
git add tools/实验.txt tests/test_experiment_commands.py
git commit -m "docs: add copy-ready MFRInstSeg experiment commands"
```
