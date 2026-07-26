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
- Create: `tools/mfrinstseg_experiment_commands.txt`

**Interfaces:**
- Consumes: one editable PowerShell variable `$Mode`
- Produces: manual training command, latest-best-checkpoint resolution, and manual full-test command

- [ ] **Step 1: Write the command-file contract test**

Create `tests/test_experiment_commands.py` to require:

```python
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
COMMAND_FILE = PROJECT_ROOT / "tools" / "mfrinstseg_experiment_commands.txt"


def test_command_file_covers_every_formal_experiment():
    text = COMMAND_FILE.read_text(encoding="utf-8")
    for mode in (
        "baseline",
        "dynamic_no_loss",
        "affinity_only",
        "instance",
        "oracle",
        "cyl",
        "point",
        "spe_pcaag",
        "full",
    ):
        assert mode in text


def test_command_file_uses_one_mode_and_latest_best_checkpoint():
    text = COMMAND_FILE.read_text(encoding="utf-8")
    assert '$Mode = "dynamic_no_loss"' in text
    assert "run_mfrinstseg_experiments.ps1" in text
    assert "-Mode $Mode" in text
    assert 'MFRInstSeg_1pct_${Mode}_seed2026' in text
    assert 'Get-ChildItem' in text
    assert 'best-*.ckpt' in text
    assert "run_mfrinstseg_test.ps1" in text
    assert "-Checkpoint $Checkpoint" in text
```

- [ ] **Step 2: Run the new test and verify RED**

Run:

```powershell
python -m pytest tests/test_experiment_commands.py -v
```

Expected: failure because `tools/mfrinstseg_experiment_commands.txt` does not exist.

- [ ] **Step 3: Write the TXT command template**

Create a UTF-8 TXT containing:

- the recommended mode sequence;
- `$Mode = "dynamic_no_loss"` as the only experiment selector;
- the existing training launcher with the fixed shared paths and parameters;
- checkpoint lookup beneath `results\MFRInstSeg_1pct_${Mode}_seed2026`;
- a guard that throws if no `best-*.ckpt` exists;
- the existing test launcher using `$Checkpoint`;
- a note that Oracle is a GT-only upper bound.

The lookup must use:

```powershell
$ResultRoot = "D:\Code\BrepMFR\results\MFRInstSeg_1pct_${Mode}_seed2026"
$Checkpoint = Get-ChildItem -LiteralPath $ResultRoot -Recurse -Filter "best-*.ckpt" |
  Sort-Object LastWriteTime -Descending |
  Select-Object -First 1 -ExpandProperty FullName
if (-not $Checkpoint) {
  throw "No best checkpoint found for mode: $Mode"
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
git add tools/mfrinstseg_experiment_commands.txt tests/test_experiment_commands.py
git commit -m "docs: add copy-ready MFRInstSeg experiment commands"
```

