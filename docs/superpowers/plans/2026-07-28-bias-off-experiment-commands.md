# Latest Bias-Off Evaluation Commands Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Append three copyable bias-disabled evaluation commands for the latest deterministic dynamic, instance, and oracle checkpoints.

**Architecture:** Keep the existing experiment command file unchanged except for a final test-only section. Use explicit checkpoint paths instead of dynamic discovery, then protect the three-command contract with a focused parser test.

**Tech Stack:** PowerShell, pytest, Python regular expressions.

## Global Constraints

- Do not start training or model evaluation during implementation.
- Do not rebuild or regenerate BIN data.
- Modify only `tools/实验.txt` and `tests/test_experiment_commands.py`.
- Use `D:\Code\Datasets\MFRInstSeg_BrepMFR_SPEPCAAG`.
- Every appended command must pass `-InstanceBiasScaleOverride 0`, `-BatchSize 16`, and `-NumWorkers 0`.
- Use the three exact checkpoints approved in the design specification.

---

### Task 1: Append and protect the latest bias-off evaluation blocks

**Files:**
- Modify: `tests/test_experiment_commands.py`
- Modify: `tools/实验.txt`

**Interfaces:**
- Consumes: the existing `COMMAND_FILE` constant and PowerShell test launcher.
- Produces: three independently copyable PowerShell blocks whose results are written by the launcher to `test_bias_scale_0`.

- [ ] **Step 1: Write the failing test**

Add this test to `tests/test_experiment_commands.py`:

```python
def test_experiment_file_has_latest_bias_off_evaluations():
    text = COMMAND_FILE.read_text(encoding="utf-8")
    expected = [
        (
            "dynamic_no_loss_positive",
            r"D:\Code\BrepMFR\results\MFRInstSeg_1pct_dynamic_no_loss_positive_seed2026\0728\174518\best-080-0.165093.ckpt",
        ),
        (
            "instance_positive",
            r"D:\Code\BrepMFR\results\MFRInstSeg_1pct_instance_positive_seed2026\0728\164959\best-095-0.142707.ckpt",
        ),
        (
            "oracle_positive",
            r"D:\Code\BrepMFR\results\MFRInstSeg_1pct_oracle_positive_seed2026\0728\164959\best-086-0.152804.ckpt",
        ),
    ]

    for label, checkpoint in expected:
        marker = "# Bias OFF: " + label
        start = text.index(marker)
        next_marker = text.find("# Bias OFF:", start + len(marker))
        block = text[start : next_marker if next_marker != -1 else len(text)]
        assert '-Checkpoint "' + checkpoint + '" `' in block
        assert "-InstanceBiasScaleOverride 0" in block
        assert "-BatchSize 16" in block
        assert "-NumWorkers 0" in block
```

- [ ] **Step 2: Run the focused test to verify it fails**

Run:

```powershell
& "G:\ProgramData\anaconda3\envs\aagmspenv\python.exe" -m pytest `
  tests/test_experiment_commands.py::test_experiment_file_has_latest_bias_off_evaluations -q
```

Expected: FAIL because `# Bias OFF: dynamic_no_loss_positive` is not yet present.

- [ ] **Step 3: Append the minimal PowerShell commands**

Append these three blocks to `tools/实验.txt`:

```powershell
# ============================================================
# Bias OFF: dynamic_no_loss_positive
# Same trained checkpoint; disable only residual relation bias at test time.
# ============================================================

powershell.exe -ExecutionPolicy Bypass -File `
  "D:\Code\BrepMFR\tools\run_mfrinstseg_test.ps1" `
  -Checkpoint "D:\Code\BrepMFR\results\MFRInstSeg_1pct_dynamic_no_loss_positive_seed2026\0728\174518\best-080-0.165093.ckpt" `
  -PreparedDataset "D:\Code\Datasets\MFRInstSeg_BrepMFR_SPEPCAAG" `
  -InstanceBiasScaleOverride 0 `
  -BatchSize 16 `
  -NumWorkers 0


# ============================================================
# Bias OFF: instance_positive
# Same trained checkpoint; disable only residual relation bias at test time.
# ============================================================

powershell.exe -ExecutionPolicy Bypass -File `
  "D:\Code\BrepMFR\tools\run_mfrinstseg_test.ps1" `
  -Checkpoint "D:\Code\BrepMFR\results\MFRInstSeg_1pct_instance_positive_seed2026\0728\164959\best-095-0.142707.ckpt" `
  -PreparedDataset "D:\Code\Datasets\MFRInstSeg_BrepMFR_SPEPCAAG" `
  -InstanceBiasScaleOverride 0 `
  -BatchSize 16 `
  -NumWorkers 0


# ============================================================
# Bias OFF: oracle_positive
# Same trained checkpoint; disable only residual relation bias at test time.
# ============================================================

powershell.exe -ExecutionPolicy Bypass -File `
  "D:\Code\BrepMFR\tools\run_mfrinstseg_test.ps1" `
  -Checkpoint "D:\Code\BrepMFR\results\MFRInstSeg_1pct_oracle_positive_seed2026\0728\164959\best-086-0.152804.ckpt" `
  -PreparedDataset "D:\Code\Datasets\MFRInstSeg_BrepMFR_SPEPCAAG" `
  -InstanceBiasScaleOverride 0 `
  -BatchSize 16 `
  -NumWorkers 0
```

- [ ] **Step 4: Run focused and command-file tests**

Run:

```powershell
& "G:\ProgramData\anaconda3\envs\aagmspenv\python.exe" -m pytest `
  tests/test_experiment_commands.py -q
```

Expected: all tests in `tests/test_experiment_commands.py` pass.

- [ ] **Step 5: Verify the diff and commit**

Run:

```powershell
git diff --check
git diff -- tools/实验.txt tests/test_experiment_commands.py
git add tools/实验.txt tests/test_experiment_commands.py
git commit -m "docs: add latest bias-off evaluation commands"
```
