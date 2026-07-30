# Final Multi-Seed Experiment Matrix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Seed42, Seed2025, Seed2026, and Seed2027 execute the same
ten-experiment thesis matrix, including the final SPE-PCAAG plus instance-bias
model.

**Architecture:** Keep `run_mfrinstseg_core_seed.ps1` as the only experiment
matrix implementation. Add one explicit `spe_instance` Python alias, then make
all four fixed-seed scripts thin wrappers that forward only their seed and
shared paths to the generic runner.

**Tech Stack:** Python, PyTorch configuration, PowerShell, pytest.

## Global Constraints

- Do not change the existing 10D point representation or geometry extraction.
- Do not change instance-bias equations, affinity loss, sampling, split logic,
  training epochs, or test metrics.
- Use `positive_only` for `dynamic_no_loss`, `affinity_only`, `instance`,
  `oracle`, and `spe_instance`.
- Use cache base
  `D:\Code\Datasets\MFRInstSeg_BrepMFR_SPEPCAAG10D`.
- Each fixed-seed wrapper invokes the generic runner exactly once.
- Run unit/smoke tests only; do not run training or full conversion.

---

### Task 1: Add the explicit final combined experiment alias

**Files:**
- Modify: `tests/test_experiment_configuration.py`
- Modify: `tests/test_experiment_commands.py`
- Modify: `segmentation.py`
- Modify: `tools/run_mfrinstseg_experiments.ps1`

**Interfaces:**
- Consumes: `ExperimentSwitches` and `resolve_experiment_alias(alias)`.
- Produces: accepted alias `spe_instance` with
  `ExperimentSwitches("dynamic_affinity", False, True, True, 1.0,
  "late_gated")`.

- [ ] **Step 1: Write failing alias and launcher tests**

Add this parameter row to `test_alias_mapping`:

```python
(
    "spe_instance",
    "dynamic_affinity",
    False,
    True,
    True,
    1.0,
    "late_gated",
),
```

Add `spe_instance` to the expected Python aliases and PowerShell
`ValidateSet` order.

- [ ] **Step 2: Run the focused tests and verify failure**

Run:

```powershell
G:\ProgramData\anaconda3\envs\aagmspenv\python.exe -m pytest `
  tests/test_experiment_configuration.py -q
```

Expected: failure because `spe_instance` is not accepted or mapped.

- [ ] **Step 3: Implement the minimal alias**

Add to `EXPERIMENT_ALIASES` and `_EXPERIMENT_SWITCHES`:

```python
"spe_instance": ExperimentSwitches(
    "dynamic_affinity", False, True, True, 1.0, "late_gated"
),
```

Add `spe_instance` to both the `ValidateSet` and `-All` mode array in
`run_mfrinstseg_experiments.ps1`.

- [ ] **Step 4: Run the focused tests and verify success**

Run the command from Step 2. Expected: all tests pass.

- [ ] **Step 5: Commit the alias change**

```powershell
git add segmentation.py tools/run_mfrinstseg_experiments.ps1 `
  tests/test_experiment_configuration.py tests/test_experiment_commands.py
git commit -m "feat: add final SPE instance experiment"
```

### Task 2: Replace the generic runner with the final ten-experiment matrix

**Files:**
- Modify: `tests/test_multiseed_core_runner.py`
- Modify: `tools/run_mfrinstseg_core_seed.ps1`

**Interfaces:**
- Consumes: training launcher parameters `Mode`,
  `InstanceRelationMapping`, `Seed`, and `PreparedDatasetPath`.
- Produces: ten ordered train/test event pairs and result labels matching
  launcher naming.

- [ ] **Step 1: Replace the three-experiment test with the exact final matrix**

Define:

```python
EXPECTED_FINAL_MATRIX = [
    ("topo_aag", "signed"),
    ("uv_aag", "signed"),
    ("pc_aag", "signed"),
    ("cyl_aag", "signed"),
    ("spe_pcaag", "signed"),
    ("dynamic_no_loss", "positive_only"),
    ("affinity_only", "positive_only"),
    ("instance", "positive_only"),
    ("oracle", "positive_only"),
    ("spe_instance", "positive_only"),
]
```

Assert exactly twenty alternating train/test events, the same seed on every
event, and one shared `<prepared-base>_seed<seed>` path.

- [ ] **Step 2: Add a default-cache-base source test**

Assert the generic runner contains:

```text
D:\Code\Datasets\MFRInstSeg_BrepMFR_SPEPCAAG10D
```

and does not contain the old default ending in `SPEPCAAG`.

- [ ] **Step 3: Run the runner tests and verify failure**

Run:

```powershell
G:\ProgramData\anaconda3\envs\aagmspenv\python.exe -m pytest `
  tests/test_multiseed_core_runner.py -q
```

Expected: matrix-count/order and cache-base failures.

- [ ] **Step 4: Implement the exact PowerShell matrix**

Set:

```powershell
[string]$PreparedDatasetBase =
    "D:\Code\Datasets\MFRInstSeg_BrepMFR_SPEPCAAG10D"
```

Define ten descriptors in the order specified by
`EXPECTED_FINAL_MATRIX`. Use result labels:

```text
topo_aag
uv_aag
pc_aag
cyl_aag
spe_pcaag
dynamic_no_loss_positive
affinity_only_positive
instance_positive
oracle_positive
spe_instance_positive
```

Retain the existing fresh-checkpoint boundary and fail-fast behavior.

- [ ] **Step 5: Run the runner tests and verify success**

Run the command from Step 3. Expected: all tests pass.

- [ ] **Step 6: Commit the generic runner**

```powershell
git add tools/run_mfrinstseg_core_seed.ps1 `
  tests/test_multiseed_core_runner.py
git commit -m "feat: run final multiseed experiment matrix"
```

### Task 3: Normalize all four fixed-seed entrypoints

**Files:**
- Modify: `tools/run_mfrinstseg_core_seed42.ps1`
- Modify: `tools/run_mfrinstseg_core_seed2025.ps1`
- Create: `tools/run_mfrinstseg_core_seed2026.ps1`
- Modify: `tools/run_mfrinstseg_core_seed2027.ps1`
- Modify: `tests/test_multiseed_core_runner.py`

**Interfaces:**
- Consumes: generic runner parameters.
- Produces: four wrappers that differ only by the integer passed to `-Seed`.

- [ ] **Step 1: Extend the wrapper parameterization test**

Use:

```python
[
    ("run_mfrinstseg_core_seed42.ps1", 42),
    ("run_mfrinstseg_core_seed2025.ps1", 2025),
    ("run_mfrinstseg_core_seed2026.ps1", 2026),
    ("run_mfrinstseg_core_seed2027.ps1", 2027),
]
```

For each wrapper assert exactly ten training events matching
`EXPECTED_FINAL_MATRIX`, not eleven events from an appended one-off
`affinity_only` call.

- [ ] **Step 2: Run the wrapper tests and verify failure**

Run:

```powershell
G:\ProgramData\anaconda3\envs\aagmspenv\python.exe -m pytest `
  tests/test_multiseed_core_runner.py::test_fixed_seed_entrypoint_forwards_declared_seed `
  -q
```

Expected: Seed2026 is missing and existing wrappers emit an extra event.

- [ ] **Step 3: Make every wrapper a thin delegator**

Each wrapper keeps the shared optional parameters and contains only:

```powershell
$ErrorActionPreference = "Stop"
$Runner = Join-Path $PSScriptRoot "run_mfrinstseg_core_seed.ps1"
& $Runner -Seed <fixed-seed> @PSBoundParameters
if (-not $?) {
    throw "Seed <fixed-seed> experiment matrix failed"
}
```

Use the 10D prepared-cache base as the default parameter value in all four
files.

- [ ] **Step 4: Run the wrapper tests and verify success**

Run the command from Step 2. Expected: four parameter cases pass.

- [ ] **Step 5: Commit all fixed-seed wrappers**

```powershell
git add tools/run_mfrinstseg_core_seed42.ps1 `
  tools/run_mfrinstseg_core_seed2025.ps1 `
  tools/run_mfrinstseg_core_seed2026.ps1 `
  tools/run_mfrinstseg_core_seed2027.ps1 `
  tests/test_multiseed_core_runner.py
git commit -m "feat: add consistent fixed-seed experiment runners"
```

### Task 4: Update copyable commands and verify the final release

**Files:**
- Modify: `tools/实验.txt`
- Modify: `tests/test_experiment_commands.py`

**Interfaces:**
- Consumes: the four fixed-seed wrapper paths.
- Produces: copy-ready one-line PowerShell commands for all four seeds.

- [ ] **Step 1: Add failing command-file assertions**

Assert `tools/实验.txt` contains one copyable invocation for each:

```text
run_mfrinstseg_core_seed42.ps1
run_mfrinstseg_core_seed2025.ps1
run_mfrinstseg_core_seed2026.ps1
run_mfrinstseg_core_seed2027.ps1
```

and describes the ten-experiment matrix plus the 10D cache reuse rule.

- [ ] **Step 2: Run the command-file tests and verify failure**

Run:

```powershell
G:\ProgramData\anaconda3\envs\aagmspenv\python.exe -m pytest `
  tests/test_experiment_commands.py -q
```

Expected: missing fixed-seed command section.

- [ ] **Step 3: Add a final multi-seed section to `tools/实验.txt`**

Include four independent copy blocks:

```powershell
powershell.exe -ExecutionPolicy Bypass -File `
  "D:\Code\BrepMFR\tools\run_mfrinstseg_core_seed<seed>.ps1"
```

State that each block runs ten experiments sequentially and that the first
experiment prepares the seed-specific 10D cache.

- [ ] **Step 4: Run all affected tests**

Run:

```powershell
G:\ProgramData\anaconda3\envs\aagmspenv\python.exe -m pytest `
  tests/test_multiseed_core_runner.py `
  tests/test_experiment_configuration.py `
  tests/test_experiment_commands.py `
  tests/test_geometry_features.py `
  tests/test_mfrinstseg_to_bin_geometry.py `
  tests/test_prepare_dataset.py `
  tests/test_dataset_geometry_fields.py `
  tests/test_geometry_encoders.py -q
```

Expected: zero failures.

- [ ] **Step 5: Perform non-training release checks**

Run:

```powershell
git diff --check
git status --short
```

Parse all five PowerShell files through
`System.Management.Automation.Language.Parser.ParseFile` and assert zero
syntax errors. Do not invoke any real training launcher.

- [ ] **Step 6: Request independent code review**

Review the working-tree diff against
`docs/superpowers/specs/2026-07-30-final-experiment-matrix-design.md`. Fix all
Critical and Important findings, rerun Step 4, then report the exact command
the user should copy first.
