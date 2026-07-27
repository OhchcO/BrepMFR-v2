# Cyl + Affinity Experiment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a copy-ready `cyl_affinity` experiment that combines cylindrical positions with affinity supervision while disabling point clouds and attention-bias injection.

**Architecture:** Extend the existing formal alias mapping rather than changing model internals. Propagate the alias through the PowerShell launcher and human command file; checkpoint hyperparameters and the shared geometry-complete BIN cache retain their existing behavior.

**Tech Stack:** Python 3, argparse, pytest, PowerShell.

## Global Constraints

- Do not change model layers, dataset conversion, split membership, sampling, or existing experiment mappings.
- `cyl_affinity` maps to `dynamic_affinity`, no face point cloud, cylindrical positions enabled, and instance bias scale `0.0`.
- Do not run training or full-dataset conversion.
- Preserve unrelated modifications in `tools/prepare_dataset.py` and `tests/test_prepare_dataset.py`.

---

### Task 1: Formal alias and launcher contract

**Files:**
- Modify: `tests/test_experiment_configuration.py`
- Modify: `segmentation.py`
- Modify: `tools/run_mfrinstseg_experiments.ps1`

**Interfaces:**
- Consumes: `resolve_experiment_alias(alias: str) -> ExperimentSwitches`
- Produces: formal alias `cyl_affinity` accepted by Python and PowerShell launchers

- [ ] **Step 1: Write failing mapping and launcher tests**

Add the literal mapping:

```python
("cyl_affinity", "dynamic_affinity", False, True, 0.0)
```

to the alias table and add `"cyl_affinity"` immediately after `"cyl"` in the expected PowerShell modes.

- [ ] **Step 2: Run tests and verify the missing mode fails**

Run:

```powershell
pytest tests/test_experiment_configuration.py -q
```

Expected: failures report unsupported `cyl_affinity` and/or the missing PowerShell ValidateSet value.

- [ ] **Step 3: Add the minimal production mapping**

Add `cyl_affinity` to `EXPERIMENT_ALIASES`, `_EXPERIMENT_SWITCHES`, the launcher `ValidateSet`, and its `$All` mode list. Use:

```python
ExperimentSwitches("dynamic_affinity", False, True, 0.0)
```

- [ ] **Step 4: Run the configuration tests**

Run:

```powershell
pytest tests/test_experiment_configuration.py -q
```

Expected: all tests pass.

### Task 2: Copy-ready experiment command

**Files:**
- Modify: `tests/test_experiment_commands.py`
- Modify: `tools/实验.txt`

**Interfaces:**
- Consumes: `tools/run_mfrinstseg_experiments.ps1 -Mode cyl_affinity`
- Produces: one copyable training block and one best-checkpoint test block

- [ ] **Step 1: Write the failing command-file expectation**

Insert `"cyl_affinity"` after `"cyl"` in `EXPECTED_MODES`.

- [ ] **Step 2: Run the command-file tests and verify failure**

Run:

```powershell
pytest tests/test_experiment_commands.py -q
```

Expected: the parsed mode list lacks `cyl_affinity`.

- [ ] **Step 3: Add the command block**

Copy the existing `cyl` command shape, change mode/result names to `cyl_affinity`, and keep the same raw dataset, prepared dataset, sample percent, seed, batch size, and worker count.

- [ ] **Step 4: Run command-file tests**

Run:

```powershell
pytest tests/test_experiment_commands.py -q
```

Expected: both tests pass, including PowerShell syntax parsing.

### Task 3: Regression verification

**Files:**
- Verify only: all files changed by Tasks 1 and 2

**Interfaces:**
- Consumes: completed implementation
- Produces: fresh test evidence without training

- [ ] **Step 1: Run focused regression tests**

```powershell
pytest tests/test_experiment_configuration.py tests/test_experiment_commands.py -q
```

- [ ] **Step 2: Compile the Python entry point**

```powershell
python -m py_compile segmentation.py
```

- [ ] **Step 3: Review the exact diff**

```powershell
git diff -- segmentation.py tools/run_mfrinstseg_experiments.ps1 tools/实验.txt tests/test_experiment_configuration.py tests/test_experiment_commands.py docs/superpowers/specs/2026-07-27-cyl-affinity-experiment-design.md docs/superpowers/plans/2026-07-27-cyl-affinity-experiment.md
```

Confirm no existing alias changed and no unrelated dirty files entered the patch.
