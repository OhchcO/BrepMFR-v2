# Parameter-Surface Instance Combination Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add four explicit point-cloud instance aliases and a resume-safe seed42 A-E runner for the best parameter-surface configuration.

**Architecture:** Extend the central alias-to-switch mapping without changing existing aliases or model internals. Drive the five controlled experiments through a dedicated PowerShell wrapper that delegates training and testing to the existing launchers and performs a bias-off test only for the full predicted-instance model.

**Tech Stack:** Python 3, pytest, PowerShell 5+, PyTorch Lightning experiment entry points

## Global Constraints

- Work only on `tmp/param-surface-instance-combination`.
- Do not run formal training or full-dataset conversion.
- Preserve every existing experiment alias and its behavior.
- Reuse `MFRInstSeg_BrepMFR_ParamSurface_v4_seed42` unless rebuilding is explicitly requested.
- Use TDD: each production change follows an observed failing test.

---

### Task 1: Point-cloud instance aliases

**Files:**
- Modify: `tests/test_experiment_configuration.py`
- Modify: `segmentation.py`
- Modify: `tools/run_mfrinstseg_experiments.ps1`

**Interfaces:**
- Consumes: `ExperimentSwitches` and `resolve_experiment_alias(alias)`.
- Produces: aliases `pc_dynamic_no_loss`, `pc_affinity_only`, `pc_instance`, and `pc_oracle` accepted by Python and PowerShell entry points.

- [ ] Add four expected rows to `test_alias_mapping` with UV false, point true, cylindrical false, and the approved mode/scale/strategy values.
- [ ] Add the four aliases to the exact accepted-mode assertion.
- [ ] Run `pytest tests/test_experiment_configuration.py::test_alias_mapping tests/test_experiment_configuration.py::test_launcher_accepts_all_formal_experiment_modes -q` and observe failure because the aliases are absent.
- [ ] Add the four aliases to `EXPERIMENT_ALIASES`, `_EXPERIMENT_SWITCHES`, the PowerShell `ValidateSet`, and the `-All` matrix.
- [ ] Re-run the focused tests and confirm they pass.

### Task 2: Seed42 A-E runner

**Files:**
- Create: `tests/test_param_surface_instance_seed42_runner.py`
- Create: `tools/run_mfrinstseg_param_surface_instance_seed42.ps1`

**Interfaces:**
- Consumes: `run_mfrinstseg_experiments.ps1` and `run_mfrinstseg_test.ps1`.
- Produces: a single seed42 entry point with fake-launcher-testable `TrainingLauncher` and `TestLauncher` parameters.

- [ ] Write a failing integration test whose fake trainer records the mode, mapping, and geometry arguments and creates a checkpoint under the expected result root.
- [ ] Assert A-E order, default mapping for A, `positive_only` for B-E, param-surface/intrinsic/untrimmed/gated settings, five normal tests, and one `InstanceBiasScaleOverride=0` test for `pc_instance`.
- [ ] Run `pytest tests/test_param_surface_instance_seed42_runner.py -q` and observe failure because the runner does not exist.
- [ ] Implement the runner with existing-launcher delegation, fresh-checkpoint detection, complete-result skipping, and opt-in first-run rebuild.
- [ ] Re-run the new test and confirm it passes.

### Task 3: Regression and static verification

**Files:**
- Verify: `segmentation.py`
- Verify: `tools/run_mfrinstseg_experiments.ps1`
- Verify: `tools/run_mfrinstseg_param_surface_instance_seed42.ps1`
- Verify: `tests/test_experiment_configuration.py`
- Verify: `tests/test_param_surface_instance_seed42_runner.py`

**Interfaces:**
- Consumes: all Task 1-2 outputs.
- Produces: verified branch state ready for user-run training.

- [ ] Run the focused experiment-configuration and new-runner tests.
- [ ] Run the complete pytest suite without launching formal training.
- [ ] Parse both PowerShell launchers with `System.Management.Automation.Language.Parser` and require zero errors.
- [ ] Run `git diff --check` and inspect `git diff --stat` plus `git status --short --branch`.
- [ ] Commit the implementation on the temporary branch without merging or deleting it.
