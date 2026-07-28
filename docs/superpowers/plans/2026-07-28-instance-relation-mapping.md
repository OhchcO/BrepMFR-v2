# Instance Relation Mapping Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a backward-compatible `signed/positive_only` switch to the existing instance relation bias.

**Architecture:** Transform the already-computed signed relation immediately before pair masking. Thread one string configuration value through the existing training, checkpoint, evaluator, and PowerShell paths.

**Tech Stack:** Python, PyTorch, argparse, pytest, Windows PowerShell.

## Global Constraints

- Do not add model modules, losses, stages, or data fields.
- `signed` must preserve current numerical behavior.
- Old checkpoints must load as `signed`.
- Do not train or run the full dataset.
- Preserve all existing dirty-worktree changes.

---

### Task 1: Relation mapping behavior

**Files:**
- Modify: `models/modules/instance_bias.py`
- Modify: `tests/test_instance_bias.py`

**Interfaces:**
- Consumes: `relation_mapping: str` in `InstanceRelationBias.__init__`.
- Produces: signed values unchanged or negative values clamped to zero.

- [ ] Add failing tests for dynamic signed compatibility, dynamic
      positive-only clamping, Oracle different-instance neutrality, and invalid
      mappings.
- [ ] Run the focused tests and confirm RED.
- [ ] Add mapping constants, constructor validation, and the one-line relation
      transform.
- [ ] Run the focused tests and confirm GREEN.

### Task 2: Configuration and provenance

**Files:**
- Modify: `segmentation.py`
- Modify: `models/brepseg_model.py`
- Modify: `models/modules/brep_encoder.py`
- Modify: `tools/test_mfrinstseg.py`
- Modify: `tests/test_experiment_configuration.py`

**Interfaces:**
- Consumes: `--instance_relation_mapping signed|positive_only`.
- Produces: checkpoint and test-log field `instance_relation_mapping`.

- [ ] Add failing parser, model propagation, legacy default, and test-log
      provenance assertions.
- [ ] Run the focused tests and confirm RED.
- [ ] Thread the value through model construction and evaluator reporting.
- [ ] Run the focused tests and confirm GREEN.

### Task 3: PowerShell launcher

**Files:**
- Modify: `tools/run_mfrinstseg_experiments.ps1`
- Modify: `tests/test_experiment_configuration.py`

**Interfaces:**
- Consumes: `-InstanceRelationMapping signed|positive_only`.
- Produces: Python argument and `_positive` result label for positive-only.

- [ ] Add a failing launcher behavior test that captures actual arguments.
- [ ] Run the focused test and confirm RED.
- [ ] Add the validated parameter, argument forwarding, and result naming.
- [ ] Run the focused test and confirm GREEN.

### Task 4: Verification

**Files:**
- Verify only.

**Interfaces:**
- Produces: copyable `oracle positive_only` training command.

- [ ] Run all related instance-bias/configuration tests.
- [ ] Compile changed Python files.
- [ ] Parse the PowerShell launcher with the native parser.
- [ ] Run `git diff --check` and review scoped differences.
