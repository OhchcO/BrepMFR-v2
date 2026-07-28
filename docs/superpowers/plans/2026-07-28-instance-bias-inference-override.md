# Instance Bias Inference Override Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a backward-compatible inference-only instance-bias scale override.

**Architecture:** Apply the optional override to the loaded
`InstanceRelationBias` module before evaluation. Keep command parsing,
mutation/validation, and result reporting explicit and independently testable.

**Tech Stack:** Python, PyTorch, pytest, Windows PowerShell.

## Global Constraints

- Do not train models or run the full dataset.
- Do not regenerate or modify BIN caches.
- Omitted override must preserve existing behavior.
- Override values must be finite and non-negative.
- Preserve all existing dirty-worktree changes.

---

### Task 1: Python evaluator override

**Files:**
- Modify: `tools/test_mfrinstseg.py`
- Modify: `tests/test_instance_evaluation.py`

**Interfaces:**
- Consumes: a loaded `BrepSeg`-compatible model and `float | None`.
- Produces: `apply_instance_bias_scale_override(model, override) -> tuple[float, float, bool]`.

- [ ] Add tests showing `None` preserves the scale, zero disables it, positive
      values replace it, and invalid values raise `ValueError`.
- [ ] Run the focused tests and confirm they fail because the helper is absent.
- [ ] Implement the helper and CLI option.
- [ ] Apply the override immediately after checkpoint loading.
- [ ] Record checkpoint/effective/overridden values in JSON and text logs.
- [ ] Route overridden evaluations to a scale-specific child directory unless
      the caller explicitly supplies `--output-dir`.
- [ ] Run the focused tests and confirm they pass.

### Task 2: PowerShell forwarding

**Files:**
- Modify: `tools/run_mfrinstseg_test.ps1`
- Modify: `tests/test_experiment_commands.py`

**Interfaces:**
- Consumes: nullable `-InstanceBiasScaleOverride`.
- Produces: Python `--instance-bias-scale-override` only when explicitly set.

- [ ] Add a wrapper integration test using a temporary fake Python executable
      to capture forwarded arguments.
- [ ] Run the focused test and confirm it fails because the parameter is absent.
- [ ] Add the nullable PowerShell parameter, validate it, and append the Python
      argument only when present.
- [ ] Run the focused test and PowerShell syntax test.

### Task 3: Verification

**Files:**
- Verify only; no additional production changes expected.

**Interfaces:**
- Consumes: completed Tasks 1-2.
- Produces: commands for saved-scale and zero-scale evaluation.

- [ ] Run the focused Python tests.
- [ ] Run Python compilation for the evaluator.
- [ ] Parse the PowerShell wrapper using the native PowerShell parser.
- [ ] Review `git diff` for scope and preservation of unrelated changes.
