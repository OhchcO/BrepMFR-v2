# Parameter-Surface Seed42 Runner Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add one directly runnable seed42 PowerShell experiment matrix.

**Architecture:** A thin orchestration script delegates each training run to
`run_mfrinstseg_experiments.ps1`, discovers only the fresh checkpoint, and
delegates evaluation to `run_mfrinstseg_test.ps1`.

**Tech Stack:** PowerShell, pytest static contract test.

## Global Constraints

- Do not execute training during implementation.
- Use the dedicated parameter-surface v4 prepared-data directory.
- Stop on the first failed command.

---

### Task 1: Runner contract and implementation

**Files:**
- Create: `tools/run_mfrinstseg_param_surface_seed42.ps1`
- Create: `tests/test_param_surface_seed42_runner.py`

**Interfaces:**
- Consumes: existing training and test PowerShell launchers.
- Produces: a directly executable three-experiment seed42 runner.

- [ ] Write a static test for experiment order, switches, paths, and launchers.
- [ ] Run it and verify it fails because the runner does not exist.
- [ ] Implement the runner with fresh-checkpoint discovery and fail-fast checks.
- [ ] Run the static test and verify it passes.
- [ ] Parse the script using the PowerShell AST parser without executing it.
- [ ] Run the relevant launcher/configuration unit tests.
