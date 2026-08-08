# Repository-Local AAG Assets Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace every old external AAG asset reference with versioned repository-local defaults and verify one real STEP-to-BIN conversion.

**Architecture:** Store the extractor and schema beside the conversion tools, resolve defaults relative to Python module paths or `$PSScriptRoot`, and retain explicit overrides. Keep `TopologyUtils` as the installed OCC implementation and preserve content-based cache fingerprinting.

**Tech Stack:** Python 3.8, pythonocc-core, occwl, DGL, pytest, PowerShell 5, Bash

## Global Constraints

- `tools/AAGExtractor.py` and `tools/schema.json` are the only default AAG assets.
- `OCC.Extend.TopologyUtils` remains unchanged.
- The checked-in schema uses `num_srf_u=5`, `num_srf_v=5`, and `num_crv_u=5`.
- Explicit extractor/schema overrides remain supported.
- No tracked file may reference the predecessor checkout after migration.
- Do not run full dataset conversion or formal training.

---

### Task 1: Canonical assets and schema contract

**Files:**
- Add: `tools/AAGExtractor.py`
- Add: `tools/schema.json`
- Modify: `tests/test_mfrinstseg_to_bin_geometry.py`

**Interfaces:**
- Consumes: `load_schema(path)` and `load_extractor(path)`.
- Produces: valid repository-local extractor and schema assets.

- [ ] **Step 1: Write failing asset tests**

Add tests that resolve `PROJECT_ROOT / "tools" / ...`, assert
`load_schema` returns UV dimensions `(5, 5, 5)`, and assert the dynamically
loaded module exposes `AAGExtractor` and `TopologyUtils`.

- [ ] **Step 2: Run tests and verify RED**

```powershell
& 'F:\ProgramData\anaconda3\envs\aagmspenv\python.exe' -m pytest `
  tests/test_mfrinstseg_to_bin_geometry.py -q
```

Expected: schema test fails because `num_crv_u` is `0`.

- [ ] **Step 3: Fix the schema and add both assets**

Change only `UV-grid.num_crv_u` from `0` to `5`; retain the supplied extractor
implementation and its `from OCC.Extend import TopologyUtils` dependency.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run the Task 1 command and require zero failures.

- [ ] **Step 5: Commit canonical assets**

```powershell
git add tools/AAGExtractor.py tools/schema.json tests/test_mfrinstseg_to_bin_geometry.py
git commit -m "feat: version local AAG extraction assets"
```

---

### Task 2: Python defaults

**Files:**
- Modify: `tools/mfrinstseg_to_bin.py`
- Modify: `tools/prepare_dataset.py`
- Modify: `segmentation.py`
- Modify: `tests/test_mfrinstseg_to_bin_geometry.py`
- Modify: `tests/test_prepare_dataset.py`
- Modify: `tests/test_experiment_configuration.py`

**Interfaces:**
- Produces: `DEFAULT_EXTRACTOR_FILE` and `DEFAULT_SCHEMA_FILE` paths derived from repository code locations.
- Preserves: all existing `--extractor-file`, `--schema-file`, `--extractor_file`, and `--schema_file` overrides.

- [ ] **Step 1: Write failing parser-default tests**

Assert each parser resolves omitted asset arguments to:

```python
PROJECT_ROOT / "tools" / "AAGExtractor.py"
PROJECT_ROOT / "tools" / "schema.json"
```

Change the preparation parser test to verify these flags are optional while
explicit temporary paths still win.

- [ ] **Step 2: Run parser tests and verify RED**

```powershell
& 'F:\ProgramData\anaconda3\envs\aagmspenv\python.exe' -m pytest `
  tests/test_mfrinstseg_to_bin_geometry.py `
  tests/test_prepare_dataset.py `
  tests/test_experiment_configuration.py -q
```

Expected: defaults still resolve to the old absolute paths or `None`, and the
preparation parser still requires both flags.

- [ ] **Step 3: Implement repository-relative Python defaults**

Use `Path(__file__).resolve()` constants in each entry point. Do not derive
defaults from `Path.cwd()`. Keep selected paths in preparation fingerprints.

- [ ] **Step 4: Run parser tests and verify GREEN**

Run the Task 2 test command and require zero failures.

- [ ] **Step 5: Commit Python migration**

```powershell
git add segmentation.py tools/mfrinstseg_to_bin.py tools/prepare_dataset.py tests
git commit -m "fix: default to repository AAG assets"
```

---

### Task 3: Launcher and documentation migration

**Files:**
- Modify: `tools/run_mfrinstseg_experiments.ps1`
- Modify: `tools/run_mfrinstseg_test.ps1`
- Modify: `tools/prepare_mfrinstseg.sh`
- Modify: relevant files under `docs/superpowers/specs/` and `docs/superpowers/plans/`
- Modify: `tests/test_experiment_configuration.py`
- Modify: `tests/test_experiment_commands.py`

**Interfaces:**
- PowerShell defaults resolve via `$PSScriptRoot`.
- Bash defaults resolve via `${PROJECT_ROOT}/tools`.

- [ ] **Step 1: Write failing launcher behavior tests**

Extend launcher tests to invoke fake Python/preparation boundaries without
passing asset overrides and record that both local paths are forwarded. Add a
source-tree assertion that no tracked text file contains the predecessor name.

- [ ] **Step 2: Run launcher tests and verify RED**

```powershell
$env:PYTHONUTF8='1'
& 'F:\ProgramData\anaconda3\envs\aagmspenv\python.exe' -m pytest `
  tests/test_experiment_configuration.py tests/test_experiment_commands.py -q
```

Expected: launchers forward old absolute defaults and the reference scan finds
legacy paths.

- [ ] **Step 3: Replace active launcher defaults and historical references**

Use `$PSScriptRoot` for PowerShell, `${PROJECT_ROOT}/tools` for Bash, and revise
historical prose that says the extractor must remain external. Preserve CLI
override parameters.

- [ ] **Step 4: Verify launchers and reference removal**

Run Task 3 tests, parse both PowerShell files with
`System.Management.Automation.Language.Parser`, and run:

```powershell
$LegacyName = 'BrepMFR' + '-g'
rg -n $LegacyName . -g '!.git/**' -g '!results/**' -g '!tmp/**'
```

Expected: no matches.

- [ ] **Step 5: Commit launcher and documentation migration**

```powershell
git add tools tests docs
git commit -m "docs: remove external AAG asset references"
```

---

### Task 4: Verification and one-sample conversion

**Files:**
- Verify: all files changed in Tasks 1-3

**Interfaces:**
- Consumes: repository-local AAG assets through default paths.
- Produces: one valid temporary DGL BIN without explicit extractor/schema flags.

- [ ] **Step 1: Run focused and repository tests**

Run all focused suites from Tasks 1-3, then the repository suite with
`PYTHONUTF8=1`. Record unrelated missing optional packages separately.

- [ ] **Step 2: Validate asset imports and scripts**

Compile `tools/AAGExtractor.py`, call `load_schema` and `load_extractor`, parse
PowerShell scripts, and run `git diff --check`.

- [ ] **Step 3: Run one-sample conversion smoke test**

Create temporary train/val/test manifests selecting one real ID under
`D:\Code\Datasets\MFRInstSeg`, invoke `tools/mfrinstseg_to_bin.py`
without asset flags, and validate the resulting BIN with the converter's
existing validation function. Delete only the temporary smoke-test directory
after resolving and checking its absolute path.

- [ ] **Step 4: Inspect final state**

Require no predecessor-checkout references, no unintended working-tree changes, and a
clean `git diff --check`. Report the exact smoke-test sample and conversion
result.

- [ ] **Step 5: Commit verification adjustments if any**

```powershell
git add tools tests docs segmentation.py
git commit -m "test: verify repository-local AAG assets"
```
