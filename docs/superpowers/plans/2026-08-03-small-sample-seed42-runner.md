# Seed42 Small-Sample Experiment Runner Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add percentage-correct result naming and one resumable seed42 PowerShell entry point that runs the fixed baseline, parameter-surface, affinity-only, and instance matrix for 1%, 5%, or 10% data.

**Architecture:** The existing generic launcher remains the source of training argument construction but derives its result prefix from `SamplePercent`. A new orchestration script validates the three supported percentages, chooses an isolated prepared cache, delegates each experiment to the generic training launcher, selects a fresh checkpoint, and delegates official testing.

**Tech Stack:** PowerShell 5.1, Python 3, pytest, existing BrepMFR training and test launchers.

## Global Constraints

- Work only on `tmp/param-surface-instance-combination`; do not merge or delete the branch.
- Accept exactly `0.01`, `0.05`, and `0.10` in the new runner.
- Preserve the established 1% cache `D:\Code\Datasets\MFRInstSeg_BrepMFR_ParamSurface_v4_seed42`.
- Use isolated `_5pct_seed42` and `_10pct_seed42` caches for new ratios.
- Run exactly `baseline`, best parameter-surface `pc_aag`, `affinity_only`, and `instance` in that order.
- Do not run dataset conversion or formal training during implementation.
- Write each behavioral test first and observe the expected failure before changing production scripts.

---

### Task 1: Percentage-Correct Generic Result Identity

**Files:**
- Modify: `tests/test_experiment_configuration.py`
- Modify: `tools/run_mfrinstseg_experiments.ps1`

**Interfaces:**
- Consumes: existing `-SamplePercent [double]` launcher parameter.
- Produces: result names whose first component is `1pct`, `5pct`, or `10pct` for the supported inputs.

- [ ] **Step 1: Add a parameterized failing launcher test**

Parameterize the existing real-launcher test
`test_launcher_forwards_positive_mapping_and_uses_positive_result_name`. Keep
its complete temporary dataset and fake-Python setup, add `-SamplePercent` to
its existing command string, and inspect the actual `--experiment_name`
argument:

```python
@pytest.mark.parametrize(
    ("sample_percent", "expected_prefix"),
    [(0.01, "1pct"), (0.05, "5pct"), (0.10, "10pct")],
)
def test_launcher_derives_result_identity_from_sample_percent(
    tmp_path, sample_percent, expected_prefix
):
    dataset = tmp_path / "prepared"
    dataset.mkdir()
    for split in ("train", "val"):
        (dataset / f"{split}.txt").write_text("sample-id\n", encoding="utf-8")
    captured = tmp_path / "captured-args.txt"
    fake_python = tmp_path / "fake-python.ps1"
    fake_python.write_text(
        r'''param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$CapturedArguments
)
[System.IO.File]::WriteAllLines(
    $env:MFRINSTSEG_TRAIN_ARGUMENTS,
    $CapturedArguments
)
exit 0
'''.strip() + "\n",
        encoding="utf-8",
    )
    launcher = (
        Path(__file__).resolve().parents[1]
        / "tools"
        / "run_mfrinstseg_experiments.ps1"
    )
    quote = lambda value: str(value).replace("'", "''")
    command = (
        "& '{launcher}' -Mode baseline -AutoPrepare $false "
        "-DatasetPath '{dataset}' -PythonExe '{python}' "
        "-SamplePercent {sample_percent}"
    ).format(
        launcher=quote(launcher),
        dataset=quote(dataset),
        python=quote(fake_python),
        sample_percent=sample_percent,
    )
    environment = os.environ.copy()
    environment["MFRINSTSEG_TRAIN_ARGUMENTS"] = str(captured)
    completed = subprocess.run(
        [
            "powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass",
            "-Command", command,
        ],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    arguments = captured.read_text(encoding="utf-8").splitlines()
    name_index = arguments.index("--experiment_name")
    assert arguments[name_index + 1] == (
        f"MFRInstSeg_{expected_prefix}_baseline_seed2026"
    )
```

Keep any helper constants/functions local to the test module and make them
exercise the real launcher; do not assert on source text.

- [ ] **Step 2: Run the new test and verify the non-1% cases fail**

Run:

```powershell
& 'G:\ProgramData\anaconda3\envs\aagmspenv\python.exe' -m pytest `
  tests/test_experiment_configuration.py::test_launcher_derives_result_identity_from_sample_percent -q
```

Expected: the 5% and 10% cases fail because both are still named `1pct`; the
1% case passes as a characterization of existing behavior.

- [ ] **Step 3: Implement invariant percentage formatting in the launcher**

Add a small function before result-name construction:

```powershell
function Get-SamplePercentLabel {
    param([double]$Value)
    if (
        [double]::IsNaN($Value) -or
        [double]::IsInfinity($Value) -or
        $Value -le 0.0 -or
        $Value -gt 1.0
    ) {
        throw "SamplePercent must be finite and in the interval (0, 1]."
    }
    $PercentValue = 100.0 * $Value
    $PercentText = $PercentValue.ToString(
        "0.############",
        [System.Globalization.CultureInfo]::InvariantCulture
    ).Replace(".", "p")
    return "${PercentText}pct"
}
```

Compute `$SampleLabel = Get-SamplePercentLabel -Value $SamplePercent` once and
replace the hard-coded result identity with:

```powershell
$ExperimentName = "MFRInstSeg_${SampleLabel}_${ExperimentLabel}_seed${Seed}"
```

- [ ] **Step 4: Re-run the percentage identity test**

Run the Step 2 command. Expected: `3 passed`.

- [ ] **Step 5: Run the existing launcher configuration tests**

Run:

```powershell
& 'G:\ProgramData\anaconda3\envs\aagmspenv\python.exe' -m pytest `
  tests/test_experiment_configuration.py -q
```

Expected: all tests pass and the established 1% oracle identity remains
`MFRInstSeg_1pct_oracle_positive_seed2026`.

---

### Task 2: Single-Percentage Four-Experiment Runner

**Files:**
- Create: `tests/test_small_sample_seed42_runner.py`
- Create: `tools/run_mfrinstseg_small_sample_seed42.ps1`

**Interfaces:**
- Consumes: `-SamplePercent`, raw-data path, cache base path, results root, Python executable, resource settings, optional rebuild, and injectable launcher paths.
- Produces: one percentage-specific prepared cache and four tested experiment result roots.

- [ ] **Step 1: Write a failing end-to-end fake-launcher test**

Create fake training and test PowerShell launchers. The fake trainer records
mode, mapping, sample percentage, geometry options, cache path, seed, rebuild
flag, and creates a timestamped `best-fake.ckpt` beneath the literal expected
result directory supplied through test environment variables. The fake tester
records its checkpoint and prepared dataset.

The main test invokes the missing runner with `-SamplePercent 0.05` and asserts
these hand-derived outcomes:

```python
assert [row["mode"] for row in trains] == [
    "baseline", "pc_aag", "affinity_only", "instance"
]
assert [row["mapping"] for row in trains] == [
    "signed", "signed", "positive_only", "positive_only"
]
assert [row["encoder"] for row in trains] == [
    "mini_point", "param_surface", "mini_point", "mini_point"
]
assert trains[1]["metric"] == "intrinsic"
assert trains[1]["trim"] == "False"
assert trains[1]["fusion"] == "gated"
assert all(row["sample"] == "0.05" for row in trains)
assert all(row["seed"] == "42" for row in trains)
assert all(row["prepared"].endswith(
    "MFRInstSeg_BrepMFR_ParamSurface_v4_5pct_seed42"
) for row in trains)
assert len(tests) == 4
assert {row["checkpoint"] for row in tests} == {
    row["checkpoint"] for row in trains
}
```

Pre-create an older checkpoint in every result root and assert that no test
event uses it. This mutation catches removal of fresh-checkpoint filtering.

In the same test file, write the unsupported-percentage and completed-result
tests before the runner exists:

```python
@pytest.mark.parametrize("sample_percent", [0, 0.02, 1.01])
def test_runner_rejects_unsupported_percentage_before_launch(
    tmp_path, sample_percent
):
    completed, events = invoke_runner(tmp_path, sample_percent)
    assert completed.returncode != 0
    assert "SamplePercent must be one of 0.01, 0.05, or 0.10" in (
        completed.stdout + completed.stderr
    )
    assert not events.exists()


def test_runner_skips_four_completed_results(tmp_path):
    create_completed_result_roots(tmp_path, label="10pct")
    completed, events = invoke_runner(tmp_path, 0.10)
    assert completed.returncode == 0
    assert not events.exists()
```

`invoke_runner` must execute the real runner with the fake launcher paths and
return `(CompletedProcess, events_path)`. `create_completed_result_roots` must
create these four literal directories, each with `fake-run/test_log.txt` and
`fake-run/best-fake.ckpt`:

```python
COMPLETED_10PCT_ROOTS = [
    "MFRInstSeg_10pct_baseline_seed42",
    "MFRInstSeg_10pct_pc_aag_param_surface_intrinsic_untrimmed_gated_seed42",
    "MFRInstSeg_10pct_affinity_only_positive_seed42",
    "MFRInstSeg_10pct_instance_positive_seed42",
]
```

- [ ] **Step 2: Run the new integration test and observe the missing-runner failure**

Run the complete new test file:

```powershell
& 'G:\ProgramData\anaconda3\envs\aagmspenv\python.exe' -m pytest `
  tests/test_small_sample_seed42_runner.py -q
```

Expected: FAIL because `tools/run_mfrinstseg_small_sample_seed42.ps1` does not
exist.

- [ ] **Step 3: Implement runner validation, identity, matrix, and orchestration**

Create the runner with a mandatory double `SamplePercent`, runtime membership
validation using tolerance `1e-12`, and this literal label/cache mapping:

```powershell
$SupportedSamples = @(
    [pscustomobject]@{ Value = 0.01; Label = "1pct"; CacheSuffix = "seed42" },
    [pscustomobject]@{ Value = 0.05; Label = "5pct"; CacheSuffix = "5pct_seed42" },
    [pscustomobject]@{ Value = 0.10; Label = "10pct"; CacheSuffix = "10pct_seed42" }
)
```

Resolve the selected entry before validating or invoking launchers, then derive:

```powershell
$PreparedDatasetPath = "${PreparedDatasetBasePath}_$($Selected.CacheSuffix)"
```

Define four experiment records with exact `Mode`, `Mapping`, `ResultLabel`,
`FaceGeometryEncoder`, `SurfaceNeighborMetric`, `SurfaceTrimAware`, and
`SurfaceFusion` values from the approved design.

For each experiment:

- derive `MFRInstSeg_<label>_<result-label>_seed42`;
- skip when the latest normal `test_log.txt` exists;
- record every existing `best-*.ckpt` path before training;
- invoke the training launcher using PowerShell splatting;
- forward rebuild only for index zero;
- select only a checkpoint path absent from the pre-training set;
- call the test launcher with that checkpoint and shared cache;
- throw immediately on launcher failure or a missing fresh checkpoint.

Keep log discovery outside any `test_bias_scale_0` directory. The runner does
not perform bias-scale override tests.

- [ ] **Step 4: Run all runner tests and verify they pass**

Run:

```powershell
& 'G:\ProgramData\anaconda3\envs\aagmspenv\python.exe' -m pytest `
  tests/test_small_sample_seed42_runner.py -q
```

Expected: all tests pass.

---

### Task 3: Verification and Commit

**Files:**
- Verify: `tools/run_mfrinstseg_experiments.ps1`
- Verify: `tools/run_mfrinstseg_small_sample_seed42.ps1`
- Verify: all modified tests and documentation

**Interfaces:**
- Consumes: completed Tasks 1 and 2.
- Produces: a clean, committed temporary branch ready for user-run training.

- [ ] **Step 1: Parse both PowerShell files with the AST parser**

```powershell
$files = @(
  'tools\run_mfrinstseg_experiments.ps1',
  'tools\run_mfrinstseg_small_sample_seed42.ps1'
)
foreach ($file in $files) {
  $tokens = $null
  $errors = $null
  [System.Management.Automation.Language.Parser]::ParseFile(
    (Resolve-Path $file), [ref]$tokens, [ref]$errors
  ) | Out-Null
  if ($errors.Count -ne 0) { throw ($errors | Out-String) }
}
```

Expected: no parser errors.

- [ ] **Step 2: Run focused tests**

```powershell
& 'G:\ProgramData\anaconda3\envs\aagmspenv\python.exe' -m pytest -q `
  tests/test_experiment_configuration.py `
  tests/test_small_sample_seed42_runner.py
```

Expected: zero failures.

- [ ] **Step 3: Run the repository suite without the known optional dependency blocker**

```powershell
& 'G:\ProgramData\anaconda3\envs\aagmspenv\python.exe' -m pytest -q `
  --ignore=tests/test_model_result_paths.py
```

Expected: zero failures. Record that the full unfiltered collection remains
environment-blocked only if `prettytable` is still unavailable.

- [ ] **Step 4: Check diffs and branch state**

```powershell
git diff --check
git status --short --branch
git diff --stat
```

Expected: no whitespace errors; only planned files are modified or new.

- [ ] **Step 5: Commit implementation**

```powershell
git add tools/run_mfrinstseg_experiments.ps1 `
  tools/run_mfrinstseg_small_sample_seed42.ps1 `
  tests/test_experiment_configuration.py `
  tests/test_small_sample_seed42_runner.py `
  docs/superpowers/plans/2026-08-03-small-sample-seed42-runner.md
git commit -m "feat: add percentage-aware small-sample runner"
```

- [ ] **Step 6: Verify the final commit and clean worktree**

```powershell
git status --short --branch
git log -4 --oneline --decorate
```

Expected: branch `tmp/param-surface-instance-combination` is clean and contains
the implementation commit; `main` remains unchanged.
