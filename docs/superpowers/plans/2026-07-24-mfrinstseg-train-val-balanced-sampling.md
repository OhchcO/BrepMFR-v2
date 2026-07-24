# MFRInstSeg Train/Validation Balanced Sampling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the existing sampler so one configurable ratio deterministically produces independently balanced training and validation manifests.

**Architecture:** Keep all production behavior in `tools/sample_mfrinstseg_1pct.py`. Generalize the existing one-split workflow into a reusable split-selection function, call it for `train.txt` and `val.txt`, then write loader-compatible manifests and per-split reports. Use a temporary synthetic dataset for the red/green CLI test so no permanent second Python deliverable is added.

**Tech Stack:** Python 3.8, NumPy, JSON/CSV, argparse.

## Global Constraints

- `--ratio` controls both training and validation and defaults to `0.01`.
- `--seed` defaults to the fixed value `2026`.
- Classes `0-23` participate in balancing; class `24` remains in selected models but is excluded from the objective.
- Official source split files and labels are never modified.
- Test split is not sampled.

---

### Task 1: Add deterministic train/validation sampling

**Files:**
- Modify: `tools/sample_mfrinstseg_1pct.py`
- Test temporarily: `tmp/test_sample_mfrinstseg_train_val.py`

**Interfaces:**
- Consumes: official `<dataset-root>/train.txt`, `<dataset-root>/val.txt`, and `<dataset-root>/labels/*.json`.
- Produces: `<output-dir>/train.txt`, `<output-dir>/val.txt`, `train_class_statistics.csv`, `val_class_statistics.csv`, and `selection_report.json`.

- [ ] **Step 1: Write the failing end-to-end test**

Create a temporary dataset with 48 one-face samples in each split. Every class from 0 through 23 appears twice per split. Invoke:

```python
result = subprocess.run(
    [
        sys.executable,
        str(script),
        "--dataset-root", str(dataset_root),
        "--output-dir", str(output_root),
        "--ratio", "0.5",
        "--seed", "2026",
        "--swap-attempts", "100",
    ],
    text=True,
    capture_output=True,
)
assert result.returncode == 0, result.stderr
assert len((output_root / "train.txt").read_text().splitlines()) == 24
assert len((output_root / "val.txt").read_text().splitlines()) == 24
assert (output_root / "train_class_statistics.csv").is_file()
assert (output_root / "val_class_statistics.csv").is_file()
```

- [ ] **Step 2: Run the test and verify RED**

Run:

```powershell
G:\ProgramData\anaconda3\envs\aagmspenv\python.exe tmp\test_sample_mfrinstseg_train_val.py
```

Expected: FAIL because the current script does not create `val.txt`.

- [ ] **Step 3: Implement the minimal split-generalization**

Add a split result container and a `sample_split(...)` helper that reads one split, loads its labels, computes `floor(unique_count * ratio)`, performs balanced and seeded-random selection, and returns all data needed for reporting.

Change output writing so:

```python
for split_name, result in split_results.items():
    write_manifest(output_dir / (split_name + ".txt"), result.selected_records)
    write_class_statistics(
        output_dir / (split_name + "_class_statistics.csv"), result
    )
```

Build `selection_report.json` with shared configuration and nested `train` and `val` result objects. Keep the existing `--ratio` and `--seed` arguments, applying their values to both calls:

```python
split_results = {
    name: sample_split(
        dataset_root=dataset_root,
        split_file=dataset_root / (name + ".txt"),
        class_ids=class_ids,
        ratio=args.ratio,
        seed=args.seed,
        swap_attempts=args.swap_attempts,
    )
    for name in ("train", "val")
}
```

- [ ] **Step 4: Run the temporary test and verify GREEN**

Run the command from Step 2.

Expected: PASS; both manifests contain 24 unique IDs, and a second run produces identical file hashes.

- [ ] **Step 5: Run against the complete MFRInstSeg splits**

Run:

```powershell
G:\ProgramData\anaconda3\envs\aagmspenv\python.exe tools\sample_mfrinstseg_1pct.py `
  --dataset-root D:\Code\Datasets\MFRInstSeg `
  --output-dir D:\Code\Datasets\MFRInstSeg\subsets\train_val_1pct_balanced `
  --ratio 0.01 `
  --seed 2026
```

Expected:

- `train.txt` contains `floor(unique_train_models * 0.01)` unique official training IDs.
- `val.txt` contains `floor(unique_val_models * 0.01)` unique official validation IDs.
- Both reports show lower coefficient of variation and spread than the seeded random baselines.
- Repeating the command produces identical SHA256 hashes for both manifests.

- [ ] **Step 6: Review and commit only intended files**

Run:

```powershell
git diff --check
git status --short
```

Do not include the user's existing `.gitignore` modification. Commit only the sampler and this plan:

```powershell
git add -- tools/sample_mfrinstseg_1pct.py docs/superpowers/plans/2026-07-24-mfrinstseg-train-val-balanced-sampling.md
git commit -m "feat: balance train and validation subsets"
```
