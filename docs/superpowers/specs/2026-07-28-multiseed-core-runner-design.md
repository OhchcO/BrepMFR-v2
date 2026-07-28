# Multi-Seed Core Experiment Runner

## Goal

Provide three directly runnable PowerShell entry points for Seeds 42, 2025,
and 2027. Each entry point runs the same three thesis-core experiments in a
fixed order and evaluates the best checkpoint from each newly completed
training run.

## Files and Responsibilities

- `tools/run_mfrinstseg_core_seed.ps1`
  - Implements the shared sequential training and evaluation workflow.
  - Accepts a required integer `Seed`.
- `tools/run_mfrinstseg_core_seed42.ps1`
  - Calls the shared runner with Seed 42.
- `tools/run_mfrinstseg_core_seed2025.ps1`
  - Calls the shared runner with Seed 2025.
- `tools/run_mfrinstseg_core_seed2027.ps1`
  - Calls the shared runner with Seed 2027.

The existing training and test launchers remain the only components that
invoke Python:

- `tools/run_mfrinstseg_experiments.ps1`
- `tools/run_mfrinstseg_test.ps1`

## Experiment Sequence

Each seed entry point runs these experiments sequentially:

1. `baseline` with the default signed mapping;
2. `dynamic_no_loss` with `positive_only`;
3. `instance` with `positive_only`.

Every experiment uses:

- raw dataset: `D:\Code\Datasets\MFRInstSeg`;
- sample percentage: `0.01`;
- batch size: `16`;
- maximum epochs: `100`;
- data-loader workers: `0`;
- the fixed seed belonging to the entry point.

After each training command returns successfully, the runner finds the best
checkpoint created by that invocation and immediately evaluates it on the
prepared dataset's complete `test.txt`.

Bias-disabled inference is outside this workflow. These scripts produce only
the standard Bias ON results required for multi-seed comparison.

## Seed-Isolated Prepared Data

The preparation fingerprint contains the seed. Reusing the Seed 2026 prepared
directory for a different seed would produce a fingerprint mismatch.

The shared runner therefore derives a separate prepared directory:

- Seed 42: `D:\Code\Datasets\MFRInstSeg_BrepMFR_SPEPCAAG_seed42`;
- Seed 2025: `D:\Code\Datasets\MFRInstSeg_BrepMFR_SPEPCAAG_seed2025`;
- Seed 2027: `D:\Code\Datasets\MFRInstSeg_BrepMFR_SPEPCAAG_seed2027`.

The first experiment for a seed prepares that cache. The next two experiments
reuse the same exact cache. The runner never passes `-RebuildPreparedData`, so
it cannot overwrite Seed 2026 or another seed's prepared data.

## Checkpoint Selection

Result roots follow the current naming convention:

- `MFRInstSeg_1pct_baseline_seed<seed>`;
- `MFRInstSeg_1pct_dynamic_no_loss_positive_seed<seed>`;
- `MFRInstSeg_1pct_instance_positive_seed<seed>`.

Before starting each training command, the runner records the current time.
After training, it accepts only a `best-*.ckpt` whose last-write time belongs
to the current invocation. This prevents a failed or incomplete run from
silently evaluating a stale checkpoint from an earlier attempt.

## Failure Behavior

- `$ErrorActionPreference` is `Stop`.
- A nonzero training exit code stops the seed workflow.
- A missing fresh checkpoint stops the workflow with the experiment label in
  the error message.
- A nonzero test exit code stops the workflow.
- Later experiments do not run after a failure.
- Existing checkpoints, prepared datasets, and test logs are not deleted.

## Verification Without Training

Automated tests use temporary fake training and test launchers. They verify:

1. the experiment order is baseline, dynamic, instance;
2. Seed and shared per-seed prepared path are forwarded consistently;
3. positive mapping is supplied only to dynamic and instance;
4. each fresh checkpoint is passed to the test launcher;
5. stale checkpoints are rejected;
6. all four PowerShell files have valid syntax;
7. each fixed-seed entry point selects its declared seed.

The verification process must not execute dataset conversion, model training,
or model evaluation.
