# Latest Bias-Off Evaluation Commands

## Goal

Append three copyable PowerShell evaluation blocks to `tools/实验.txt` so the
latest deterministic checkpoints can be tested with attention bias disabled.
These runs isolate the inference-time contribution of the residual relation
bias without retraining.

## Scope

The appended blocks evaluate exactly these checkpoints:

1. `dynamic_no_loss + positive_only`
   - `results/MFRInstSeg_1pct_dynamic_no_loss_positive_seed2026/0728/174518/best-080-0.165093.ckpt`
2. `instance + positive_only`
   - `results/MFRInstSeg_1pct_instance_positive_seed2026/0728/164959/best-095-0.142707.ckpt`
3. `oracle + positive_only`
   - `results/MFRInstSeg_1pct_oracle_positive_seed2026/0728/164959/best-086-0.152804.ckpt`

Each block calls `tools/run_mfrinstseg_test.ps1` with:

- the explicit checkpoint path above;
- the shared prepared dataset
  `D:\Code\Datasets\MFRInstSeg_BrepMFR_SPEPCAAG`;
- `BatchSize=16`;
- `NumWorkers=0`;
- `InstanceBiasScaleOverride=0`.

## Non-Goals

- Do not start training.
- Do not regenerate or rebuild BIN data.
- Do not discover checkpoints dynamically, because that could select an older
  or newer incompatible run.
- Do not change existing experiment blocks or application code.
- Do not add multi-seed training commands in this change.

## Output and Failure Behavior

The existing test launcher validates the checkpoint and dataset paths, fails
with a nonzero exit code on errors, and writes each result under the
checkpoint's `test_bias_scale_0` directory. Existing bias-on results remain
unchanged.

## Verification

After editing:

1. Confirm all three explicit checkpoints exist.
2. Confirm each new block contains `-InstanceBiasScaleOverride 0`.
3. Run the experiment-command unit tests to ensure the existing 18 training
   identities remain unchanged and the appended test-only blocks are valid.
4. Do not execute model evaluation or training.
