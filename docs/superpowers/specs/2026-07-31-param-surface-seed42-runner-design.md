# Parameter-Surface Seed42 Runner Design

## Goal

Provide one PowerShell entry point that trains and tests the three seed42
parameter-surface screening experiments in their approved order.

## Behaviour

- Use seed 42, 1% sampling, batch size 16, 100 epochs, and zero workers.
- Use a dedicated v4 cache:
  `D:\Code\Datasets\MFRInstSeg_BrepMFR_ParamSurface_v4_seed42`.
- Run `pc_aag` with:
  1. intrinsic neighbours, untrimmed, additive fusion;
  2. intrinsic neighbours, trim-aware, additive fusion;
  3. intrinsic neighbours, trim-aware, gated fusion.
- Stop immediately if training, checkpoint discovery, or testing fails.
- Test only a checkpoint created by the current training invocation.
- Reuse the prepared cache by default. An optional
  `-RebuildPreparedData` switch applies only to the first experiment.
- Do not run training while developing or validating this script.

## Interface

The script accepts path, Python executable, batch-size, epoch, and worker
overrides while retaining safe project defaults. It delegates training and
testing to the existing launchers rather than duplicating their logic.
