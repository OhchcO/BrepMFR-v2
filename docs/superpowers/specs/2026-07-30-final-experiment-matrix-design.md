# BrepMFR Final Experiment Matrix Design

## Objective

Provide one reproducible experiment runner shared by Seed42, Seed2025,
Seed2026, and Seed2027. Each seed must execute the same final ten-experiment
matrix, reuse one seed-specific 10D BIN cache, train each experiment, select a
fresh best checkpoint, and evaluate the complete official `test.txt`.

The existing 10D point representation, instance-relation implementation, loss
definitions, sampling policy, dataset selection policy, and 100-epoch training
policy remain unchanged.

## Final Experiment Matrix

Every seed runs these experiments in this order:

1. `topo_aag`: topology/AAG only; UV off, point cloud off, cylindrical off.
2. `uv_aag`: original BRepMFR representation baseline; UV on, point cloud off,
   cylindrical off.
3. `pc_aag`: topology plus 10D point cloud; UV off, cylindrical off.
4. `cyl_aag`: topology plus cylindrical position; UV off, point cloud off.
5. `spe_pcaag`: topology plus 10D point cloud and cylindrical position; UV
   off.
6. `dynamic_no_loss` with `positive_only`: predicted late-gated instance bias
   without affinity supervision.
7. `affinity_only` with `positive_only`: affinity supervision with attention
   bias scale set to zero.
8. `instance` with `positive_only`: predicted late-gated instance bias plus
   affinity supervision.
9. `oracle` with `positive_only`: GT instance relation through the same
   late-gated path; reported only as an upper bound.
10. `spe_instance` with `positive_only`: final combined model using
    SPE-PCAAG, predicted late-gated instance bias, and affinity supervision.

`spe_instance` is a new explicit experiment alias with:

```text
experiment_mode = dynamic_affinity
use_face_uv = false
use_face_pointcloud = true
use_cylindrical_pos = true
instance_bias_scale = 1.0
instance_bias_strategy = late_gated
```

The legacy `full` alias remains available only for command compatibility and
is not part of the final matrix. Checkpoints trained with the former 6D
point-cloud encoder are intentionally not load-compatible with the finalized
10D encoder; reproducing those checkpoints requires the former code and cache.

## Runner Architecture

`tools/run_mfrinstseg_core_seed.ps1` is the single source of truth for the
matrix. It accepts `-Seed`, builds the seed-specific cache path, loops through
the ten experiment descriptors, launches training, requires a checkpoint
created during the current experiment, and then launches test evaluation with
the same seed and cache.

The four seed scripts are thin wrappers:

```text
run_mfrinstseg_core_seed42.ps1
run_mfrinstseg_core_seed2025.ps1
run_mfrinstseg_core_seed2026.ps1
run_mfrinstseg_core_seed2027.ps1
```

Each wrapper invokes the generic runner exactly once and does not append extra
experiments.

## Cache and Result Naming

The default prepared-cache base is:

```text
D:\Code\Datasets\MFRInstSeg_BrepMFR_SPEPCAAG10D
```

The runner appends `_seed<seed>`, producing:

```text
MFRInstSeg_BrepMFR_SPEPCAAG10D_seed42
MFRInstSeg_BrepMFR_SPEPCAAG10D_seed2025
MFRInstSeg_BrepMFR_SPEPCAAG10D_seed2026
MFRInstSeg_BrepMFR_SPEPCAAG10D_seed2027
```

All ten experiments for one seed reuse that seed's cache. No wrapper passes
`-RebuildPreparedData`.

Geometry result labels equal their aliases. Instance experiments use a
`_positive` result label, matching the launcher's `positive_only` naming.
The final model result label is `spe_instance_positive`.

## Failure Handling

The runner stops immediately when:

- a launcher path is missing;
- training exits unsuccessfully;
- no best checkpoint newer than the experiment start time is found;
- test evaluation exits unsuccessfully.

An earlier checkpoint from the same result directory must never be silently
used after a failed training run.

## Verification

Unit tests must verify:

- the exact `spe_instance` switch mapping;
- Python and PowerShell mode acceptance;
- the generic runner's exact ordered ten-experiment matrix;
- `positive_only` on the five relation-aware experiments;
- the 10D cache base and seed-specific suffix;
- all four wrappers delegate exactly once with the correct seed;
- wrappers contain no appended one-off experiment;
- PowerShell syntax is valid.

Only unit tests and launcher smoke tests are permitted during implementation.
No training or full-dataset conversion is run by Codex.
