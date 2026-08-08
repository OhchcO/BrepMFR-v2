# Parameter-Surface Instance Combination Design

## Goal

Add an unambiguous experiment path that combines the established best face
geometry configuration (`param_surface + intrinsic + untrimmed + gated`) with
the four instance-relation conditions used by the existing ablation study.

## Branch and Scope

- All work is performed on `tmp/param-surface-instance-combination`.
- Existing aliases keep their current behavior so historical results remain
  reproducible.
- Model internals, losses, attention equations, BIN schema, and prepared data
  generation are unchanged.
- Formal training is outside this change. Only unit/static tests may run.

## Considered Approaches

1. Reuse `instance` while passing `FaceGeometryEncoder=param_surface`.
   Rejected because `instance` disables the point-cloud branch, so the encoder
   is not consumed even though the result name can claim otherwise.
2. Reuse `spe_instance` with cylindrical scale zero.
   Rejected because it still enables an unnecessary cylindrical branch and
   gives the experiment a misleading identity.
3. Add dedicated point-cloud instance aliases. Selected because it preserves
   every old experiment and makes the consumed inputs explicit.

## Experiment Aliases

The following aliases use `use_face_uv=False`,
`use_face_pointcloud=True`, and `use_cylindrical_pos=False`:

| Alias | Internal mode | Affinity loss | Bias scale | Bias strategy |
|---|---|---:|---:|---|
| `pc_dynamic_no_loss` | `dynamic` | no | 1.0 | `late_gated` |
| `pc_affinity_only` | `dynamic_affinity` | yes | 0.0 | `late_gated` |
| `pc_instance` | `dynamic_affinity` | yes | 1.0 | `late_gated` |
| `pc_oracle` | `oracle` | GT relation | 1.0 | `late_gated` |

## Seed42 Runner

Create `tools/run_mfrinstseg_param_surface_instance_seed42.ps1`. It runs this
matrix in order:

1. `pc_aag` (A, representation-only control)
2. `pc_affinity_only` (B)
3. `pc_dynamic_no_loss` (C)
4. `pc_instance` (D)
5. `pc_oracle` (E)

Every entry passes the same geometry configuration. B-E pass
`InstanceRelationMapping=positive_only`; A keeps the default `signed` identity
because it does not consume instance relations and must reuse the established
representation-only control result.

- `FaceGeometryEncoder=param_surface`
- `SurfaceNeighborMetric=intrinsic`
- `SurfaceTrimAware=false`
- `SurfaceFusion=gated`
- seed 42 and the versioned v4 prepared dataset by default

The runner locates the fresh checkpoint created by each training call, invokes
the official test wrapper, and invokes a second test with
`InstanceBiasScaleOverride=0` only for D (`pc_instance`). A completed result is
skipped only when all tests required for that entry already exist. Rebuilding
prepared data is opt-in and applies only to the first experiment.

## Result Identity

The generic launcher continues to derive result names from the alias and face
geometry settings. Expected roots include, for example:

`MFRInstSeg_1pct_pc_instance_positive_param_surface_intrinsic_untrimmed_gated_seed42`

This prevents UV-only runs from being confused with parameter-surface runs.

## Verification

- Unit-test all four alias mappings.
- Verify the generic PowerShell launcher's accepted mode list.
- Execute the new runner against fake training/test launchers and assert the
  A-E order, shared geometry flags, positive-only mapping, fresh checkpoint
  forwarding, and the single bias-off test.
- Parse all modified PowerShell files with the PowerShell AST parser.
- Run the relevant unit-test suite and `git diff --check`.
