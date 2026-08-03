# Parameter-Surface Encoder Design

## Scope

Add three independently configurable representation changes without changing
the existing Graphormer attention or instance-bias implementation:

1. a lightweight hierarchical encoder for sampled parametric surfaces;
2. trim-aware intrinsic neighbourhoods in each B-Rep face;
3. residual gated fusion of local surface and cylindrical features with AAG
   node features.

The existing 10D MiniPoint encoder remains the default so old experiments and
checkpoints keep their meaning. Formal training is outside this change.

## BIN contract

Keep the existing `face_points [F,96,10]`, mask, count, and cylindrical fields.
Add two node tensors:

- `face_surface_metric [F,96,3]`, float32: normalized first fundamental form
  coefficients `(E,F,G)`, divided by the squared full-part characteristic
  length.
- `face_trim_neighbors [F,96,16]`, int16: deterministic neighbour indices;
  `-1` denotes padding or a rejected trim-crossing edge.

The UV sampling stage retains the complete inside/outside grid. Candidate
points are selected by the existing deterministic 3D FPS. Intrinsic pair
distance uses the average first fundamental form of both endpoints. A segment
is accepted only if sampled positions along its normalized-UV line remain
inside the retained trim visibility grid.

The converter/cache version is bumped. Legacy BIN files remain usable by
MiniPoint modes, while the new encoder explicitly requires the new fields.

## Encoder

`ParametricSurfaceFaceEncoder` consumes the 10D points plus `(E,F,G)`.
XYZ is centered and scaled per face; normals, normalized UV, dimensionless
curvatures, and metrics are retained. A point MLP is followed by one local edge
aggregation stage. Neighbour selection is configurable as `xyz`, `uv`, or
`intrinsic`; intrinsic trim-aware mode consumes the precomputed BIN graph.
The FPS order is reused for a fixed coarse subset before masked global pooling.

This is intentionally a lightweight PointNet++-style hierarchy rather than a
full point segmentation network.

## Fusion

`add` preserves the old direct residual addition. `gated` predicts a
per-channel gate from `[base AAG, geometry branch]` for the surface and
cylindrical branches separately. The final gate projection is zero-initialized,
so the initial output is exactly the base AAG representation. Existing branch
scale parameters remain active.

## Configuration

- `--face_geometry_encoder {mini_point,param_surface}`
- `--surface_neighbor_metric {xyz,uv,intrinsic}`
- `--surface_trim_aware` / `--no_surface_trim_aware`
- `--surface_fusion {add,gated}`

Defaults preserve existing behaviour: `mini_point`, `intrinsic`, trim-aware
off, and additive fusion. The new surface encoder requires face pointcloud
input; trim-aware mode is valid only with `param_surface` and `intrinsic`.

## Verification

Unit tests cover metric normalization, trim crossing, deterministic neighbours,
padding invariance, encoder gradients, exact zero-gate equivalence, data
contract validation/collation, and CLI-to-model threading. Only focused tests
and a synthetic/single-sample forward smoke test are run.
