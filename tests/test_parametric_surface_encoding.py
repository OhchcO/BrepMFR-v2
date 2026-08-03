import numpy as np
import pytest
import torch

from models.modules.pointcloud_encoder import (
    GeometryFeatureFusion,
    ParametricSurfaceFaceEncoder,
)
from tools.geometry_features import (
    SURFACE_NEIGHBOR_COUNT,
    build_trim_aware_neighbors,
    dimensionless_surface_metrics,
    normalize_surface_metrics,
)
from tools import geometry_features


def test_normalize_surface_metrics_accounts_for_uv_reparameterization():
    raw = np.asarray([[4.0, 1.0, 9.0]], dtype=np.float64)
    normalized = normalize_surface_metrics(raw, u_extent=4.0, v_extent=2.0)
    # u = u_mid + u_normalized * du/2, likewise for v.
    assert np.allclose(normalized, [[16.0, 2.0, 9.0]])


def test_dimensionless_surface_metrics_use_part_length_squared():
    metrics = np.asarray([[[16.0, 2.0, 9.0]]], dtype=np.float32)
    mask = np.asarray([[True]])
    scaled = dimensionless_surface_metrics(metrics, mask, characteristic_length=2.0)
    assert np.allclose(scaled, [[[4.0, 0.5, 2.25]]])


def test_trim_aware_neighbors_reject_segment_crossing_a_hole():
    uv = np.asarray(
        [[-0.8, 0.0], [0.8, 0.0], [-0.8, 0.8], [0.8, 0.8]],
        dtype=np.float32,
    )
    metrics = np.tile(np.asarray([[1.0, 0.0, 1.0]], dtype=np.float32), (4, 1))
    visibility = np.ones((11, 11), dtype=np.bool_)
    visibility[4:7, 4:7] = False

    neighbors = build_trim_aware_neighbors(
        uv,
        metrics,
        visibility,
        neighbor_count=2,
        segment_samples=9,
    )

    assert neighbors.shape == (4, 2)
    assert 1 not in neighbors[0]
    assert 0 not in neighbors[1]
    assert neighbors.dtype == np.int16


def test_occ_face_produces_valid_surface_metrics_and_neighbors():
    solid_module = pytest.importorskip("occwl.solid")
    face = next(iter(solid_module.Solid.make_box(2.0, 3.0, 4.0).faces()))
    candidates = geometry_features._valid_uv_candidates(face, 20)
    selected = candidates[:16]
    metrics = geometry_features._surface_metrics_for_normalized_uv(
        face, selected[:, 6:8]
    )
    neighbors = build_trim_aware_neighbors(
        selected[:, 6:8],
        metrics,
        geometry_features._visibility_grid(face, 20),
    )
    determinant = metrics[:, 0] * metrics[:, 2] - metrics[:, 1] ** 2
    assert np.all(metrics[:, 0] > 0)
    assert np.all(metrics[:, 2] > 0)
    assert np.all(determinant >= -1e-5)
    assert neighbors.shape == (16, SURFACE_NEIGHBOR_COUNT)


def _surface_inputs(face_count=2, point_capacity=12, valid_count=8):
    torch.manual_seed(10)
    points = torch.randn(face_count, point_capacity, 10)
    points[..., 3:6] = torch.nn.functional.normalize(points[..., 3:6], dim=-1)
    points[..., 6:8] = torch.tanh(points[..., 6:8])
    mask = torch.zeros(face_count, point_capacity, dtype=torch.bool)
    mask[:, :valid_count] = True
    points[:, valid_count:] = 0
    metrics = torch.zeros(face_count, point_capacity, 3)
    metrics[:, :valid_count, 0] = 1.0
    metrics[:, :valid_count, 2] = 1.0
    neighbors = torch.full(
        (face_count, point_capacity, SURFACE_NEIGHBOR_COUNT),
        -1,
        dtype=torch.int16,
    )
    for point_index in range(valid_count):
        choices = [
            other
            for other in range(valid_count)
            if other != point_index
        ][:SURFACE_NEIGHBOR_COUNT]
        neighbors[:, point_index, : len(choices)] = torch.tensor(
            choices, dtype=torch.int16
        )
    return points, mask, metrics, neighbors


def test_parametric_surface_encoder_ignores_padding_and_has_gradients():
    encoder = ParametricSurfaceFaceEncoder(
        hidden_dim=32,
        neighbor_metric="intrinsic",
        trim_aware=True,
        neighbor_count=SURFACE_NEIGHBOR_COUNT,
        coarse_count=4,
    )
    points, mask, metrics, neighbors = _surface_inputs()
    points.requires_grad_()
    metrics.requires_grad_()

    expected = encoder(points, mask, metrics, neighbors)
    changed_points = points.detach().clone()
    changed_metrics = metrics.detach().clone()
    changed_points[:, 8:] = 1000
    changed_metrics[:, 8:] = 1000
    actual = encoder(changed_points, mask, changed_metrics, neighbors)

    assert expected.shape == (2, 32)
    assert torch.allclose(expected.detach(), actual, atol=1e-6, rtol=1e-6)
    expected.square().mean().backward()
    assert points.grad is not None
    assert metrics.grad is not None
    assert torch.isfinite(points.grad).all()
    assert torch.isfinite(metrics.grad).all()


def test_parametric_surface_encoder_consumes_trim_neighbors():
    encoder = ParametricSurfaceFaceEncoder(
        hidden_dim=16,
        neighbor_metric="intrinsic",
        trim_aware=True,
        neighbor_count=SURFACE_NEIGHBOR_COUNT,
        coarse_count=4,
    ).eval()
    points, mask, metrics, neighbors = _surface_inputs(face_count=1)
    alternative = neighbors.clone()
    alternative[:, :8, 1:] = -1

    first = encoder(points, mask, metrics, neighbors)
    second = encoder(points, mask, metrics, alternative)
    assert not torch.allclose(first, second)


def test_gated_fusion_is_exact_base_at_initialization_then_can_open():
    fusion = GeometryFeatureFusion(hidden_dim=8, mode="gated", branch_count=2)
    base = torch.randn(4, 8)
    surface = torch.randn(4, 8)
    cylindrical = torch.randn(4, 8)

    initial = fusion(base, (surface, cylindrical), (1.0, 0.5))
    assert torch.equal(initial, base)

    with torch.no_grad():
        fusion.gates[0][-1].bias.fill_(0.5)
    opened = fusion(base, (surface, cylindrical), (1.0, 0.5))
    assert not torch.allclose(opened, base)


@pytest.mark.parametrize("mode", ["add", "gated"])
def test_fusion_rejects_branch_count_mismatch(mode):
    fusion = GeometryFeatureFusion(hidden_dim=8, mode=mode, branch_count=2)
    with pytest.raises(ValueError, match="branch"):
        fusion(torch.zeros(1, 8), (torch.zeros(1, 8),), (1.0,))
