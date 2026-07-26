import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from tools import geometry_features
from tools.geometry_features import (
    PointSamplingConfig,
    adaptive_point_count,
    cylindrical_positions,
    extract_geometry_features,
    farthest_point_indices,
)


def test_area_adaptive_counts_are_clipped_and_rounded_to_eight():
    config = PointSamplingConfig()

    assert adaptive_point_count(0.01, 1.0, config) == 32
    assert adaptive_point_count(1.0, 1.0, config) == 64
    assert adaptive_point_count(100.0, 1.0, config) == 96


def test_area_adaptive_count_uses_deterministic_half_up_rounding():
    config = PointSamplingConfig(area_exponent=1.0)

    assert adaptive_point_count(68.0, 64.0, config) == 72


def test_adaptive_count_always_uses_base_64_not_fixed_mode_count():
    default = PointSamplingConfig()
    changed_fixed_mode_count = PointSamplingConfig(fixed_count=40)

    assert adaptive_point_count(1.0, 1.0, default) == 64
    assert adaptive_point_count(1.0, 1.0, changed_fixed_mode_count) == 64


def test_fixed_point_count_ignores_area():
    config = PointSamplingConfig(mode="fixed", fixed_count=40)

    assert adaptive_point_count(0.01, 1.0, config) == 40
    assert adaptive_point_count(100.0, 1.0, config) == 40


@pytest.mark.parametrize(
    "kwargs",
    [
        {"mode": "random"},
        {"fixed_count": 0},
        {"min_count": 40, "max_count": 32},
        {"max_count": 104},
        {"area_exponent": 0.0},
        {"candidate_grids": ()},
        {"candidate_grids": (20, 16)},
        {"candidate_grids": (20, 32, 48, 64, 88)},
        {"candidate_grids": (20, 24, 40, 60, 80)},
        {"fixed_count": True},
        {"fixed_count": 64.5},
        {"min_count": False},
        {"min_count": 32.5},
        {"max_count": True},
        {"max_count": 95.5},
    ],
)
def test_point_sampling_config_rejects_invalid_contracts(kwargs):
    with pytest.raises(ValueError):
        PointSamplingConfig(**kwargs)


def test_fps_is_deterministic_and_uses_centroid_farthest_first():
    points = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [4.0, 0.0, 0.0]],
        dtype=np.float32,
    )

    indices = farthest_point_indices(points, 2)

    assert indices.tolist() == [2, 0]
    assert np.array_equal(indices, farthest_point_indices(points, 2))


def test_fps_uses_lowest_candidate_index_for_distance_ties():
    points = np.array([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32)

    assert farthest_point_indices(points, 2).tolist() == [0, 1]


def test_fps_rejects_invalid_shapes_and_counts():
    with pytest.raises(ValueError, match="shape"):
        farthest_point_indices(np.zeros((2, 2), dtype=np.float32), 1)
    with pytest.raises(ValueError, match="finite"):
        farthest_point_indices(
            np.array([[0.0, np.nan, 0.0]], dtype=np.float32), 1
        )
    with pytest.raises(ValueError, match="count"):
        farthest_point_indices(np.zeros((1, 3), dtype=np.float32), 2)


def test_cylindrical_position_uses_full_input_ranges():
    centers = np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 10.0]], dtype=np.float64)

    encoded = cylindrical_positions(centers)

    assert encoded.dtype == np.float32
    assert np.allclose(encoded[:, 0], [0.0, 1.0])
    assert np.allclose(encoded[:, 3], [0.0, 1.0])
    assert np.allclose(encoded[0, 1:3], [0.0, 1.0])


def test_cylindrical_position_zeroes_degenerate_ranges():
    centers = np.array([[1.0, 0.0, 3.0], [-1.0, 0.0, 3.0]], dtype=np.float64)

    encoded = cylindrical_positions(centers)

    assert np.array_equal(encoded[:, 0], np.zeros(2, dtype=np.float32))
    assert np.array_equal(encoded[:, 3], np.zeros(2, dtype=np.float32))
    assert np.allclose(encoded[:, 1:3], [[0.0, 1.0], [0.0, -1.0]])


class FakeProperties:
    def Mass(self):
        return 12.5

    def CentreOfMass(self):
        return SimpleNamespace(Coord=lambda: (1.0, 2.0, 3.0))


@pytest.mark.parametrize("api_style", ["legacy", "nested", "direct"])
def test_surface_properties_supports_lazy_occ_api_variants(monkeypatch, api_style):
    calls = []

    def surface_properties(shape, properties):
        calls.append((shape, properties))

    if api_style == "legacy":
        brep_gprop = SimpleNamespace(
            brepgprop_SurfaceProperties=surface_properties
        )
    elif api_style == "nested":
        brep_gprop = SimpleNamespace(
            brepgprop=SimpleNamespace(SurfaceProperties=surface_properties)
        )
    else:
        brep_gprop = SimpleNamespace(SurfaceProperties=surface_properties)
    core = ModuleType("OCC.Core")
    core.BRepGProp = brep_gprop
    core.GProp = SimpleNamespace(GProp_GProps=FakeProperties)
    occ = ModuleType("OCC")
    occ.Core = core
    monkeypatch.setitem(sys.modules, "OCC", occ)
    monkeypatch.setitem(sys.modules, "OCC.Core", core)
    face = SimpleNamespace(topods_shape=lambda: "topods-face")

    area, centroid = geometry_features._surface_properties(face)

    assert area == 12.5
    assert np.array_equal(centroid, np.array([1.0, 2.0, 3.0]))
    assert calls[0][0] == "topods-face"
    assert isinstance(calls[0][1], FakeProperties)


@pytest.mark.parametrize("singleton_channel", [False, True])
def test_uv_candidates_filter_invalid_entries_and_normalize_normals(
    monkeypatch, singleton_channel
):
    grid = 20
    points = np.zeros((grid, grid, 3), dtype=np.float64)
    normals = np.zeros((grid, grid, 3), dtype=np.float64)
    inside = np.zeros((grid, grid), dtype=np.float64)
    points.reshape(-1, 3)[0] = [1.0, 2.0, 3.0]
    normals.reshape(-1, 3)[0] = [0.0, 0.0, 2.0]
    inside.reshape(-1)[0] = 1.0
    inside.reshape(-1)[1] = np.nan
    inside.reshape(-1)[2] = 1.0
    normals.reshape(-1, 3)[2] = [0.0, 0.0, 0.0]
    inside.reshape(-1)[3] = 1.0
    points.reshape(-1, 3)[3] = [np.nan, 0.0, 0.0]
    if singleton_channel:
        inside = inside[..., None]

    def fake_uvgrid(face, num_u, num_v, method):
        assert face == "face"
        assert (num_u, num_v) == (grid, grid)
        return {"point": points, "normal": normals, "inside": inside}[method]

    uvgrid_module = ModuleType("occwl.uvgrid")
    uvgrid_module.uvgrid = fake_uvgrid
    occwl = ModuleType("occwl")
    occwl.uvgrid = uvgrid_module
    monkeypatch.setitem(sys.modules, "occwl", occwl)
    monkeypatch.setitem(sys.modules, "occwl.uvgrid", uvgrid_module)

    candidates = geometry_features._valid_uv_candidates("face", grid)

    assert candidates.dtype == np.float32
    assert candidates.shape == (1, 6)
    assert np.allclose(candidates[0], [1.0, 2.0, 3.0, 0.0, 0.0, 1.0])


def make_extraction(faces):
    return {
        "graph": {"num_nodes": len(faces)},
        "_extractor": SimpleNamespace(
            _graph=SimpleNamespace(
                nodes={
                    index: {"face": face}
                    for index, face in enumerate(faces)
                }
            )
        ),
    }


def candidate_rows(count, offset=0.0):
    rows = np.zeros((count, 6), dtype=np.float32)
    rows[:, 0] = np.arange(count, dtype=np.float32) + offset
    rows[:, 5] = 1.0
    return rows


def test_extraction_preserves_face_order_escalates_and_repeats_final_fps(
    monkeypatch,
):
    faces = ("first", "second")
    surface = {
        "first": (1.0, np.array([1.0, 0.0, 0.0])),
        "second": (1.0, np.array([0.0, 2.0, 10.0])),
    }
    calls = []
    monkeypatch.setattr(
        geometry_features, "_surface_properties", lambda face: surface[face]
    )

    def fake_candidates(face, grid):
        calls.append((face, grid))
        if face == "first":
            return candidate_rows(2 if grid == 20 else 4)
        return candidate_rows(2, offset=10.0)

    monkeypatch.setattr(
        geometry_features, "_valid_uv_candidates", fake_candidates
    )

    batch = extract_geometry_features(
        make_extraction(faces),
        PointSamplingConfig(mode="fixed", fixed_count=4),
        base_seed=2026,
        sample_id="sample",
    )

    assert calls == [
        ("first", 20),
        ("first", 32),
        ("second", 20),
        ("second", 32),
        ("second", 48),
        ("second", 64),
        ("second", 80),
    ]
    assert batch.face_point_count.tolist() == [4, 4]
    assert np.array_equal(
        batch.face_points[1, :2], batch.face_points[1, 2:4]
    )
    assert np.allclose(batch.cylindrical_pos[:, 0], [0.0, 1.0])
    assert np.allclose(batch.cylindrical_pos[:, 3], [0.0, 1.0])


def test_extraction_computes_all_surface_areas_before_adaptive_sampling(
    monkeypatch,
):
    faces = ("small", "median", "large")
    surface = {
        "small": (1.0, np.array([1.0, 0.0, 0.0])),
        "median": (4.0, np.array([2.0, 0.0, 1.0])),
        "large": (9.0, np.array([3.0, 0.0, 2.0])),
    }
    events = []

    def fake_surface(face):
        events.append(("surface", face))
        return surface[face]

    def fake_candidates(face, grid):
        events.append(("uv", face, grid))
        return candidate_rows(96)

    monkeypatch.setattr(geometry_features, "_surface_properties", fake_surface)
    monkeypatch.setattr(
        geometry_features, "_valid_uv_candidates", fake_candidates
    )

    batch = extract_geometry_features(
        make_extraction(faces),
        PointSamplingConfig(),
        base_seed=2026,
        sample_id="sample",
        include_cylindrical_pos=False,
    )

    assert events[:3] == [
        ("surface", "small"),
        ("surface", "median"),
        ("surface", "large"),
    ]
    assert batch.face_point_count.tolist() == [32, 64, 96]


def test_cylindrical_only_extraction_skips_all_uv_work(monkeypatch):
    faces = ("first", "second")
    surface = {
        "first": (1.0, np.array([1.0, 0.0, 0.0])),
        "second": (4.0, np.array([0.0, 2.0, 10.0])),
    }
    monkeypatch.setattr(
        geometry_features, "_surface_properties", lambda face: surface[face]
    )
    monkeypatch.setattr(
        geometry_features,
        "_valid_uv_candidates",
        lambda *args: pytest.fail("cylindrical-only extraction must not sample UV"),
    )

    batch = extract_geometry_features(
        make_extraction(faces),
        PointSamplingConfig(),
        base_seed=2026,
        sample_id="sample",
        include_face_pointcloud=False,
        include_cylindrical_pos=True,
    )

    assert not batch.face_point_mask.any()
    assert batch.face_point_count.tolist() == [0, 0]
    assert np.allclose(batch.cylindrical_pos[:, 0], [0.0, 1.0])


def test_point_only_extraction_skips_cylindrical_encoding(monkeypatch):
    monkeypatch.setattr(
        geometry_features,
        "_surface_properties",
        lambda face: (1.0, np.array([1.0, 0.0, 0.0])),
    )
    monkeypatch.setattr(
        geometry_features,
        "_valid_uv_candidates",
        lambda face, grid: candidate_rows(64),
    )
    monkeypatch.setattr(
        geometry_features,
        "cylindrical_positions",
        lambda *args: pytest.fail("point-only extraction must not encode cylindrical"),
    )

    batch = extract_geometry_features(
        make_extraction(("face",)),
        PointSamplingConfig(mode="fixed", fixed_count=64),
        base_seed=2026,
        sample_id="sample",
        include_face_pointcloud=True,
        include_cylindrical_pos=False,
    )

    assert batch.face_point_count.tolist() == [64]
    assert np.array_equal(
        batch.cylindrical_pos, np.zeros((1, 4), dtype=np.float32)
    )
