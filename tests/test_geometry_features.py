import numpy as np
import pytest

from tools.geometry_features import (
    PointSamplingConfig,
    adaptive_point_count,
    cylindrical_positions,
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
