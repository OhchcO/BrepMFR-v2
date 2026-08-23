from types import SimpleNamespace

import numpy as np
import pytest

from brepmfr.inference.postprocess import (
    FEATURE_NAMES,
    cluster_instances,
    geometry_requirements,
    postprocess_predictions,
)


def test_feature_names_match_mfrinstseg_contract():
    assert len(FEATURE_NAMES) == 25
    assert FEATURE_NAMES[0] == "chamfer"
    assert FEATURE_NAMES[1] == "through_hole"
    assert FEATURE_NAMES[23] == "round"
    assert FEATURE_NAMES[24] == "stock"


def test_geometry_requirements_are_derived_from_checkpoint_mapping():
    requirements = geometry_requirements(
        {
            "use_face_uv": False,
            "use_face_pointcloud": True,
            "face_geometry_encoder": "param_surface",
            "use_cylindrical_pos": True,
        }
    )

    assert requirements.use_face_uv is False
    assert requirements.require_face_pointcloud is True
    assert requirements.require_parametric_surface is True
    assert requirements.require_cylindrical_pos is True


def test_geometry_requirements_accept_object_configuration():
    requirements = geometry_requirements(
        SimpleNamespace(
            use_face_uv=True,
            use_face_pointcloud=False,
            face_geometry_encoder="pointnet",
            use_cylindrical_pos=False,
        )
    )

    assert requirements.use_face_uv is True
    assert requirements.require_face_pointcloud is False
    assert requirements.require_parametric_surface is False
    assert requirements.require_cylindrical_pos is False


def test_cluster_instances_uses_semantic_topology_and_excludes_stock():
    affinity = np.ones((5, 5), dtype=np.float32)
    semantic = np.asarray([1, 1, 1, 2, 24], dtype=np.int64)
    adjacency = np.zeros((5, 5), dtype=bool)
    adjacency[0, 1] = adjacency[1, 0] = True
    adjacency[1, 2] = adjacency[2, 1] = True
    adjacency[3, 4] = adjacency[4, 3] = True

    clusters = cluster_instances(
        affinity,
        semantic,
        threshold=0.5,
        background_class_id=24,
        adjacency=adjacency,
    )

    assert clusters.tolist() == [0, 0, 0, 1, -1]


def test_cluster_instances_respects_threshold_and_is_deterministic():
    affinity = np.asarray(
        [
            [1.0, 0.7, 0.1, 0.1],
            [0.7, 1.0, 0.6, 0.1],
            [0.1, 0.6, 1.0, 0.9],
            [0.1, 0.1, 0.9, 1.0],
        ],
        dtype=np.float32,
    )
    semantic = np.asarray([3, 3, 3, 3], dtype=np.int64)
    adjacency = np.ones((4, 4), dtype=bool)

    clusters = cluster_instances(
        affinity,
        semantic,
        threshold=0.65,
        background_class_id=24,
        adjacency=adjacency,
    )

    assert clusters.tolist() == [0, 0, 1, 1]


def test_postprocess_builds_consistent_face_and_instance_records():
    probabilities = np.zeros((3, 25), dtype=np.float32)
    probabilities[0, [0, 1, 24]] = [0.1, 0.8, 0.1]
    probabilities[1, [0, 1, 24]] = [0.1, 0.7, 0.2]
    probabilities[2, [0, 1, 24]] = [0.1, 0.1, 0.8]
    logits = np.asarray(
        [
            [1000.0, 10.0, -10.0],
            [10.0, 1000.0, -10.0],
            [-10.0, -10.0, -1000.0],
        ],
        dtype=np.float32,
    )
    spatial_pos = np.asarray(
        [[1, 2, 3], [2, 1, 3], [3, 3, 1]], dtype=np.int64
    )

    result = postprocess_predictions(probabilities, logits, spatial_pos)

    assert result.cluster_ids.tolist() == [0, 0, -1]
    assert result.instances == [
        {
            "instance_id": 0,
            "class_id": 1,
            "feature_name": "through_hole",
            "face_indices": [0, 1],
            "confidence": pytest.approx(0.75),
        }
    ]
    assert [face["instance_id"] for face in result.faces] == [0, 0, None]
    assert [face["feature_name"] for face in result.faces] == [
        "through_hole",
        "through_hole",
        "stock",
    ]
    assert result.faces[2]["confidence"] == pytest.approx(0.8)
    assert np.isfinite(result.affinity_probabilities).all()
    assert result.affinity_probabilities[0, 0] == pytest.approx(1.0)
    assert result.affinity_probabilities[2, 2] > 0.0


@pytest.mark.parametrize(
    ("probabilities", "logits", "spatial_pos", "message"),
    [
        (np.zeros((2, 24)), np.zeros((2, 2)), np.zeros((2, 2)), "25"),
        (np.zeros((2, 25)), np.zeros((3, 3)), np.zeros((2, 2)), "affinity"),
        (np.zeros((2, 25)), np.zeros((2, 2)), np.zeros((3, 3)), "spatial"),
        (
            np.full((2, 25), np.nan),
            np.zeros((2, 2)),
            np.zeros((2, 2)),
            "finite",
        ),
    ],
)
def test_postprocess_rejects_invalid_prediction_contracts(
    probabilities, logits, spatial_pos, message
):
    with pytest.raises(ValueError, match=message):
        postprocess_predictions(probabilities, logits, spatial_pos)
