from types import SimpleNamespace

import numpy as np

from brepmfr.inference.postprocess import (
    FEATURE_NAMES,
    cluster_instances,
    geometry_requirements,
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
