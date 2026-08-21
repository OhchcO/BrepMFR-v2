"""Checkpoint requirements and deterministic instance postprocessing."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Tuple

import numpy as np


FEATURE_NAMES: Tuple[str, ...] = (
    "chamfer",
    "through_hole",
    "triangular_passage",
    "rectangular_passage",
    "6sides_passage",
    "triangular_through_slot",
    "rectangular_through_slot",
    "circular_through_slot",
    "rectangular_through_step",
    "2sides_through_step",
    "slanted_through_step",
    "Oring",
    "blind_hole",
    "triangular_pocket",
    "rectangular_pocket",
    "6sides_pocket",
    "circular_end_pocket",
    "rectangular_blind_slot",
    "v_circular_end_blind_slot",
    "h_circular_end_blind_slot",
    "triangular_blind_step",
    "circular_blind_step",
    "rectangular_blind_step",
    "round",
    "stock",
)


@dataclass(frozen=True)
class GeometryRequirements:
    """Strict BIN fields selected by checkpoint-saved model switches."""

    use_face_uv: bool
    require_face_pointcloud: bool
    require_parametric_surface: bool
    require_cylindrical_pos: bool


def configuration_value(configuration, name, default):
    """Read a checkpoint option from a mapping or attribute-backed object."""
    if isinstance(configuration, Mapping):
        return configuration.get(name, default)
    return getattr(configuration, name, default)


def geometry_requirements(configuration):
    """Translate checkpoint hyperparameters into strict dataset requirements."""
    return GeometryRequirements(
        use_face_uv=bool(configuration_value(configuration, "use_face_uv", True)),
        require_face_pointcloud=bool(
            configuration_value(configuration, "use_face_pointcloud", False)
        ),
        require_parametric_surface=(
            configuration_value(
                configuration, "face_geometry_encoder", "mini_point"
            )
            == "param_surface"
        ),
        require_cylindrical_pos=bool(
            configuration_value(configuration, "use_cylindrical_pos", False)
        ),
    )


def _validate_square_matrix(matrix, name):
    matrix = np.asarray(matrix)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("{} must be a square matrix".format(name))
    return matrix


def cluster_instances(
    affinity,
    semantic,
    threshold=0.5,
    background_class_id=24,
    adjacency=None,
):
    """Cluster same-semantic adjacent faces with deterministic union-find."""
    affinity = _validate_square_matrix(affinity, "affinity")
    semantic = np.asarray(semantic)
    node_count = affinity.shape[0]
    if semantic.shape != (node_count,):
        raise ValueError("semantic must have one entry per face")
    if adjacency is None:
        adjacency = np.ones_like(affinity, dtype=bool)
        np.fill_diagonal(adjacency, False)
    else:
        adjacency = _validate_square_matrix(adjacency, "adjacency")
        if adjacency.shape != affinity.shape:
            raise ValueError("adjacency and affinity must have equal shape")
        adjacency = np.asarray(adjacency, dtype=bool)

    parent = np.arange(node_count, dtype=np.int64)

    def find(node):
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = int(parent[node])
        return node

    def union(left, right):
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[max(left_root, right_root)] = min(left_root, right_root)

    active = semantic != background_class_id
    eligible = (
        (affinity >= threshold)
        & (adjacency | adjacency.T)
        & active[:, None]
        & active[None, :]
        & (semantic[:, None] == semantic[None, :])
    )
    for left, right in zip(*np.where(np.triu(eligible, k=1))):
        union(int(left), int(right))

    clusters = np.full(node_count, -1, dtype=np.int64)
    root_to_cluster = {}
    for node in range(node_count):
        if not active[node]:
            continue
        root = find(node)
        if root not in root_to_cluster:
            root_to_cluster[root] = len(root_to_cluster)
        clusters[node] = root_to_cluster[root]
    return clusters

