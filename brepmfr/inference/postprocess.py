"""Checkpoint requirements and deterministic instance postprocessing."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Dict, List, Tuple

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


@dataclass(frozen=True)
class PostprocessResult:
    """Stable records and numeric arrays derived from model predictions."""

    faces: List[Dict[str, object]]
    instances: List[Dict[str, object]]
    predicted_classes: np.ndarray
    face_confidences: np.ndarray
    affinity_probabilities: np.ndarray
    cluster_ids: np.ndarray


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


def postprocess_predictions(
    semantic_probabilities,
    affinity_logits,
    spatial_pos,
    threshold=0.5,
    background_class_id=24,
):
    """Build deterministic face and feature-instance prediction records."""
    semantic_probabilities = np.asarray(semantic_probabilities)
    if semantic_probabilities.ndim != 2 or semantic_probabilities.shape[1] != len(
        FEATURE_NAMES
    ):
        raise ValueError("semantic probabilities must have shape (N, 25)")
    node_count = semantic_probabilities.shape[0]
    affinity_logits = np.asarray(affinity_logits)
    if affinity_logits.shape != (node_count, node_count):
        raise ValueError("affinity logits must have shape (N, N)")
    spatial_pos = np.asarray(spatial_pos)
    if spatial_pos.shape != (node_count, node_count):
        raise ValueError("spatial_pos must have shape (N, N)")
    if not (
        np.isfinite(semantic_probabilities).all()
        and np.isfinite(affinity_logits).all()
        and np.isfinite(spatial_pos).all()
    ):
        raise ValueError("prediction arrays must contain only finite values")
    threshold = float(threshold)
    if not np.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("affinity threshold must be finite and in [0, 1]")
    background_class_id = int(background_class_id)
    if not 0 <= background_class_id < len(FEATURE_NAMES):
        raise ValueError("background class ID is outside the feature-name table")

    predicted_classes = semantic_probabilities.argmax(axis=1).astype(
        np.int64, copy=False
    )
    face_confidences = semantic_probabilities[
        np.arange(node_count), predicted_classes
    ]
    affinity_probabilities = 1.0 / (
        1.0 + np.exp(-np.clip(affinity_logits, -60.0, 60.0))
    )
    cluster_ids = cluster_instances(
        affinity_probabilities,
        predicted_classes,
        threshold=threshold,
        background_class_id=background_class_id,
        adjacency=spatial_pos == 2,
    )

    instances = []
    for instance_id in sorted(
        int(value) for value in np.unique(cluster_ids) if value >= 0
    ):
        indices = np.flatnonzero(cluster_ids == instance_id)
        class_id = int(predicted_classes[indices[0]])
        instances.append(
            {
                "instance_id": instance_id,
                "class_id": class_id,
                "feature_name": FEATURE_NAMES[class_id],
                "face_indices": [int(index) for index in indices],
                "confidence": float(np.mean(face_confidences[indices])),
            }
        )

    faces = []
    for face_index in range(node_count):
        class_id = int(predicted_classes[face_index])
        cluster_id = int(cluster_ids[face_index])
        faces.append(
            {
                "face_index": face_index,
                "class_id": class_id,
                "feature_name": FEATURE_NAMES[class_id],
                "confidence": float(face_confidences[face_index]),
                "instance_id": cluster_id if cluster_id >= 0 else None,
            }
        )

    return PostprocessResult(
        faces=faces,
        instances=instances,
        predicted_classes=predicted_classes,
        face_confidences=face_confidences,
        affinity_probabilities=affinity_probabilities,
        cluster_ids=cluster_ids,
    )

