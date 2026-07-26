"""Deterministic per-face point clouds and full-part cylindrical positions.

The pure NumPy helpers in this module are import-safe without pythonocc-core or
occwl.  CAD bindings are imported only by :func:`extract_geometry_features`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Tuple

import numpy as np

from utils.reproducibility import derive_seed


MAX_FACE_POINTS = 96
NORMAL_EPSILON = 1e-12
ADAPTIVE_BASE_COUNT = 64
DEFAULT_CANDIDATE_GRIDS = (20, 32, 48, 64, 80)


@dataclass(frozen=True)
class PointSamplingConfig:
    """Point-count and UV-candidate settings stored in each geometry BIN."""

    mode: str = "area_adaptive"
    fixed_count: int = 64
    min_count: int = 32
    max_count: int = 96
    area_exponent: float = 0.5
    candidate_grids: Tuple[int, int, int, int, int] = DEFAULT_CANDIDATE_GRIDS

    def __post_init__(self) -> None:
        object.__setattr__(self, "candidate_grids", tuple(self.candidate_grids))
        if self.mode not in ("fixed", "area_adaptive"):
            raise ValueError("mode must be 'fixed' or 'area_adaptive'")
        for name in ("fixed_count", "min_count", "max_count"):
            value = getattr(self, name)
            if not isinstance(value, (int, np.integer)) or isinstance(
                value, (bool, np.bool_)
            ):
                raise ValueError("{0} must be a non-boolean integer".format(name))
        if not 1 <= self.fixed_count <= MAX_FACE_POINTS:
            raise ValueError("fixed_count must be in [1, 96]")
        if not 1 <= self.min_count <= self.max_count <= MAX_FACE_POINTS:
            raise ValueError("point counts must satisfy 1 <= min <= max <= 96")
        if self.mode == "area_adaptive" and (
            self.min_count % 8 != 0 or self.max_count % 8 != 0
        ):
            raise ValueError("area-adaptive min_count and max_count must be multiples of 8")
        if not math.isfinite(self.area_exponent) or self.area_exponent <= 0.0:
            raise ValueError("area_exponent must be positive and finite")
        if self.candidate_grids != DEFAULT_CANDIDATE_GRIDS:
            raise ValueError(
                "candidate_grids must be exactly (20, 32, 48, 64, 80)"
            )


@dataclass(frozen=True)
class GeometryFeatureBatch:
    """Fixed-shape geometry arrays aligned to complete-extraction face order."""

    face_points: np.ndarray
    face_point_mask: np.ndarray
    face_point_count: np.ndarray
    cylindrical_pos: np.ndarray


def adaptive_point_count(
    area: float, median_area: float, config: PointSamplingConfig
) -> int:
    """Return a fixed or area-adaptive target count with half-up round-to-eight."""
    area = float(area)
    median_area = float(median_area)
    if not math.isfinite(area) or area <= 0.0:
        raise ValueError("area must be positive and finite")
    if not math.isfinite(median_area) or median_area <= 0.0:
        raise ValueError("median_area must be positive and finite")
    if config.mode == "fixed":
        return int(config.fixed_count)
    requested = ADAPTIVE_BASE_COUNT * math.pow(
        area / median_area, config.area_exponent
    )
    clipped = min(float(config.max_count), max(float(config.min_count), requested))
    rounded = int(math.floor(clipped / 8.0 + 0.5)) * 8
    return int(min(config.max_count, max(config.min_count, rounded)))


def expected_point_counts(
    face_areas: np.ndarray, config: PointSamplingConfig
) -> np.ndarray:
    """Compute exact targets from float32-stored areas using float64 arithmetic."""
    stored_areas = np.asarray(face_areas)
    if stored_areas.ndim != 1 or stored_areas.shape[0] == 0:
        raise ValueError("face_areas must have shape [N] with N > 0")
    if not np.isfinite(stored_areas).all() or bool((stored_areas <= 0).any()):
        raise ValueError("face_areas must be positive and finite")
    calculation_areas = stored_areas.astype(np.float32).astype(np.float64)
    median_area = float(np.median(calculation_areas))
    return np.asarray(
        [
            adaptive_point_count(area, median_area, config)
            for area in calculation_areas
        ],
        dtype=np.int32,
    )


def farthest_point_indices(points_xyz: np.ndarray, count: int) -> np.ndarray:
    """Select deterministic 3D FPS indices with centroid-farthest initialization."""
    points = np.asarray(points_xyz)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points_xyz must have shape [K, 3]")
    if not np.isfinite(points).all():
        raise ValueError("points_xyz must contain only finite values")
    if (
        not isinstance(count, (int, np.integer))
        or isinstance(count, (bool, np.bool_))
        or count < 1
        or count > points.shape[0]
    ):
        raise ValueError("count must be in [1, number of candidate points]")

    working = points.astype(np.float64, copy=False)
    centroid = working.mean(axis=0)
    first = int(np.argmax(np.sum((working - centroid) ** 2, axis=1)))
    selected = np.empty(int(count), dtype=np.int64)
    selected[0] = first
    minimum_distances = np.sum((working - working[first]) ** 2, axis=1)
    minimum_distances[first] = -1.0

    for selection_index in range(1, int(count)):
        next_index = int(np.argmax(minimum_distances))
        selected[selection_index] = next_index
        next_distances = np.sum(
            (working - working[next_index]) ** 2, axis=1
        )
        minimum_distances = np.minimum(minimum_distances, next_distances)
        minimum_distances[selected[: selection_index + 1]] = -1.0
    return selected


def cylindrical_positions(
    surface_centroids: np.ndarray, eps: float = 1e-12
) -> np.ndarray:
    """Encode ``[r_norm, sin(theta), cos(theta), z_norm]`` over full ranges."""
    centers = np.asarray(surface_centroids, dtype=np.float64)
    if centers.ndim != 2 or centers.shape[1] != 3 or centers.shape[0] == 0:
        raise ValueError("surface_centroids must have shape [N, 3] with N > 0")
    if not np.isfinite(centers).all():
        raise ValueError("surface_centroids must contain only finite values")
    if not math.isfinite(float(eps)) or eps < 0.0:
        raise ValueError("eps must be finite and non-negative")

    radii = np.hypot(centers[:, 0], centers[:, 1])
    radius_range = float(radii.max() - radii.min())
    z_values = centers[:, 2]
    z_range = float(z_values.max() - z_values.min())
    radius_normalized = (
        (radii - radii.min()) / radius_range
        if radius_range >= eps and radius_range > 0.0
        else np.zeros_like(radii)
    )
    z_normalized = (
        (z_values - z_values.min()) / z_range
        if z_range >= eps and z_range > 0.0
        else np.zeros_like(z_values)
    )
    theta = np.arctan2(centers[:, 1], centers[:, 0])
    return np.stack(
        [radius_normalized, np.sin(theta), np.cos(theta), z_normalized],
        axis=1,
    ).astype(np.float32)


def _surface_properties(face: object) -> Tuple[float, np.ndarray]:
    """Return true OCC surface area and center of mass for one occwl face."""
    from OCC.Core import BRepGProp, GProp

    properties = GProp.GProp_GProps()
    topods_face = face.topods_shape()
    if hasattr(BRepGProp, "brepgprop_SurfaceProperties"):
        BRepGProp.brepgprop_SurfaceProperties(topods_face, properties)
    elif hasattr(BRepGProp, "brepgprop"):
        BRepGProp.brepgprop.SurfaceProperties(topods_face, properties)
    else:
        BRepGProp.SurfaceProperties(topods_face, properties)
    area = float(properties.Mass())
    centroid = np.asarray(properties.CentreOfMass().Coord(), dtype=np.float64)
    if not math.isfinite(area) or area <= 0.0:
        raise ValueError("OCC returned a non-positive or non-finite face area")
    if centroid.shape != (3,) or not np.isfinite(centroid).all():
        raise ValueError("OCC returned an invalid surface centroid")
    return area, centroid


def _valid_uv_candidates(face: object, grid: int) -> np.ndarray:
    """Evaluate and filter one occwl UV candidate grid."""
    from occwl.uvgrid import uvgrid

    points = np.asarray(uvgrid(face, grid, grid, method="point"), dtype=np.float64)
    normals = np.asarray(uvgrid(face, grid, grid, method="normal"), dtype=np.float64)
    visibility = np.asarray(uvgrid(face, grid, grid, method="inside"))
    if points.shape != (grid, grid, 3) or normals.shape != (grid, grid, 3):
        raise ValueError("occwl.uvgrid returned an unexpected point or normal shape")
    if visibility.shape not in ((grid, grid), (grid, grid, 1)):
        raise ValueError("occwl.uvgrid returned an unexpected visibility-mask shape")

    points = points.reshape(-1, 3)
    normals = normals.reshape(-1, 3)
    visibility = visibility.reshape(-1)
    valid_visibility = np.isfinite(visibility) & (visibility > 0.5)
    normal_norm = np.linalg.norm(normals, axis=1)
    valid = (
        valid_visibility
        & np.isfinite(points).all(axis=1)
        & np.isfinite(normals).all(axis=1)
        & np.isfinite(normal_norm)
        & (normal_norm > NORMAL_EPSILON)
    )
    points = points[valid]
    normals = normals[valid]
    if points.shape[0] == 0:
        return np.empty((0, 6), dtype=np.float32)
    normals = normals / normal_norm[valid, None]
    return np.concatenate([points, normals], axis=1).astype(np.float32)


def _face_objects(extraction: Mapping) -> Tuple[object, ...]:
    try:
        graph_data = extraction["graph"]
        node_count = int(graph_data["num_nodes"])
        nx_graph = extraction["_extractor"]._graph
    except (AttributeError, KeyError, TypeError, ValueError) as error:
        raise ValueError(
            "Extraction must retain graph.num_nodes and _extractor._graph faces"
        ) from error
    if node_count < 1:
        raise ValueError("Geometry extraction requires at least one face")
    try:
        return tuple(nx_graph.nodes[index]["face"] for index in range(node_count))
    except (KeyError, IndexError, TypeError) as error:
        raise ValueError("Extraction faces are not densely indexed from zero") from error


def extract_geometry_features(
    extraction: Mapping,
    config: PointSamplingConfig,
    base_seed: int,
    sample_id: str,
    include_face_pointcloud: bool = True,
    include_cylindrical_pos: bool = True,
) -> GeometryFeatureBatch:
    """Extract complete-model geometry and pack each face to 96 positions.

    FPS is fully deterministic and uses no random choices.  Deriving the
    purpose-scoped seed here keeps conversion provenance on the shared seed
    contract without making tie-breaking depend on a process RNG.
    """
    derive_seed(base_seed, "point_sampling", sample_id)
    faces = _face_objects(extraction)
    areas = np.empty(len(faces), dtype=np.float64)
    centroids = np.empty((len(faces), 3), dtype=np.float64)
    for face_index, face in enumerate(faces):
        areas[face_index], centroids[face_index] = _surface_properties(face)

    target_counts = (
        expected_point_counts(areas, config)
        if include_face_pointcloud
        else np.zeros(len(faces), dtype=np.int32)
    )
    packed_points = np.zeros(
        (len(faces), MAX_FACE_POINTS, 6), dtype=np.float32
    )
    packed_mask = np.zeros((len(faces), MAX_FACE_POINTS), dtype=np.bool_)

    if include_face_pointcloud:
        for face_index, (face, target_count) in enumerate(
            zip(faces, target_counts)
        ):
            candidates = np.empty((0, 6), dtype=np.float32)
            for grid in config.candidate_grids:
                candidates = _valid_uv_candidates(face, grid)
                if candidates.shape[0] >= int(target_count):
                    break
            if candidates.shape[0] == 0:
                raise ValueError(
                    "Face {0} has no valid UV candidates".format(face_index)
                )
            unique_count = min(int(target_count), candidates.shape[0])
            fps_order = farthest_point_indices(candidates[:, :3], unique_count)
            if unique_count < int(target_count):
                repeats = int(math.ceil(float(target_count) / unique_count))
                selected = np.tile(fps_order, repeats)[: int(target_count)]
            else:
                selected = fps_order
            packed_points[face_index, : int(target_count)] = candidates[selected]
            packed_mask[face_index, : int(target_count)] = True

    return GeometryFeatureBatch(
        face_points=packed_points,
        face_point_mask=packed_mask,
        face_point_count=target_counts,
        cylindrical_pos=(
            cylindrical_positions(centroids)
            if include_cylindrical_pos
            else np.zeros((len(faces), 4), dtype=np.float32)
        ),
    )
