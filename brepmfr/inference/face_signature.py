"""OCC/AAG face-signature contract for NX face mapping."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from OCC.Core.Bnd import Bnd_Box
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
from OCC.Core.BRepBndLib import brepbndlib_AddOptimal
from OCC.Core.BRepGProp import brepgprop_LinearProperties, brepgprop_SurfaceProperties
from OCC.Core.GeomAbs import (
    GeomAbs_BSplineSurface,
    GeomAbs_BezierSurface,
    GeomAbs_Cone,
    GeomAbs_Cylinder,
    GeomAbs_Plane,
    GeomAbs_Sphere,
    GeomAbs_SurfaceOfRevolution,
    GeomAbs_Torus,
)
from OCC.Core.GProp import GProp_GProps
from OCC.Extend import TopologyUtils


FACE_MAPPING_CONTRACT_VERSION = "1.0"
NORMALIZATION_KIND = "aag_unit_box"

_NORMALIZATION = {
    "kind": NORMALIZATION_KIND,
    "box_side": 1.0,
    "scale_rule": "2_over_longest_bbox_extent",
}
_FACE_FIELDS = {
    "face_index",
    "surface_type",
    "area",
    "perimeter",
    "centroid",
    "bounding_box",
    "loop_count",
    "edge_count",
    "neighbor_count",
    "surface_parameters",
    "sample_points",
}
_PARAMETER_FIELDS = {
    "plane": {"origin", "axis"},
    "cylinder": {"origin", "axis", "radius"},
    "cone": {"origin", "axis", "half_angle"},
    "sphere": {"origin", "radius"},
    "torus": {"origin", "axis", "radius", "secondary_radius"},
    "bspline": set(),
    "revolution": set(),
    "other": set(),
}


def _point(value: object) -> List[float]:
    return [float(value.X()), float(value.Y()), float(value.Z())]


def _direction(value: object) -> List[float]:
    components = np.asarray(_point(value), dtype=np.float64)
    norm = float(np.linalg.norm(components))
    if not math.isfinite(norm) or norm <= 0.0:
        raise ValueError("Surface axis must have finite non-zero length")
    components /= norm
    for component in components:
        if abs(float(component)) > 1.0e-12:
            if component < 0.0:
                components *= -1.0
            break
    return [float(component) for component in components]


def _closest_point_on_plane_to_origin(
    point: Sequence[float], normal: Sequence[float]
) -> List[float]:
    location = np.asarray(point, dtype=np.float64)
    axis = np.asarray(normal, dtype=np.float64)
    closest = location - axis * float(np.dot(location, axis))
    return [float(component) for component in closest]


def _closest_point_on_axis_to_origin(
    point: Sequence[float], direction: Sequence[float]
) -> List[float]:
    location = np.asarray(point, dtype=np.float64)
    axis = np.asarray(direction, dtype=np.float64)
    closest = location - axis * float(np.dot(location, axis))
    return [float(component) for component in closest]


def _face_bbox(face: object) -> List[float]:
    box = Bnd_Box()
    box.SetGap(0.0)
    brepbndlib_AddOptimal(face, box, False, False)
    return [float(value) for value in box.Get()]


def _face_properties(face: object) -> Tuple[float, List[float]]:
    properties = GProp_GProps()
    brepgprop_SurfaceProperties(face, properties)
    return float(properties.Mass()), _point(properties.CentreOfMass())


def _face_topology(face: object) -> Tuple[float, int, int]:
    explorer = TopologyUtils.TopologyExplorer(face, ignore_orientation=True)
    edges = list(explorer.edges())
    perimeter = 0.0
    for edge in edges:
        properties = GProp_GProps()
        brepgprop_LinearProperties(edge, properties)
        perimeter += float(properties.Mass())
    return perimeter, len(list(explorer.wires())), len(edges)


def _surface_signature(face: object) -> Tuple[str, Dict[str, object]]:
    surface = BRepAdaptor_Surface(face)
    surface_type = surface.GetType()
    if surface_type == GeomAbs_Plane:
        plane = surface.Plane()
        axis = _direction(plane.Axis().Direction())
        return "plane", {
            "origin": _closest_point_on_plane_to_origin(
                _point(plane.Location()), axis
            ),
            "axis": axis,
        }
    if surface_type == GeomAbs_Cylinder:
        cylinder = surface.Cylinder()
        axis = _direction(cylinder.Axis().Direction())
        return "cylinder", {
            "origin": _closest_point_on_axis_to_origin(
                _point(cylinder.Axis().Location()), axis
            ),
            "axis": axis,
            "radius": float(cylinder.Radius()),
        }
    if surface_type == GeomAbs_Cone:
        cone = surface.Cone()
        return "cone", {
            "origin": _point(cone.Apex()),
            "axis": _direction(cone.Axis().Direction()),
            "half_angle": float(cone.SemiAngle()),
        }
    if surface_type == GeomAbs_Sphere:
        sphere = surface.Sphere()
        return "sphere", {
            "origin": _point(sphere.Location()),
            "radius": float(sphere.Radius()),
        }
    if surface_type == GeomAbs_Torus:
        torus = surface.Torus()
        return "torus", {
            "origin": _point(torus.Location()),
            "axis": _direction(torus.Axis().Direction()),
            "radius": float(torus.MajorRadius()),
            "secondary_radius": float(torus.MinorRadius()),
        }
    if surface_type in (GeomAbs_BSplineSurface, GeomAbs_BezierSurface):
        return "bspline", {}
    if surface_type == GeomAbs_SurfaceOfRevolution:
        return "revolution", {}
    return "other", {}


def _sample_points(grid: object) -> List[List[float]]:
    array = np.asarray(grid, dtype=np.float64)
    if array.ndim != 3 or array.shape[0] != 7:
        raise ValueError("graph_face_grid entries must have shape [7, u, v]")
    mask = array[6] > 0.5
    points = np.transpose(array[:3], (1, 2, 0))[mask]
    if len(points) == 0:
        raise ValueError("Every face must provide non-empty sample_points")
    return sorted(
        ([float(component) for component in point] for point in points),
        key=lambda point: (point[0], point[1], point[2]),
    )


def extract_face_mapping(
    extractor: object, extraction: Mapping[str, object]
) -> Dict[str, object]:
    graph = getattr(extractor, "_graph", None)
    if graph is None:
        raise ValueError("AAG extractor did not retain its topology graph")
    grids = extraction.get("graph_face_grid")
    if not isinstance(grids, (list, tuple, np.ndarray)):
        raise ValueError("AAG extraction is missing graph_face_grid")
    node_indices = sorted(int(index) for index in graph.nodes)
    if node_indices != list(range(len(node_indices))):
        raise ValueError("AAG face indices must be continuous from zero")
    if len(grids) != len(node_indices):
        raise ValueError("graph_face_grid count must match AAG face count")

    adjacency = sorted(
        {
            tuple(sorted((int(left), int(right))))
            for left, right in graph.edges
            if int(left) != int(right)
        }
    )
    degrees = [0] * len(node_indices)
    for left, right in adjacency:
        if left < 0 or right >= len(node_indices):
            raise ValueError("AAG adjacency endpoint is outside the face range")
        degrees[left] += 1
        degrees[right] += 1

    faces = []
    for face_index in node_indices:
        occwl_face = graph.nodes[face_index]["face"]
        topods_face = occwl_face.topods_shape()
        area, centroid = _face_properties(topods_face)
        perimeter, loop_count, edge_count = _face_topology(topods_face)
        surface_type, parameters = _surface_signature(topods_face)
        faces.append(
            {
                "face_index": face_index,
                "surface_type": surface_type,
                "area": area,
                "perimeter": perimeter,
                "centroid": centroid,
                "bounding_box": _face_bbox(topods_face),
                "loop_count": loop_count,
                "edge_count": edge_count,
                "neighbor_count": degrees[face_index],
                "surface_parameters": parameters,
                "sample_points": _sample_points(grids[face_index]),
            }
        )

    return validate_face_mapping_document(
        {
            "contract_version": FACE_MAPPING_CONTRACT_VERSION,
            "status": "ready",
            "normalization": dict(_NORMALIZATION),
            "faces": faces,
            "adjacency_edges": [list(edge) for edge in adjacency],
        },
        expected_face_count=len(faces),
    )


def _require_mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError("{0} must be an object".format(name))
    return value


def _require_keys(
    value: Mapping[str, object], expected: set, name: str
) -> None:
    actual = set(value)
    if actual != expected:
        raise ValueError(
            "{0} fields must be exactly {1}; got {2}".format(
                name, sorted(expected), sorted(actual)
            )
        )


def _require_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("{0} must be a finite number".format(name))
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError("{0} must be finite".format(name))
    return converted


def _require_integer(value: object, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(
            "{0} must be an integer >= {1}".format(name, minimum)
        )
    return value


def _require_vector(value: object, length: int, name: str) -> List[float]:
    if not isinstance(value, list) or len(value) != length:
        raise ValueError("{0} must contain exactly {1} values".format(name, length))
    return [_require_number(component, name) for component in value]


def _validate_surface_parameters(
    surface_type: object, value: object, face_index: int
) -> None:
    if surface_type not in _PARAMETER_FIELDS:
        raise ValueError("face[{0}].surface_type is unsupported".format(face_index))
    parameters = _require_mapping(
        value, "face[{0}].surface_parameters".format(face_index)
    )
    _require_keys(
        parameters,
        _PARAMETER_FIELDS[surface_type],
        "face[{0}].surface_parameters".format(face_index),
    )
    for name in ("origin", "axis"):
        if name in parameters:
            vector = _require_vector(
                parameters[name],
                3,
                "face[{0}].surface_parameters.{1}".format(face_index, name),
            )
            if name == "axis" and not math.isclose(
                math.sqrt(sum(component * component for component in vector)),
                1.0,
                abs_tol=1.0e-6,
                rel_tol=0.0,
            ):
                raise ValueError(
                    "face[{0}].surface_parameters.axis must be unit length".format(
                        face_index
                    )
                )
    for name in ("radius", "secondary_radius", "half_angle"):
        if name in parameters and _require_number(
            parameters[name],
            "face[{0}].surface_parameters.{1}".format(face_index, name),
        ) <= 0.0:
            raise ValueError(
                "face[{0}].surface_parameters.{1} must be positive".format(
                    face_index, name
                )
            )


def _validate_ready_document(
    document: Mapping[str, object], expected_face_count: Optional[int]
) -> None:
    _require_keys(
        document,
        {
            "contract_version",
            "status",
            "normalization",
            "faces",
            "adjacency_edges",
        },
        "face_mapping",
    )
    normalization = _require_mapping(document["normalization"], "normalization")
    if normalization != _NORMALIZATION:
        raise ValueError("normalization does not match the aag_unit_box contract")
    faces = document["faces"]
    if not isinstance(faces, list):
        raise ValueError("faces must be an array")
    if expected_face_count is not None and len(faces) != expected_face_count:
        raise ValueError("face_count does not match face_mapping.faces")

    for expected_index, raw_face in enumerate(faces):
        face = _require_mapping(raw_face, "face[{0}]".format(expected_index))
        _require_keys(face, _FACE_FIELDS, "face[{0}]".format(expected_index))
        if face["face_index"] != expected_index:
            raise ValueError("face_index values must be unique and continuous")
        _require_number(face["area"], "face[{0}].area".format(expected_index))
        _require_number(
            face["perimeter"], "face[{0}].perimeter".format(expected_index)
        )
        _require_vector(
            face["centroid"], 3, "face[{0}].centroid".format(expected_index)
        )
        bounding_box = _require_vector(
            face["bounding_box"],
            6,
            "face[{0}].bounding_box".format(expected_index),
        )
        if any(bounding_box[axis] > bounding_box[axis + 3] for axis in range(3)):
            raise ValueError("face[{0}].bounding_box is inverted".format(expected_index))
        _require_integer(
            face["loop_count"], "face[{0}].loop_count".format(expected_index), 1
        )
        _require_integer(
            face["edge_count"], "face[{0}].edge_count".format(expected_index), 1
        )
        _require_integer(
            face["neighbor_count"],
            "face[{0}].neighbor_count".format(expected_index),
        )
        _validate_surface_parameters(
            face["surface_type"], face["surface_parameters"], expected_index
        )
        samples = face["sample_points"]
        if not isinstance(samples, list) or not samples:
            raise ValueError(
                "face[{0}].sample_points must be non-empty".format(expected_index)
            )
        for sample in samples:
            _require_vector(
                sample, 3, "face[{0}].sample_points".format(expected_index)
            )

    raw_edges = document["adjacency_edges"]
    if not isinstance(raw_edges, list):
        raise ValueError("adjacency_edges must be an array")
    canonical_edges = []
    seen = set()
    degrees = [0] * len(faces)
    for edge in raw_edges:
        if not isinstance(edge, list) or len(edge) != 2:
            raise ValueError("adjacency edge must contain two endpoints")
        left = _require_integer(edge[0], "adjacency endpoint")
        right = _require_integer(edge[1], "adjacency endpoint")
        if left == right:
            raise ValueError("adjacency edge must not be a self-loop")
        if left > right:
            raise ValueError("adjacency edge endpoints must be sorted")
        if right >= len(faces):
            raise ValueError("adjacency endpoint is outside the face range")
        pair = (left, right)
        if pair in seen:
            raise ValueError("adjacency_edges must not contain duplicate edges")
        seen.add(pair)
        canonical_edges.append(pair)
        degrees[left] += 1
        degrees[right] += 1
    if canonical_edges != sorted(canonical_edges):
        raise ValueError("adjacency_edges must be sorted")
    for face_index, face in enumerate(faces):
        if face["neighbor_count"] != degrees[face_index]:
            raise ValueError(
                "face[{0}].neighbor_count does not match adjacency_edges".format(
                    face_index
                )
            )


def validate_face_mapping_document(
    document: Mapping[str, object], expected_face_count: Optional[int] = None
) -> Dict[str, object]:
    document = _require_mapping(document, "face_mapping")
    if document.get("contract_version") != FACE_MAPPING_CONTRACT_VERSION:
        raise ValueError("Unsupported face_mapping contract_version")
    status = document.get("status")
    if status == "unavailable":
        _require_keys(
            document,
            {"contract_version", "status", "reason"},
            "unavailable face_mapping",
        )
        reason = document["reason"]
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("Unavailable face_mapping requires a non-empty reason")
    elif status == "ready":
        _validate_ready_document(document, expected_face_count)
    else:
        raise ValueError("face_mapping status must be ready or unavailable")
    try:
        return json.loads(
            json.dumps(document, ensure_ascii=False, allow_nan=False)
        )
    except (TypeError, ValueError) as error:
        raise ValueError("face_mapping must be finite JSON: {0}".format(error)) from error


def read_face_mapping(
    path: Path, expected_face_count: Optional[int] = None
) -> Dict[str, object]:
    path = Path(path)
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("Invalid face mapping file {0}: {1}".format(path, error)) from error
    return validate_face_mapping_document(document, expected_face_count)


def face_mapping_json_bytes(document: Mapping[str, object]) -> bytes:
    validated = validate_face_mapping_document(document)
    return (
        json.dumps(validated, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
