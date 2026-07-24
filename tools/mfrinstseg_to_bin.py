#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Convert sampled MFRInstSeg STEP files to the BrepMFR DGL BIN contract.

This utility intentionally has no import-time dependency on the legacy OCC
extractor.  The extractor is loaded from ``--extractor-file`` only when a
conversion is requested, which keeps this module usable for CLI help and
source inspection in environments without PyOCC.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import dgl
import numpy as np
import torch
from dgl.data.utils import load_graphs, save_graphs


FACE_TYPE = {
    "Plane": 0,
    "Cylinder": 1,
    "Cone": 2,
    "SphereFaceAttribute": 3,
    "TorusFaceAttribute": 4,
}
EDGE_TYPE = {
    "StraightEdgeAttribute": 0,
    "CircularEdgeAttribute": 1,
    "EllipticalEdgeAttribute": 2,
    "NonRationalBSplineEdgeAttribute": 3,
    "RationalBSplineEdgeAttribute": 4,
}
REQUIRED_NODE_KEYS = {"x", "y", "z", "l", "a", "f"}
REQUIRED_EDGE_KEYS = {"x", "t", "l", "a", "c"}
REQUIRED_METADATA_KEYS = {
    "edges_path",
    "spatial_pos",
    "d2_distance",
    "angle_distance",
    "instance",
    "bottom",
}
GRID_SIZE = 5
FEATURE_CHANNELS = 7
MAX_HOPS = 6
RBF_BINS = 64
UNREACHABLE_DISTANCE = 512


@dataclass(frozen=True)
class LabelData:
    """Dense MFRInstSeg annotations for a single CAD model."""

    sample_id: str
    semantic: np.ndarray
    instance: np.ndarray
    bottom: np.ndarray


@dataclass(frozen=True)
class ConversionPaths:
    """Filesystem locations shared by one conversion worker."""

    dataset_root: Path
    subset_root: Path
    output_root: Path

    @property
    def bins_root(self) -> Path:
        return self.output_root / "bin"

    def step_file(self, sample_id: str) -> Path:
        primary = self.dataset_root / "steps" / (sample_id + ".step")
        if primary.is_file():
            return primary
        alternate = primary.with_suffix(".stp")
        return alternate if alternate.is_file() else primary

    def label_file(self, sample_id: str) -> Path:
        return self.dataset_root / "labels" / (sample_id + ".json")

    def bin_file(self, sample_id: str) -> Path:
        return self.bins_root / (sample_id + ".bin")


def load_extractor(extractor_file: Path):
    """Dynamically import and return ``AAGExtractor`` and ``TopologyUtils``."""
    extractor_file = extractor_file.resolve()
    if not extractor_file.is_file():
        raise FileNotFoundError("Extractor file not found: {0}".format(extractor_file))
    specification = importlib.util.spec_from_file_location(
        "mfrinstseg_legacy_extractor", str(extractor_file)
    )
    if specification is None or specification.loader is None:
        raise ImportError("Cannot create an import specification for {0}".format(extractor_file))
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    try:
        return module.AAGExtractor, module.TopologyUtils
    except AttributeError as error:
        raise ImportError(
            "Extractor must define AAGExtractor and TopologyUtils: {0}".format(
                extractor_file
            )
        ) from error


def load_schema(schema_file: Path) -> dict:
    """Read and validate face attributes, edge attributes, and the UV grid."""
    try:
        with schema_file.open("r", encoding="utf-8") as stream:
            schema = json.load(stream)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("Cannot read schema {0}: {1}".format(schema_file, error)) from error

    if not isinstance(schema, dict):
        raise ValueError("Schema must be a JSON object: {0}".format(schema_file))
    for key in ("face_attributes", "edge_attributes", "UV-grid"):
        if key not in schema:
            raise ValueError("Schema is missing {0}: {1}".format(key, schema_file))
    if not isinstance(schema["face_attributes"], list) or not isinstance(
        schema["edge_attributes"], list
    ):
        raise ValueError("Schema attribute lists are invalid: {0}".format(schema_file))
    required_face = set(FACE_TYPE).union({"FaceAreaAttribute"})
    required_edge = set(EDGE_TYPE).union(
        {"Concave edge", "Convex edge", "Smooth", "EdgeLengthAttribute"}
    )
    missing_face = sorted(required_face.difference(schema["face_attributes"]))
    missing_edge = sorted(required_edge.difference(schema["edge_attributes"]))
    if missing_face or missing_edge:
        raise ValueError(
            "Schema is missing face attributes {0} or edge attributes {1}".format(
                missing_face, missing_edge
            )
        )
    uv_grid = schema["UV-grid"]
    if not isinstance(uv_grid, dict):
        raise ValueError("Schema UV-grid must be an object: {0}".format(schema_file))
    expected_uv = {"num_srf_u": GRID_SIZE, "num_srf_v": GRID_SIZE, "num_crv_u": GRID_SIZE}
    for key, expected in expected_uv.items():
        if uv_grid.get(key) != expected:
            raise ValueError(
                "Schema {0} must set {1} to {2}".format(schema_file, key, expected)
            )
    return schema


def _unwrap_label(document: object, label_file: Path) -> Tuple[str, Mapping[str, object]]:
    if (
        isinstance(document, list)
        and len(document) == 1
        and isinstance(document[0], list)
        and len(document[0]) == 2
        and isinstance(document[0][1], dict)
    ):
        return str(document[0][0]), document[0][1]
    if isinstance(document, dict):
        return label_file.stem, document
    raise ValueError("Unsupported label format: {0}".format(label_file))


def _dense_semantic(values: object, label_file: Path) -> np.ndarray:
    if isinstance(values, dict):
        try:
            indexed = {int(key): int(value) for key, value in values.items()}
        except (TypeError, ValueError) as error:
            raise ValueError("Invalid semantic labels: {0}".format(label_file)) from error
        if set(indexed) != set(range(len(indexed))):
            raise ValueError(
                "Semantic face indices are not contiguous from zero: {0}".format(
                    label_file
                )
            )
        return np.asarray([indexed[index] for index in range(len(indexed))], dtype=np.int32)
    if isinstance(values, list):
        return np.asarray(values, dtype=np.int32)
    raise ValueError("Label seg must be a dictionary or list: {0}".format(label_file))


def load_label(label_file: Path) -> LabelData:
    """Return dense semantic, instance, and bottom arrays."""
    try:
        with label_file.open("r", encoding="utf-8") as stream:
            document = json.load(stream)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("Cannot read label {0}: {1}".format(label_file, error)) from error

    sample_id, annotation = _unwrap_label(document, label_file)
    if not {"seg", "inst", "bottom"}.issubset(annotation):
        raise ValueError("Label is missing seg, inst, or bottom: {0}".format(label_file))
    semantic = _dense_semantic(annotation["seg"], label_file)
    instance = np.asarray(annotation["inst"], dtype=np.int8)
    bottom = np.asarray(annotation["bottom"], dtype=np.int32)
    count = len(semantic)
    if instance.shape != (count, count):
        raise ValueError(
            "Instance matrix must be square for {0}; got {1}".format(
                label_file, tuple(instance.shape)
            )
        )
    if bottom.shape != (count,):
        raise ValueError(
            "Bottom labels must have {0} entries for {1}; got {2}".format(
                count, label_file, tuple(bottom.shape)
            )
        )
    return LabelData(sample_id, semantic, instance, bottom)


def _attribute_index(attributes: Sequence[object], name: str) -> int:
    try:
        return list(attributes).index(name)
    except ValueError as error:
        raise ValueError("Schema is missing required attribute: {0}".format(name)) from error


def _finite_array(value: object, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if not np.isfinite(array).all():
        raise ValueError("Non-finite values in {0}".format(name))
    return array


def _signed_edge_angles(edge_grid: np.ndarray, convexity: np.ndarray) -> np.ndarray:
    """Return one signed dihedral angle per edge from normal grid channels."""
    left = edge_grid[:, 6:9, :].transpose(0, 2, 1)
    right = edge_grid[:, 9:12, :].transpose(0, 2, 1)
    dots = np.clip(np.sum(left * right, axis=-1), -1.0, 1.0)
    valid = (np.linalg.norm(left, axis=-1) > 0.0) & (
        np.linalg.norm(right, axis=-1) > 0.0
    )
    valid_count = valid.sum(axis=1)
    unsigned = np.divide(
        (np.arccos(dots) * valid).sum(axis=1),
        valid_count,
        out=np.zeros(edge_grid.shape[0], dtype=np.float32),
        where=valid_count > 0,
    ).astype(np.float32)
    signed = np.zeros_like(unsigned)
    signed[convexity == 1] = unsigned[convexity == 1]
    signed[convexity == 2] = -unsigned[convexity == 2]
    return signed


def _face_loop_counts(extraction: Mapping[str, object], node_count: int, topology_utils: object) -> np.ndarray:
    """Read the real wire count of each retained occwl face."""
    try:
        nx_graph = extraction["_extractor"]._graph
    except (AttributeError, KeyError) as error:
        raise ValueError("Extraction does not retain _extractor._graph topology") from error
    loop_counts = np.zeros(node_count, dtype=np.int32)
    for face_index in range(node_count):
        try:
            occwl_face = nx_graph.nodes[face_index]["face"]
            explorer = topology_utils.TopologyExplorer(
                occwl_face.topods_shape(), ignore_orientation=True
            )
            loop_counts[face_index] = int(explorer.number_of_wires())
        except (AttributeError, KeyError, IndexError, TypeError, ValueError) as error:
            raise ValueError(
                "Cannot obtain the loop count for face {0}".format(face_index)
            ) from error
    return loop_counts


def build_graph(
    extraction: dict,
    labels: LabelData,
    schema: dict,
    topology_utils: object,
):
    """Return a DGL graph containing x/y/z/l/a/f and x/t/l/a/c."""
    graph_data = extraction.get("graph")
    if not isinstance(graph_data, dict) or "edges" not in graph_data:
        raise ValueError("Extraction is missing graph edges")
    try:
        source, destination = graph_data["edges"]
        node_count = int(graph_data["num_nodes"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Extraction graph is malformed") from error
    if node_count != len(labels.semantic):
        raise ValueError(
            "Extractor has {0} faces but label has {1}".format(
                node_count, len(labels.semantic)
            )
        )
    if len(source) != len(destination):
        raise ValueError("Extraction graph source and destination lengths differ")
    graph = dgl.graph((source, destination), num_nodes=node_count, idtype=torch.int64)
    edge_count = graph.num_edges()

    node_grid_raw = _finite_array(extraction.get("graph_face_grid"), "graph_face_grid")
    if node_grid_raw.shape != (node_count, FEATURE_CHANNELS, GRID_SIZE, GRID_SIZE):
        raise ValueError(
            "Face grid must have shape {0}; got {1}".format(
                (node_count, FEATURE_CHANNELS, GRID_SIZE, GRID_SIZE),
                tuple(node_grid_raw.shape),
            )
        )
    node_grid = np.transpose(node_grid_raw, (0, 2, 3, 1)).copy()
    face_attributes = _finite_array(extraction.get("graph_face_attr"), "graph_face_attr")
    if face_attributes.ndim != 2 or face_attributes.shape[0] != node_count:
        raise ValueError("Face attributes do not match the face count")
    face_names = schema["face_attributes"]
    area_index = _attribute_index(face_names, "FaceAreaAttribute")
    if face_attributes.shape[1] <= area_index:
        raise ValueError("Face attributes do not include FaceAreaAttribute")
    face_codes = np.zeros(node_count, dtype=np.int32)
    for name, code in FACE_TYPE.items():
        index = _attribute_index(face_names, name)
        if face_attributes.shape[1] <= index:
            raise ValueError("Face attributes do not include {0}".format(name))
        face_codes[face_attributes[:, index] > 0.5] = code

    edge_grid_raw = _finite_array(extraction.get("graph_edge_grid"), "graph_edge_grid")
    edge_attributes = _finite_array(extraction.get("graph_edge_attr"), "graph_edge_attr")
    if edge_grid_raw.shape != (edge_count, 12, GRID_SIZE):
        raise ValueError(
            "Edge grid must have shape {0}; got {1}".format(
                (edge_count, 12, GRID_SIZE), tuple(edge_grid_raw.shape)
            )
        )
    if edge_attributes.ndim != 2 or edge_attributes.shape[0] != edge_count:
        raise ValueError("Edge attributes do not match the edge count")
    edge_names = schema["edge_attributes"]
    edge_codes = np.zeros(edge_count, dtype=np.int32)
    for name, code in EDGE_TYPE.items():
        index = _attribute_index(edge_names, name)
        if edge_attributes.shape[1] <= index:
            raise ValueError("Edge attributes do not include {0}".format(name))
        edge_codes[edge_attributes[:, index] > 0.5] = code
    convexity = np.zeros(edge_count, dtype=np.int32)
    for name, code in (("Concave edge", 1), ("Convex edge", 2)):
        index = _attribute_index(edge_names, name)
        if edge_attributes.shape[1] <= index:
            raise ValueError("Edge attributes do not include {0}".format(name))
        convexity[edge_attributes[:, index] > 0.5] = code
    length_index = _attribute_index(edge_names, "EdgeLengthAttribute")
    if edge_attributes.shape[1] <= length_index:
        raise ValueError("Edge attributes do not include EdgeLengthAttribute")
    signed_angles = _signed_edge_angles(edge_grid_raw, convexity)
    edge_grid = np.concatenate(
        [edge_grid_raw[:, :6, :], np.repeat(signed_angles[:, None, None], GRID_SIZE, axis=2)],
        axis=1,
    ).transpose(0, 2, 1).astype(np.float32, copy=False)

    graph.ndata["x"] = torch.from_numpy(node_grid).to(torch.float32)
    graph.ndata["y"] = torch.from_numpy(face_attributes[:, area_index]).to(torch.float32)
    graph.ndata["z"] = torch.from_numpy(face_codes).to(torch.int32)
    graph.ndata["l"] = torch.from_numpy(
        _face_loop_counts(extraction, node_count, topology_utils)
    ).to(torch.int32)
    graph.ndata["a"] = graph.in_degrees().to(torch.int32)
    graph.ndata["f"] = torch.from_numpy(labels.semantic).to(torch.int32)
    graph.edata["x"] = torch.from_numpy(edge_grid).to(torch.float32)
    graph.edata["t"] = torch.from_numpy(edge_codes).to(torch.int32)
    graph.edata["l"] = torch.from_numpy(edge_attributes[:, length_index]).to(torch.float32)
    graph.edata["a"] = torch.from_numpy(signed_angles).to(torch.float32)
    graph.edata["c"] = torch.from_numpy(convexity).to(torch.int32)
    return graph


def rbf_encode(values: np.ndarray, bins: int = RBF_BINS) -> np.ndarray:
    """Encode normalized pairwise values with Gaussian radial bases."""
    if bins < 2:
        raise ValueError("RBF encoding requires at least two bins")
    values = np.clip(np.asarray(values, dtype=np.float32), 0.0, 1.0)
    centers = np.linspace(0.0, 1.0, bins, dtype=np.float32)
    width = float(centers[1] - centers[0])
    return np.exp(-0.5 * ((values[..., None] - centers) / width) ** 2).astype(
        np.float32
    )


def _shortest_paths(graph) -> Tuple[np.ndarray, np.ndarray]:
    """Return undirected hops and predecessors while retaining DGL edge ids."""
    count = graph.num_nodes()
    distance = np.full((count, count), UNREACHABLE_DISTANCE, dtype=np.int32)
    predecessors = np.full((count, count), -1, dtype=np.int64)
    adjacency = [[] for _ in range(count)]
    source, destination = graph.edges()
    for edge_id, (start, end) in enumerate(zip(source.tolist(), destination.tolist())):
        adjacency[int(start)].append(int(end))
        adjacency[int(end)].append(int(start))
    for start in range(count):
        distance[start, start] = 0
        queue = deque([start])
        while queue:
            current = queue.popleft()
            for neighbour in adjacency[current]:
                if distance[start, neighbour] != UNREACHABLE_DISTANCE:
                    continue
                distance[start, neighbour] = distance[start, current] + 1
                predecessors[start, neighbour] = current
                queue.append(neighbour)
    return distance, predecessors


def shortest_path_metadata(graph, node_grid: np.ndarray) -> Dict[str, torch.Tensor]:
    """Build spatial_pos, edges_path, d2_distance, and angle_distance."""
    node_grid = np.asarray(node_grid, dtype=np.float32)
    count = graph.num_nodes()
    if node_grid.shape != (count, GRID_SIZE, GRID_SIZE, FEATURE_CHANNELS):
        raise ValueError("node_grid has an unexpected shape: {0}".format(tuple(node_grid.shape)))
    spatial_pos, predecessors = _shortest_paths(graph)
    edges_path = np.full((count, count, MAX_HOPS), -1, dtype=np.int64)
    source, destination = graph.edges()
    edge_ids = {(int(u), int(v)): index for index, (u, v) in enumerate(zip(source.tolist(), destination.tolist()))}
    for start in range(count):
        for end in range(count):
            if start == end or spatial_pos[start, end] >= UNREACHABLE_DISTANCE:
                continue
            vertices = [end]
            current = end
            while current != start:
                current = int(predecessors[start, current])
                if current < 0:
                    raise ValueError("Shortest-path predecessor is missing")
                vertices.append(current)
            vertices.reverse()
            for hop, (left, right) in enumerate(zip(vertices, vertices[1:])):
                if hop >= MAX_HOPS:
                    break
                edge_id = edge_ids.get((left, right))
                if edge_id is None:
                    edge_id = edge_ids.get((right, left))
                if edge_id is None:
                    raise ValueError("Shortest path has no corresponding DGL edge")
                edges_path[start, end, hop] = edge_id

    centers = node_grid[:, GRID_SIZE // 2, GRID_SIZE // 2, :3]
    normals = node_grid[:, GRID_SIZE // 2, GRID_SIZE // 2, 3:6]
    pair_distance = np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=-1)
    maximum_distance = float(pair_distance.max())
    if maximum_distance > 0.0:
        pair_distance = pair_distance / maximum_distance
    normal_lengths = np.linalg.norm(normals, axis=-1, keepdims=True)
    unit_normals = np.divide(
        normals,
        normal_lengths,
        out=np.zeros_like(normals),
        where=normal_lengths > 0.0,
    )
    dots = np.clip(np.matmul(unit_normals, unit_normals.T), -1.0, 1.0)
    pair_angle = np.arccos(dots).astype(np.float32) / math.pi
    return {
        "spatial_pos": torch.from_numpy(spatial_pos).to(torch.int32),
        "edges_path": torch.from_numpy(edges_path).to(torch.int64),
        "d2_distance": torch.from_numpy(rbf_encode(pair_distance)).to(torch.float32),
        "angle_distance": torch.from_numpy(rbf_encode(pair_angle)).to(torch.float32),
    }


def validate_bin(path: Path, expected_nodes: int) -> None:
    """Reload one BIN and validate required keys and exact leading shapes."""
    graphs, metadata = load_graphs(str(path))
    if len(graphs) != 1:
        raise ValueError("BIN must contain exactly one graph: {0}".format(path))
    graph = graphs[0]
    if graph.num_nodes() != expected_nodes:
        raise ValueError("BIN node count differs from label count: {0}".format(path))
    if not REQUIRED_NODE_KEYS.issubset(graph.ndata):
        raise ValueError("BIN is missing node fields: {0}".format(path))
    if not REQUIRED_EDGE_KEYS.issubset(graph.edata):
        raise ValueError("BIN is missing edge fields: {0}".format(path))
    if not REQUIRED_METADATA_KEYS.issubset(metadata):
        raise ValueError("BIN is missing metadata fields: {0}".format(path))
    count, edge_count = graph.num_nodes(), graph.num_edges()
    shapes = {
        "node_x": tuple(graph.ndata["x"].shape),
        "node_y": tuple(graph.ndata["y"].shape),
        "node_z": tuple(graph.ndata["z"].shape),
        "node_l": tuple(graph.ndata["l"].shape),
        "node_a": tuple(graph.ndata["a"].shape),
        "node_f": tuple(graph.ndata["f"].shape),
        "edge_x": tuple(graph.edata["x"].shape),
        "edge_t": tuple(graph.edata["t"].shape),
        "edge_l": tuple(graph.edata["l"].shape),
        "edge_a": tuple(graph.edata["a"].shape),
        "edge_c": tuple(graph.edata["c"].shape),
        "spatial_pos": tuple(metadata["spatial_pos"].shape),
        "edges_path": tuple(metadata["edges_path"].shape),
        "d2_distance": tuple(metadata["d2_distance"].shape),
        "angle_distance": tuple(metadata["angle_distance"].shape),
        "instance": tuple(metadata["instance"].shape),
        "bottom": tuple(metadata["bottom"].shape),
    }
    expected = {
        "node_x": (count, GRID_SIZE, GRID_SIZE, FEATURE_CHANNELS),
        "node_y": (count,),
        "node_z": (count,),
        "node_l": (count,),
        "node_a": (count,),
        "node_f": (count,),
        "edge_x": (edge_count, GRID_SIZE, FEATURE_CHANNELS),
        "edge_t": (edge_count,),
        "edge_l": (edge_count,),
        "edge_a": (edge_count,),
        "edge_c": (edge_count,),
        "spatial_pos": (count, count),
        "edges_path": (count, count, MAX_HOPS),
        "d2_distance": (count, count, RBF_BINS),
        "angle_distance": (count, count, RBF_BINS),
        "instance": (count, count),
        "bottom": (count,),
    }
    if shapes != expected:
        raise ValueError("BIN shapes differ from the contract: {0}".format(shapes))
    expected_dtypes = {
        "node_x": torch.float32,
        "node_y": torch.float32,
        "node_z": torch.int32,
        "node_l": torch.int32,
        "node_a": torch.int32,
        "node_f": torch.int32,
        "edge_x": torch.float32,
        "edge_t": torch.int32,
        "edge_l": torch.float32,
        "edge_a": torch.float32,
        "edge_c": torch.int32,
        "spatial_pos": torch.int32,
        "edges_path": torch.int64,
        "d2_distance": torch.float32,
        "angle_distance": torch.float32,
        "instance": torch.int8,
        "bottom": torch.int32,
    }
    actual_tensors = {
        "node_x": graph.ndata["x"],
        "node_y": graph.ndata["y"],
        "node_z": graph.ndata["z"],
        "node_l": graph.ndata["l"],
        "node_a": graph.ndata["a"],
        "node_f": graph.ndata["f"],
        "edge_x": graph.edata["x"],
        "edge_t": graph.edata["t"],
        "edge_l": graph.edata["l"],
        "edge_a": graph.edata["a"],
        "edge_c": graph.edata["c"],
        "spatial_pos": metadata["spatial_pos"],
        "edges_path": metadata["edges_path"],
        "d2_distance": metadata["d2_distance"],
        "angle_distance": metadata["angle_distance"],
        "instance": metadata["instance"],
        "bottom": metadata["bottom"],
    }
    for key, tensor in actual_tensors.items():
        if tensor.dtype != expected_dtypes[key]:
            raise ValueError("BIN dtype for {0} is invalid".format(key))
        if tensor.is_floating_point() and not torch.isfinite(tensor).all():
            raise ValueError("BIN contains non-finite values in {0}".format(key))


def convert_one(
    sample_id: str,
    paths: ConversionPaths,
    extractor_class: object,
    topology_utils: object,
    schema: dict,
    overwrite: bool,
) -> str:
    """Return ``converted`` or ``skipped``; raise on invalid input/output."""
    label = load_label(paths.label_file(sample_id))
    if label.sample_id != sample_id:
        raise ValueError(
            "Label sample id {0} does not match manifest id {1}".format(
                label.sample_id, sample_id
            )
        )
    target = paths.bin_file(sample_id)
    temporary = target.with_suffix(target.suffix + ".tmp")
    if target.is_file() and not overwrite:
        try:
            validate_bin(target, len(label.semantic))
            return "skipped"
        except (OSError, ValueError):
            pass
    step_file = paths.step_file(sample_id)
    if not step_file.is_file():
        raise FileNotFoundError("STEP file not found: {0}".format(step_file))
    paths.bins_root.mkdir(parents=True, exist_ok=True)
    try:
        extractor = extractor_class(step_file=step_file, attribute_schema=schema)
        extraction = extractor.process()
        graph = build_graph(extraction, label, schema, topology_utils)
        metadata = shortest_path_metadata(graph, graph.ndata["x"].cpu().numpy())
        metadata["instance"] = torch.from_numpy(label.instance).to(torch.int8)
        metadata["bottom"] = torch.from_numpy(label.bottom).to(torch.int32)
        save_graphs(str(temporary), [graph], metadata)
        validate_bin(temporary, len(label.semantic))
        temporary.replace(target)
        return "converted"
    finally:
        if temporary.exists():
            temporary.unlink()


def _read_ids(path: Path) -> List[str]:
    if not path.is_file():
        raise FileNotFoundError("Split manifest not found: {0}".format(path))
    seen = set()
    sample_ids = []
    for line in path.read_text(encoding="utf-8").splitlines():
        sample_id = line.strip()
        if sample_id and sample_id not in seen:
            seen.add(sample_id)
            sample_ids.append(sample_id)
    return sample_ids


def _reject_source_output(paths: ConversionPaths) -> None:
    output = paths.output_root.resolve()
    source = paths.dataset_root.resolve()
    subset = paths.subset_root.resolve()
    if output == source or output == subset:
        raise ValueError("Output root must differ from dataset and subset roots")


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    """Parse the sampled-subset conversion CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path(r"D:\Code\Datasets\MFRInstSeg"))
    parser.add_argument(
        "--subset-root",
        type=Path,
        default=Path(r"D:\Code\Datasets\MFRInstSeg\subsets\train_val_balanced"),
    )
    parser.add_argument("--output-root", type=Path, default=Path(r"D:\Code\Datasets\MFRInstSeg_BrepMFR"))
    parser.add_argument(
        "--extractor-file",
        type=Path,
        default=Path(r"D:\Code\BrepMFR-g\data_set_line\AAGExtractor.py"),
    )
    parser.add_argument(
        "--schema-file",
        type=Path,
        default=Path(r"D:\Code\BrepMFR-g\data_set_line\schema.json"),
    )
    parser.add_argument("--split", choices=("train", "val", "all"), default="all")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args(argv)


def main(argv: Optional[Iterable[str]] = None) -> int:
    """Convert selected manifests and write successful output manifests."""
    args = parse_args(argv)
    if args.limit is not None and args.limit < 1:
        print("Error: --limit must be positive", file=sys.stderr)
        return 2
    paths = ConversionPaths(args.dataset_root, args.subset_root, args.output_root)
    try:
        _reject_source_output(paths)
        schema = load_schema(args.schema_file)
        extractor_class, topology_utils = load_extractor(args.extractor_file)
        splits = ("train", "val") if args.split == "all" else (args.split,)
        failures = []
        for split in splits:
            sample_ids = _read_ids(paths.subset_root / (split + ".txt"))
            if args.limit is not None:
                sample_ids = sample_ids[: args.limit]
            successful = []
            for sample_id in sample_ids:
                try:
                    convert_one(
                        sample_id, paths, extractor_class, topology_utils, schema, args.overwrite
                    )
                    successful.append(sample_id)
                except Exception as error:
                    temporary = paths.bin_file(sample_id).with_suffix(".bin.tmp")
                    if temporary.exists():
                        temporary.unlink()
                    failures.append(
                        {
                            "split": split,
                            "sample_id": sample_id,
                            "exception_type": type(error).__name__,
                            "message": str(error),
                        }
                    )
            (paths.output_root / (split + ".txt")).parent.mkdir(parents=True, exist_ok=True)
            (paths.output_root / (split + ".txt")).write_text(
                "\n".join(successful) + ("\n" if successful else ""), encoding="utf-8"
            )
        (paths.output_root / "conversion_failures.json").write_text(
            json.dumps(failures, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        return 1 if failures else 0
    except (FileNotFoundError, ImportError, OSError, ValueError) as error:
        print("Error: {0}".format(error), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
