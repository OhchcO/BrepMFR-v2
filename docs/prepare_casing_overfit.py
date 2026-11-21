"""Build a one-parent casing memorization dataset from reviewed STEP labels.

The train/val/test manifests deliberately name the same child graphs. This is
only a pipeline and memorization check, never a held-out evaluation split.
"""

from __future__ import annotations

import argparse
from collections import Counter, deque
import hashlib
import json
from pathlib import Path
import sys

import dgl
import numpy as np
import torch
from dgl.data.utils import save_graphs


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.geometry_features import PointSamplingConfig, extract_geometry_features
from tools.mfrinstseg_to_bin import (
    BinRequirements,
    ConversionOptions,
    LabelData,
    attach_geometry_features,
    build_graph,
    geometry_metadata,
    load_extractor,
    load_schema,
    shortest_path_metadata,
    surface_geometry_metadata,
    validate_bin,
)


SEMANTIC_CODES = (
    0, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110,
    201, 202, 203, 204, 301, 302, 303, 304, 305, 306,
    401, 402, 403, 404, 501, 502, 503, 601, 602,
)
CODE_TO_INDEX = {code: index for index, code in enumerate(SEMANTIC_CODES)}


def map_semantic_codes(codes):
    """Map paper semantic codes to the network's fixed 0..29 class indices."""
    unknown = sorted(set(codes) - set(CODE_TO_INDEX))
    if unknown:
        raise ValueError("unknown semantic codes: {}".format(unknown))
    return np.asarray([CODE_TO_INDEX[code] for code in codes], dtype=np.int32)


def build_instance_matrix(instance_ids):
    instance_ids = np.asarray(instance_ids, dtype=np.int32)
    if instance_ids.ndim != 1:
        raise ValueError("instance_ids must be a vector")
    valid = instance_ids > 0
    return ((instance_ids[:, None] == instance_ids[None, :])
            & valid[:, None] & valid[None, :]).astype(np.int8)


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_labels(path, step_name):
    with Path(path).open(encoding="utf-8") as stream:
        document = json.load(stream)
    if document.get("source_file") != step_name:
        raise ValueError("label source_file must exactly match STEP filename")
    count = document.get("num_faces")
    codes = document.get("labels")
    instance_ids = document.get("instance_ids")
    if not isinstance(count, int) or count < 1:
        raise ValueError("num_faces must be positive")
    if not isinstance(codes, list) or not isinstance(instance_ids, list):
        raise ValueError("labels and instance_ids must be arrays")
    if len(codes) != count or len(instance_ids) != count:
        raise ValueError("label array lengths differ from num_faces")
    if any(type(value) is not int for value in codes + instance_ids):
        raise ValueError("labels and instance_ids must contain integers")
    mapped = map_semantic_codes(codes)
    semantic_by_instance = {}
    for code, instance in zip(codes, instance_ids):
        if (code == 0 and instance != -1) or (code > 0 and instance <= 0):
            raise ValueError("semantic and instance labels disagree")
        if instance > 0:
            previous = semantic_by_instance.setdefault(instance, code)
            if previous != code:
                raise ValueError("one instance crosses semantic classes")
    return codes, mapped, np.asarray(instance_ids, dtype=np.int32)


def _verify_scaled_face_order(raw, scaled, schema):
    """Check per-index areas and centroids under one common affine scaling."""
    raw_attr = np.asarray(raw["graph_face_attr"], dtype=np.float64)
    scaled_attr = np.asarray(scaled["graph_face_attr"], dtype=np.float64)
    if raw_attr.shape != scaled_attr.shape:
        raise ValueError("raw and normalized AAGs have different face counts")
    names = schema["face_attributes"]
    area_index = names.index("FaceAreaAttribute")
    center_index = names.index("FaceCentroidAttribute")
    raw_area = raw_attr[:, area_index]
    scaled_area = scaled_attr[:, area_index]
    if np.any(raw_area <= 0) or np.any(scaled_area <= 0):
        raise ValueError("nonpositive face area prevents index verification")
    ratio = scaled_area / raw_area
    area_scale = float(np.median(ratio))
    if not np.allclose(ratio, area_scale, rtol=1e-5, atol=1e-10):
        raise ValueError("face order changed during AAG normalization")
    linear_scale = float(np.sqrt(area_scale))
    raw_centers = raw_attr[:, center_index:center_index + 3]
    scaled_centers = scaled_attr[:, center_index:center_index + 3]
    translation = np.median(scaled_centers - raw_centers * linear_scale, axis=0)
    if not np.allclose(
        scaled_centers, raw_centers * linear_scale + translation,
        rtol=1e-5, atol=1e-4,
    ):
        raise ValueError("face centroids changed order during normalization")


def _crop_nodes(graph, seed, node_limit):
    neighbours = [set() for _ in range(graph.num_nodes())]
    sources, destinations = graph.edges()
    for source, destination in zip(sources.tolist(), destinations.tolist()):
        neighbours[source].add(destination)
        neighbours[destination].add(source)
    visited = {seed}
    queue = deque([seed])
    while queue and len(visited) < node_limit:
        for neighbour in sorted(neighbours[queue.popleft()]):
            if neighbour not in visited:
                visited.add(neighbour)
                queue.append(neighbour)
                if len(visited) == node_limit:
                    break
    return sorted(visited)


def plan_covering_crops(graph, codes, crop_size):
    """Seed each observed class, then add crops until every parent face appears."""
    if len(codes) != graph.num_nodes():
        raise ValueError("codes do not align with graph nodes")
    planned = []
    covered = set()
    for code in sorted(set(codes)):
        nodes = _crop_nodes(graph, codes.index(code), crop_size)
        planned.append(("class_{:03d}".format(code), nodes))
        covered.update(nodes)

    neighbours = [set() for _ in range(graph.num_nodes())]
    sources, destinations = graph.edges()
    for source, destination in zip(sources.tolist(), destinations.tolist()):
        neighbours[source].add(destination)
        neighbours[destination].add(source)
    target_capacity = max(1, crop_size - max(1, crop_size // 4))
    additional_index = 0
    while len(covered) < graph.num_nodes():
        targets = [index for index in range(graph.num_nodes())
                   if index not in covered][:target_capacity]
        selected = set(targets)
        queue = deque(targets)
        while queue and len(selected) < crop_size:
            for neighbour in sorted(neighbours[queue.popleft()]):
                if neighbour not in selected:
                    selected.add(neighbour)
                    queue.append(neighbour)
                    if len(selected) == crop_size:
                        break
        nodes = sorted(selected)
        planned.append(("cover_{:03d}".format(additional_index), nodes))
        covered.update(nodes)
        additional_index += 1
    return planned


def _save_validated(path, graph, metadata, point_sampling):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".bin.tmp")
    requirements = BinRequirements(
        require_instance=True,
        require_pointcloud=True,
        require_parametric_surface=True,
        require_cylindrical=True,
    )
    try:
        save_graphs(str(temporary), [graph], metadata)
        validate_bin(
            temporary, graph.num_nodes(), requirements=requirements,
            point_sampling=point_sampling,
        )
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def build_dataset(step, labels, output, crop_size=96, seed=42):
    step, labels, output = Path(step), Path(labels), Path(output)
    if not step.is_file() or not labels.is_file():
        raise FileNotFoundError("STEP or label JSON does not exist")
    if output.resolve() in (step.resolve().parent, labels.resolve().parent):
        raise ValueError("output must be separate from source directories")
    if crop_size < 2:
        raise ValueError("crop_size must be at least 2")
    codes, mapped, instances = _load_labels(labels, step.name)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("output directory must be new or empty")

    extractor_class, topology_utils = load_extractor(
        PROJECT_ROOT / "tools" / "AAGExtractor.py"
    )
    schema = load_schema(PROJECT_ROOT / "tools" / "schema.json")
    raw_extractor = extractor_class(
        step_file=step, attribute_schema=schema, scale_body=False
    )
    raw = raw_extractor.process()
    scaled_extractor = extractor_class(
        step_file=step, attribute_schema=schema, scale_body=True
    )
    extraction = scaled_extractor.process()
    if raw["graph"]["num_nodes"] != len(codes):
        raise ValueError("STEP and JSON have different face counts")
    _verify_scaled_face_order(raw, extraction, schema)
    extraction["_extractor"] = scaled_extractor

    relation = build_instance_matrix(instances)
    labels_data = LabelData(step.stem, mapped, relation)
    graph = build_graph(extraction, labels_data, schema, topology_utils)
    graph.ndata["i"] = torch.from_numpy(instances)
    point_sampling = PointSamplingConfig(mode="fixed", fixed_count=64)
    options = ConversionOptions(
        seed=seed, point_sampling=point_sampling,
        include_face_pointcloud=True, include_parametric_surface=True,
        include_cylindrical_pos=True,
    )
    geometry = extract_geometry_features(
        extraction, point_sampling, seed, step.stem,
        include_face_pointcloud=True,
        include_cylindrical_pos=True,
        include_parametric_surface=True,
    )
    attach_geometry_features(graph, geometry, options)
    metadata = shortest_path_metadata(graph, graph.ndata["x"].cpu().numpy())
    metadata["instance"] = torch.from_numpy(relation)
    metadata.update(geometry_metadata(point_sampling))
    metadata.update(surface_geometry_metadata())
    parent_path = output / "bin" / (step.stem + "_parent.bin")
    _save_validated(parent_path, graph, metadata, point_sampling)

    child_names = []
    coverage = set()
    class_counts = Counter(codes)
    for suffix, node_ids in plan_covering_crops(graph, codes, crop_size):
        child = dgl.node_subgraph(graph, node_ids)
        source_ids = child.ndata[dgl.NID].to(torch.int64)
        child_instances = child.ndata["i"].cpu().numpy()
        child_metadata = shortest_path_metadata(child, child.ndata["x"].cpu().numpy())
        child_metadata["instance"] = torch.from_numpy(
            build_instance_matrix(child_instances)
        )
        child_metadata["parent_face_index"] = source_ids
        child_metadata.update(geometry_metadata(point_sampling))
        child_metadata.update(surface_geometry_metadata())
        sample_id = "{}_{}".format(step.stem, suffix)
        _save_validated(output / "bin" / (sample_id + ".bin"),
                        child, child_metadata, point_sampling)
        child_names.append(sample_id)
        coverage.update(node_ids)

    output.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        (output / (split + ".txt")).write_text(
            "\n".join(child_names) + "\n", encoding="utf-8"
        )
    summary = {
        "purpose": "one-parent memorization and pipeline check; no held-out data",
        "step": str(step.resolve()),
        "labels": str(labels.resolve()),
        "step_sha256": _sha256(step),
        "labels_sha256": _sha256(labels),
        "parent_faces": len(codes),
        "parent_instances": len(set(value for value in instances if value > 0)),
        "semantic_face_counts": {str(code): count for code, count in sorted(class_counts.items())},
        "index_to_semantic_code": list(SEMANTIC_CODES),
        "child_graphs": child_names,
        "crop_size_limit": crop_size,
        "covered_parent_faces": len(coverage),
        "train_val_test_are_identical": True,
        "seed": seed,
    }
    (output / "dataset_manifest.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--step", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--crop-size", type=int, default=96)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    print(json.dumps(build_dataset(
        args.step, args.labels, args.output, args.crop_size, args.seed,
    ), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
