import numpy as np
import dgl
import torch

from brepmfr.inference import conversion as inference_conversion
from brepmfr.inference.postprocess import GeometryRequirements
from tools import mfrinstseg_to_bin as converter
from tools.geometry_features import PointSamplingConfig
from tools.mfrinstseg_to_bin import (
    EDGE_TYPE,
    FACE_TYPE,
    LabelData,
    build_graph,
)


def _two_face_contract():
    face_names = list(FACE_TYPE) + ["FaceAreaAttribute"]
    edge_names = list(EDGE_TYPE) + [
        "Concave edge",
        "Convex edge",
        "Smooth",
        "EdgeLengthAttribute",
    ]
    extraction = {
        "graph": {"edges": ([0], [1]), "num_nodes": 2},
        "graph_face_grid": np.zeros((2, 7, 5, 5), dtype=np.float32),
        "graph_face_attr": np.zeros((2, len(face_names)), dtype=np.float32),
        "graph_edge_grid": np.zeros((1, 12, 5), dtype=np.float32),
        "graph_edge_attr": np.zeros((1, len(edge_names)), dtype=np.float32),
    }
    extraction["graph_face_attr"][
        :, face_names.index("FaceAreaAttribute")
    ] = 1.0
    extraction["graph_edge_attr"][
        :, edge_names.index("EdgeLengthAttribute")
    ] = 1.0
    schema = {
        "face_attributes": face_names,
        "edge_attributes": edge_names,
    }
    return extraction, schema


def _patch_topology_helpers(monkeypatch):
    monkeypatch.setattr(
        converter,
        "_signed_edge_angles",
        lambda grid, convexity: np.zeros(1, dtype=np.float32),
    )
    monkeypatch.setattr(
        converter,
        "_face_loop_counts",
        lambda extraction, count, topology: np.ones(count, dtype=np.int32),
    )


def test_build_graph_without_labels_uses_unlabeled_sentinel(monkeypatch):
    extraction, schema = _two_face_contract()
    _patch_topology_helpers(monkeypatch)

    graph = build_graph(extraction, None, schema, object())

    assert graph.ndata["f"].dtype == torch.int32
    assert graph.ndata["f"].tolist() == [-1, -1]


def test_build_graph_with_labels_preserves_training_contract(monkeypatch):
    extraction, schema = _two_face_contract()
    _patch_topology_helpers(monkeypatch)
    labels = LabelData(
        sample_id="sample",
        semantic=np.asarray([1, 24], dtype=np.int32),
        instance=np.eye(2, dtype=np.int8),
    )

    graph = build_graph(extraction, labels, schema, object())

    assert graph.ndata["f"].tolist() == [1, 24]


def test_conversion_options_follow_checkpoint_geometry_requirements():
    options = inference_conversion.conversion_options_from_requirements(
        GeometryRequirements(False, True, True, True), seed=42
    )

    assert options.seed == 42
    assert options.include_face_pointcloud is True
    assert options.include_parametric_surface is True
    assert options.include_cylindrical_pos is True


def test_default_conversion_implementation_paths_cover_contract_sources():
    paths = inference_conversion.default_conversion_implementation_paths()

    assert all(path.is_file() for path in paths)
    assert [path.name for path in paths] == [
        "conversion.py",
        "AAGExtractor.py",
        "schema.json",
        "mfrinstseg_to_bin.py",
        "geometry_features.py",
    ]


def test_validate_inference_bin_maps_requirements(monkeypatch, tmp_path):
    calls = []
    point_sampling = PointSamplingConfig()
    requirements = GeometryRequirements(False, True, True, True)
    monkeypatch.setattr(
        inference_conversion,
        "validate_bin",
        lambda path, expected_nodes, requirements, point_sampling: calls.append(
            (path, expected_nodes, requirements, point_sampling)
        ),
    )

    inference_conversion.validate_inference_bin(
        tmp_path / "input.bin",
        requirements,
        point_sampling,
        expected_nodes=7,
    )

    path, expected_nodes, bin_requirements, actual_sampling = calls[0]
    assert path == tmp_path / "input.bin"
    assert expected_nodes == 7
    assert bin_requirements.require_instance is False
    assert bin_requirements.require_pointcloud is True
    assert bin_requirements.require_parametric_surface is True
    assert bin_requirements.require_cylindrical is True
    assert actual_sampling is point_sampling


def test_conversion_builds_unlabeled_bin_without_instance_truth(
    tmp_path, monkeypatch
):
    step_path = tmp_path / "part.step"
    target_path = tmp_path / "output" / "input.bin"
    step_path.write_text(
        "ISO-10303-21;END-ISO-10303-21;", encoding="ascii"
    )
    graph = dgl.graph(([0], [1]), num_nodes=2)
    graph.ndata["x"] = torch.zeros((2, 5, 5, 7), dtype=torch.float32)

    class FakeExtractor:
        def __init__(self, step_file, attribute_schema):
            self.step_file = step_file
            self.attribute_schema = attribute_schema

        def process(self):
            return {}

    labels_seen = []
    saved_metadata = []
    geometry_calls = []
    validation_calls = []

    monkeypatch.setattr(
        converter,
        "load_label",
        lambda path: (_ for _ in ()).throw(AssertionError("labels were read")),
    )
    monkeypatch.setattr(
        inference_conversion,
        "load_extractor",
        lambda path: (FakeExtractor, object()),
    )
    monkeypatch.setattr(inference_conversion, "load_schema", lambda path: {})
    monkeypatch.setattr(
        inference_conversion,
        "build_graph",
        lambda extraction, labels, schema, topology: (
            labels_seen.append(labels) or graph
        ),
    )
    monkeypatch.setattr(
        inference_conversion,
        "shortest_path_metadata",
        lambda graph, node_grid: {"paths": torch.tensor([1])},
    )

    def fake_geometry(*args, **kwargs):
        geometry_calls.append((args, kwargs))
        return object()

    monkeypatch.setattr(
        inference_conversion, "extract_geometry_features", fake_geometry
    )
    monkeypatch.setattr(
        inference_conversion, "attach_geometry_features", lambda *args: None
    )
    monkeypatch.setattr(
        inference_conversion,
        "geometry_metadata",
        lambda point_sampling: {"points": torch.tensor([1])},
    )
    monkeypatch.setattr(
        inference_conversion,
        "surface_geometry_metadata",
        lambda: {"surface": torch.tensor([1])},
    )

    def fake_save(path, graphs, metadata):
        path = type(target_path)(path)
        assert path != target_path
        assert target_path.exists() is False
        path.write_bytes(b"test-bin")
        saved_metadata.append(dict(metadata))

    monkeypatch.setattr(inference_conversion, "save_graphs", fake_save)

    def fake_validate(path, requirements, point_sampling, expected_nodes=None):
        assert path.exists()
        assert target_path.exists() is False
        validation_calls.append((path, requirements, point_sampling, expected_nodes))

    monkeypatch.setattr(
        inference_conversion, "validate_inference_bin", fake_validate
    )

    count = inference_conversion.convert_step_to_inference_bin(
        step_path,
        target_path,
        GeometryRequirements(False, True, True, True),
        seed=42,
    )

    assert labels_seen == [None]
    assert "instance" not in saved_metadata[0]
    assert set(saved_metadata[0]) == {"paths", "points", "surface"}
    assert geometry_calls[0][1] == {
        "include_face_pointcloud": True,
        "include_cylindrical_pos": True,
        "include_parametric_surface": True,
    }
    assert validation_calls[0][3] == 2
    assert count == 2
    assert target_path.read_bytes() == b"test-bin"
    assert not target_path.with_suffix(".bin.tmp").exists()
