import numpy as np
import torch

from tools import mfrinstseg_to_bin as converter
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

