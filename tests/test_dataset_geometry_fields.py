import dgl
import pytest
import torch
from dgl.data.utils import save_graphs
from scipy.spatial.transform import Rotation

import data.dataset as dataset_module
from data.dataset import CADSynth
from utils.reproducibility import derive_seed


def _core_graph(node_count=2):
    graph = dgl.graph(([0, 1], [1, 0]), num_nodes=node_count, idtype=torch.int64)
    graph.ndata["x"] = torch.zeros((node_count, 5, 5, 7), dtype=torch.float32)
    graph.ndata["y"] = torch.ones(node_count, dtype=torch.float32)
    graph.ndata["z"] = torch.zeros(node_count, dtype=torch.int32)
    graph.ndata["l"] = torch.ones(node_count, dtype=torch.int32)
    graph.ndata["a"] = torch.ones(node_count, dtype=torch.int32)
    graph.ndata["f"] = torch.arange(node_count, dtype=torch.int32)
    graph.edata["x"] = torch.zeros((2, 5, 7), dtype=torch.float32)
    graph.edata["t"] = torch.zeros(2, dtype=torch.int32)
    graph.edata["l"] = torch.ones(2, dtype=torch.float32)
    graph.edata["a"] = torch.zeros(2, dtype=torch.float32)
    graph.edata["c"] = torch.zeros(2, dtype=torch.int32)
    return graph


def _metadata(node_count=2):
    return {
        "edges_path": torch.full((node_count, node_count, 6), -1, dtype=torch.int64),
        "spatial_pos": torch.zeros((node_count, node_count), dtype=torch.int32),
        "d2_distance": torch.zeros((node_count, node_count, 64), dtype=torch.float32),
        "angle_distance": torch.zeros((node_count, node_count, 64), dtype=torch.float32),
    }


def _write_bin(root, sample_id, include_geometry):
    graph = _core_graph()
    if include_geometry:
        points = torch.zeros((2, 96, 6), dtype=torch.float32)
        points[:, :64, 0] = 1.0
        points[:, :64, 4] = 1.0
        graph.ndata["face_points"] = points
        graph.ndata["face_point_mask"] = (
            torch.arange(96)[None, :] < 64
        ).expand(2, -1).clone()
        graph.ndata["face_point_count"] = torch.full((2,), 64, dtype=torch.int32)
        graph.ndata["cylindrical_pos"] = torch.tensor(
            [[0.0, 0.0, 1.0, 0.0], [1.0, 1.0, 0.0, 1.0]],
            dtype=torch.float32,
        )
    path = root / "bin" / (sample_id + ".bin")
    path.parent.mkdir(parents=True, exist_ok=True)
    save_graphs(str(path), [graph], _metadata())
    return path


@pytest.fixture
def old_bin_dataset(tmp_path):
    sample_id = "legacy_part_001"
    _write_bin(tmp_path, sample_id, include_geometry=False)
    (tmp_path / "train.txt").write_text(sample_id + "\n", encoding="utf-8")
    return tmp_path


@pytest.fixture
def new_bin_dataset(tmp_path):
    sample_ids = ("modern_part_001", "modern_part_002")
    for sample_id in sample_ids:
        _write_bin(tmp_path, sample_id, include_geometry=True)
    (tmp_path / "train.txt").write_text("\n".join(sample_ids) + "\n", encoding="utf-8")
    return tmp_path


def test_old_bin_loads_when_geometry_is_disabled(old_bin_dataset):
    item = CADSynth(
        old_bin_dataset,
        "train",
        require_face_pointcloud=False,
        require_cylindrical_pos=False,
    )[0]

    assert item.face_points is None
    assert item.face_point_mask is None
    assert item.face_point_count is None
    assert item.cylindrical_pos is None
    assert item.data_id == "legacy_part_001"


def test_missing_point_fields_fail_when_enabled(old_bin_dataset):
    dataset = CADSynth(old_bin_dataset, "train", require_face_pointcloud=True)

    with pytest.raises(KeyError, match=r"face_points.*legacy_part_001"):
        dataset[0]


def test_missing_cylindrical_field_fails_when_enabled(old_bin_dataset):
    dataset = CADSynth(old_bin_dataset, "train", require_cylindrical_pos=True)

    with pytest.raises(KeyError, match=r"cylindrical_pos.*legacy_part_001"):
        dataset[0]


def test_new_fields_are_concatenated_by_collator(new_bin_dataset):
    dataset = CADSynth(
        new_bin_dataset,
        "train",
        require_face_pointcloud=True,
        require_cylindrical_pos=True,
    )
    batch = dataset._collate([dataset[0], dataset[1]])

    assert batch["face_points"].shape == (batch["node_data"].shape[0], 96, 6)
    assert batch["face_point_mask"].shape == (batch["node_data"].shape[0], 96)
    assert batch["face_point_count"].shape == (batch["node_data"].shape[0],)
    assert batch["cylindrical_pos"].shape == (batch["node_data"].shape[0], 4)
    assert batch["data_ids"] == ("modern_part_001", "modern_part_002")
    assert isinstance(batch["data_ids"], tuple)


def test_collator_rejects_mixed_optional_geometry_batch(
    old_bin_dataset, new_bin_dataset
):
    old_item = CADSynth(old_bin_dataset, "train")[0]
    new_item = CADSynth(
        new_bin_dataset,
        "train",
        require_face_pointcloud=True,
        require_cylindrical_pos=True,
    )[0]

    with pytest.raises(ValueError, match="face_points.*mixed"):
        CADSynth(new_bin_dataset, "train")._collate([old_item, new_item])


def test_random_rotation_rejects_cylindrical_positions(new_bin_dataset):
    with pytest.raises(ValueError, match="random_rotate.*cylindrical"):
        CADSynth(
            new_bin_dataset,
            "train",
            random_rotate=True,
            require_cylindrical_pos=True,
        )


def test_random_rotation_rotates_face_point_positions_and_normals(new_bin_dataset, monkeypatch):
    rotation = Rotation.from_euler("z", 90, degrees=True)
    monkeypatch.setattr(dataset_module, "get_random_rotation", lambda: rotation)
    dataset = CADSynth(
        new_bin_dataset,
        "train",
        random_rotate=True,
        require_face_pointcloud=True,
    )

    item = dataset[0]
    matrix = torch.tensor(rotation.as_matrix(), dtype=torch.float32)
    expected = torch.cat(
        (
            torch.tensor([1.0, 0.0, 0.0]).matmul(matrix),
            torch.tensor([0.0, 1.0, 0.0]).matmul(matrix),
        )
    )

    assert torch.allclose(item.face_points[0, 0], expected)


def test_loader_uses_seeded_generator_and_split_default_purpose(new_bin_dataset):
    dataset = CADSynth(new_bin_dataset, "train", seed=123)
    loader = dataset.get_dataloader(batch_size=1, shuffle=True)
    expected = torch.rand(
        3, generator=torch.Generator().manual_seed(derive_seed(123, "train_loader"))
    )
    actual = torch.rand(3, generator=loader.generator)

    assert torch.equal(actual, expected)
    assert loader.worker_init_fn.__name__ == "seed_worker"
