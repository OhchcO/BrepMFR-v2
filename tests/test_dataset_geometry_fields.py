import dgl
import pytest
import torch
from dgl.data.utils import load_graphs, save_graphs
from scipy.spatial.transform import Rotation

import data.dataset as dataset_module
from data.dataset import CADSynth
from utils.reproducibility import derive_seed


def _core_graph(node_count=2):
    graph = dgl.graph(([0, 1], [1, 0]), num_nodes=node_count, idtype=torch.int64)
    graph.ndata["x"] = torch.zeros((node_count, 5, 5, 7), dtype=torch.float32)
    graph.ndata["x"][..., 0] = 1.0
    graph.ndata["x"][..., 4] = 1.0
    graph.ndata["y"] = torch.ones(node_count, dtype=torch.float32)
    graph.ndata["z"] = torch.zeros(node_count, dtype=torch.int32)
    graph.ndata["l"] = torch.ones(node_count, dtype=torch.int32)
    graph.ndata["a"] = torch.ones(node_count, dtype=torch.int32)
    graph.ndata["f"] = torch.arange(node_count, dtype=torch.int32)
    graph.edata["x"] = torch.zeros((2, 5, 7), dtype=torch.float32)
    graph.edata["x"][..., 0] = 1.0
    graph.edata["x"][..., 4] = 1.0
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
        "point_feature_contract": torch.tensor([2, 10, 1, 2], dtype=torch.int16),
    }


def _write_bin(
    root,
    sample_id,
    include_points=False,
    include_cylindrical=False,
    include_parametric=False,
):
    graph = _core_graph()
    if include_points:
        points = torch.zeros((2, 96, 10), dtype=torch.float32)
        points[:, :64, 0] = 1.0
        points[:, :64, 4] = 1.0
        points[:, :64, 6] = torch.linspace(-1.0, 1.0, 64)
        points[:, :64, 7] = torch.linspace(1.0, -1.0, 64)
        graph.ndata["face_points"] = points
        graph.ndata["face_point_mask"] = (
            torch.arange(96)[None, :] < 64
        ).expand(2, -1).clone()
        graph.ndata["face_point_count"] = torch.full((2,), 64, dtype=torch.int32)
    if include_parametric:
        graph.ndata["face_surface_metric"] = torch.zeros(
            (2, 96, 3), dtype=torch.float32
        )
        graph.ndata["face_surface_metric"][:, :64, 0] = 1.0
        graph.ndata["face_surface_metric"][:, :64, 2] = 1.0
        graph.ndata["face_trim_neighbors"] = torch.full(
            (2, 96, 16), -1, dtype=torch.int16
        )
    if include_cylindrical:
        graph.ndata["cylindrical_pos"] = torch.tensor(
            [[0.0, 0.0, 1.0, 0.0], [1.0, 1.0, 0.0, 1.0]],
            dtype=torch.float32,
        )
    path = root / "bin" / (sample_id + ".bin")
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = _metadata()
    if include_parametric:
        metadata["surface_feature_contract"] = torch.tensor(
            [1, 3, 16], dtype=torch.int16
        )
    save_graphs(str(path), [graph], metadata)
    return path


def _write_manifest(root, split, sample_ids):
    (root / (split + ".txt")).write_text(
        "\n".join(sample_ids) + ("\n" if sample_ids else ""), encoding="utf-8"
    )


def _rewrite_bin(path, mutate):
    graphs, metadata = load_graphs(str(path))
    mutate(graphs[0])
    save_graphs(str(path), graphs, metadata)


@pytest.fixture
def old_bin_dataset(tmp_path):
    sample_id = "legacy_part_001"
    _write_bin(tmp_path, sample_id)
    _write_manifest(tmp_path, "train", (sample_id,))
    return tmp_path


@pytest.fixture
def new_bin_dataset(tmp_path):
    sample_ids = ("modern_part_001", "modern_part_002")
    for sample_id in sample_ids:
        _write_bin(tmp_path, sample_id, include_points=True, include_cylindrical=True)
    _write_manifest(tmp_path, "train", sample_ids)
    return tmp_path


@pytest.fixture
def parametric_bin_dataset(tmp_path):
    sample_ids = ("surface_part_001", "surface_part_002")
    for sample_id in sample_ids:
        _write_bin(
            tmp_path,
            sample_id,
            include_points=True,
            include_parametric=True,
        )
    _write_manifest(tmp_path, "train", sample_ids)
    return tmp_path


def test_old_bin_loads_when_geometry_is_disabled(old_bin_dataset):
    item = CADSynth(
        old_bin_dataset,
        "train",
        require_face_pointcloud=False,
        require_cylindrical_pos=False,
    )[0]

    assert getattr(item, "face_points", None) is None
    assert getattr(item, "face_point_mask", None) is None
    assert getattr(item, "face_point_count", None) is None
    assert getattr(item, "cylindrical_pos", None) is None
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

    assert batch["face_points"].shape == (batch["node_data"].shape[0], 96, 10)
    assert batch["face_point_mask"].shape == (batch["node_data"].shape[0], 96)
    assert batch["face_point_mask"].dtype == torch.bool
    assert batch["face_point_count"].shape == (batch["node_data"].shape[0],)
    assert batch["cylindrical_pos"].shape == (batch["node_data"].shape[0], 4)
    assert batch["data_ids"] == ("modern_part_001", "modern_part_002")
    assert isinstance(batch["data_ids"], tuple)


def test_parametric_surface_fields_are_validated_and_collated(
    parametric_bin_dataset,
):
    dataset = CADSynth(
        parametric_bin_dataset,
        "train",
        require_face_pointcloud=True,
        require_parametric_surface=True,
    )
    batch = dataset._collate([dataset[0], dataset[1]])
    assert batch["face_surface_metric"].shape == (4, 96, 3)
    assert batch["face_trim_neighbors"].shape == (4, 96, 16)
    assert batch["face_trim_neighbors"].dtype == torch.int16


def test_dataset_normalizes_dgl_roundtrip_uint8_mask_to_bool(new_bin_dataset):
    path = new_bin_dataset / "bin" / "modern_part_001.bin"
    serialized_graph = load_graphs(str(path))[0][0]
    assert serialized_graph.ndata["face_point_mask"].dtype == torch.uint8

    item = CADSynth(
        new_bin_dataset, "train", require_face_pointcloud=True
    )[0]

    assert item.face_point_mask.dtype == torch.bool
    assert set(item.face_point_mask.unique().tolist()) == {False, True}


def test_dataset_rejects_nonbinary_uint8_mask(new_bin_dataset):
    path = new_bin_dataset / "bin" / "modern_part_001.bin"

    def make_nonbinary(graph):
        assert graph.ndata["face_point_mask"].dtype == torch.uint8
        graph.ndata["face_point_mask"][0, 0] = 2

    _rewrite_bin(path, make_nonbinary)
    dataset = CADSynth(
        new_bin_dataset, "train", require_face_pointcloud=True
    )

    with pytest.raises(ValueError, match="face_point_mask.*binary"):
            dataset[0]


def test_dataset_rejects_wrong_point_feature_contract(new_bin_dataset):
    path = new_bin_dataset / "bin" / "modern_part_001.bin"
    graphs, metadata = load_graphs(str(path))
    metadata["point_feature_contract"] = torch.tensor(
        [1, 10, 2, 1], dtype=torch.int16
    )
    save_graphs(str(path), graphs, metadata)
    dataset = CADSynth(
        new_bin_dataset, "train", require_face_pointcloud=True
    )

    with pytest.raises(ValueError, match="point_feature_contract"):
        dataset[0]


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

    with pytest.raises(ValueError, match="face_points.*atomic"):
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

    assert torch.allclose(item.face_points[0, 0, :6], expected)
    assert torch.allclose(
        item.face_points[0, 0, 6:],
        torch.tensor([-1.0, 1.0, 0.0, 0.0]),
    )
    assert torch.allclose(item.node_data[0, 0, 0, :6], expected)
    assert torch.allclose(item.edge_data[0, 0, :6], expected)


def test_loader_uses_seeded_generator_and_split_default_purpose(new_bin_dataset):
    dataset = CADSynth(new_bin_dataset, "train", seed=123)
    loader = dataset.get_dataloader(batch_size=1, shuffle=True)
    expected = torch.rand(
        3, generator=torch.Generator().manual_seed(derive_seed(123, "train_loader"))
    )
    actual = torch.rand(3, generator=loader.generator)

    assert torch.equal(actual, expected)
    assert loader.worker_init_fn.__name__ == "seed_worker"


@pytest.mark.parametrize(
    ("mutate", "requirements", "field"),
    [
        (
            lambda graph: graph.ndata.__setitem__(
                "face_points", torch.zeros((2, 96, 5), dtype=torch.float32)
            ),
            {"require_face_pointcloud": True},
            "face_points",
        ),
        (
            lambda graph: graph.ndata.__setitem__(
                "face_points", graph.ndata["face_points"].to(torch.float64)
            ),
            {"require_face_pointcloud": True},
            "face_points",
        ),
        (
            lambda graph: graph.ndata["face_points"].__setitem__((0, 0, 0), float("nan")),
            {"require_face_pointcloud": True},
            "face_points",
        ),
        (
            lambda graph: graph.ndata.__setitem__(
                "face_point_mask", torch.zeros((2, 95), dtype=torch.bool)
            ),
            {"require_face_pointcloud": True},
            "face_point_mask",
        ),
        (
            lambda graph: graph.ndata.__setitem__(
                "face_point_mask", graph.ndata["face_point_mask"].to(torch.int8)
            ),
            {"require_face_pointcloud": True},
            "face_point_mask",
        ),
        (
            lambda graph: graph.ndata.__setitem__(
                "face_point_count", torch.zeros((2, 1), dtype=torch.int32)
            ),
            {"require_face_pointcloud": True},
            "face_point_count",
        ),
        (
            lambda graph: graph.ndata.__setitem__(
                "face_point_count", graph.ndata["face_point_count"].to(torch.int64)
            ),
            {"require_face_pointcloud": True},
            "face_point_count",
        ),
        (
            lambda graph: graph.ndata.__setitem__(
                "face_point_count", torch.zeros(2, dtype=torch.int32)
            ),
            {"require_face_pointcloud": True},
            "face_point_count",
        ),
        (
            lambda graph: graph.ndata["face_point_mask"].__setitem__((0, 0), False),
            {"require_face_pointcloud": True},
            "face_point_mask",
        ),
        (
            lambda graph: graph.ndata["face_points"].__setitem__((0, 64, 0), 1.0),
            {"require_face_pointcloud": True},
            "face_points",
        ),
        (
            lambda graph: graph.ndata["face_points"].__setitem__(
                (0, 0, slice(3, 6)), torch.tensor([0.0, 0.0, 0.5])
            ),
            {"require_face_pointcloud": True},
            "face_points",
        ),
        (
            lambda graph: graph.ndata.__setitem__(
                "cylindrical_pos", torch.zeros((2, 3), dtype=torch.float32)
            ),
            {"require_cylindrical_pos": True},
            "cylindrical_pos",
        ),
        (
            lambda graph: graph.ndata.__setitem__(
                "cylindrical_pos", graph.ndata["cylindrical_pos"].to(torch.float64)
            ),
            {"require_cylindrical_pos": True},
            "cylindrical_pos",
        ),
        (
            lambda graph: graph.ndata["cylindrical_pos"].__setitem__((0, 0), float("inf")),
            {"require_cylindrical_pos": True},
            "cylindrical_pos",
        ),
        (
            lambda graph: graph.ndata["cylindrical_pos"].__setitem__((0, 0), 1.1),
            {"require_cylindrical_pos": True},
            "cylindrical_pos",
        ),
        (
            lambda graph: graph.ndata["cylindrical_pos"].__setitem__((0, 3), -0.1),
            {"require_cylindrical_pos": True},
            "cylindrical_pos",
        ),
        (
            lambda graph: graph.ndata["cylindrical_pos"].__setitem__(
                (0, slice(1, 3)), torch.tensor([0.0, 0.0])
            ),
            {"require_cylindrical_pos": True},
            "cylindrical_pos",
        ),
    ],
)
def test_requested_geometry_rejects_invalid_contract_values(
    new_bin_dataset, mutate, requirements, field
):
    path = new_bin_dataset / "bin" / "modern_part_001.bin"
    _rewrite_bin(path, mutate)
    dataset = CADSynth(new_bin_dataset, "train", **requirements)

    with pytest.raises(ValueError, match=field) as error:
        dataset[0]

    assert str(path.resolve()) in str(error.value)


def test_disabled_geometry_normalizes_mixed_bin_schemas(tmp_path):
    _write_bin(tmp_path, "legacy")
    _write_bin(tmp_path, "geometry", include_points=True, include_cylindrical=True)
    _write_manifest(tmp_path, "train", ("legacy", "geometry"))
    dataset = CADSynth(tmp_path, "train")
    batch = dataset._collate([dataset[0], dataset[1]])

    assert "face_points" not in batch["graph"].ndata
    assert "cylindrical_pos" not in batch["graph"].ndata
    assert "face_points" not in batch
    assert "cylindrical_pos" not in batch


def test_point_only_loading_normalizes_unrequested_cylindrical_field(tmp_path):
    _write_bin(tmp_path, "point_only", include_points=True)
    _write_bin(tmp_path, "point_and_cyl", include_points=True, include_cylindrical=True)
    _write_manifest(tmp_path, "train", ("point_only", "point_and_cyl"))
    dataset = CADSynth(tmp_path, "train", require_face_pointcloud=True)
    batch = dataset._collate([dataset[0], dataset[1]])

    assert batch["face_points"].shape == (4, 96, 10)
    assert "cylindrical_pos" not in batch["graph"].ndata
    assert "cylindrical_pos" not in batch


def test_collator_treats_point_cloud_fields_as_an_atomic_group(new_bin_dataset):
    item = CADSynth(new_bin_dataset, "train", require_face_pointcloud=True)[0]
    item.face_point_mask = None

    with pytest.raises(ValueError, match="face_points.*atomic"):
        CADSynth(new_bin_dataset, "train")._collate([item])


@pytest.mark.parametrize(
    ("manifest", "message"),
    [
        ((), "empty"),
        (("",), "blank"),
        (("repeat", "repeat"), "duplicate"),
        (("missing",), "missing"),
    ],
)
def test_manifest_contract_fails_closed(tmp_path, manifest, message):
    _write_bin(tmp_path, "repeat")
    _write_manifest(tmp_path, "train", manifest)

    with pytest.raises(ValueError, match=message) as error:
        CADSynth(tmp_path, "train")

    assert str((tmp_path / "train.txt").resolve()) in str(error.value)


def test_manifest_preserves_exact_nonnumeric_order(tmp_path):
    _write_bin(tmp_path, "first-nonnumeric")
    _write_bin(tmp_path, "002-middle")
    _write_bin(tmp_path, "last.part")
    expected = ("last.part", "first-nonnumeric", "002-middle")
    _write_manifest(tmp_path, "train", expected)
    dataset = CADSynth(tmp_path, "train")

    assert tuple(path.stem for path in dataset.file_paths) == expected


@pytest.mark.parametrize(
    ("split", "seed_purpose", "expected_purpose"),
    [
        ("val", None, "val_loader"),
        ("test", None, "test_loader"),
        ("train", "custom_loader", "custom_loader"),
        ("train", "", ""),
    ],
)
def test_loader_uses_split_and_explicit_seed_purposes(
    tmp_path, split, seed_purpose, expected_purpose
):
    _write_bin(tmp_path, "sample")
    _write_manifest(tmp_path, split, ("sample",))
    loader = CADSynth(tmp_path, split, seed=123).get_dataloader(
        batch_size=1, shuffle=True, num_workers=1, seed_purpose=seed_purpose
    )
    expected = torch.rand(
        3, generator=torch.Generator().manual_seed(derive_seed(123, expected_purpose))
    )
    actual = torch.rand(3, generator=loader.generator)

    assert torch.equal(actual, expected)
    assert loader.worker_init_fn is dataset_module.seed_worker
    assert loader.prefetch_factor == 2
