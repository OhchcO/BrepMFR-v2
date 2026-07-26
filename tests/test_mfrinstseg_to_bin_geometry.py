from dataclasses import replace

import dgl
import pytest
import torch
from dgl.data.utils import save_graphs

from tools import mfrinstseg_to_bin
from tools.geometry_features import GeometryFeatureBatch, PointSamplingConfig
from tools.mfrinstseg_to_bin import (
    BinRequirements,
    ConversionOptions,
    ConversionPaths,
    attach_geometry_features,
    convert_selected_samples,
    geometry_metadata,
    validate_bin,
)


def make_core_graph():
    graph = dgl.graph(([0, 1], [1, 0]), num_nodes=2, idtype=torch.int64)
    graph.ndata["x"] = torch.zeros((2, 5, 5, 7), dtype=torch.float32)
    graph.ndata["y"] = torch.ones(2, dtype=torch.float32)
    graph.ndata["z"] = torch.zeros(2, dtype=torch.int32)
    graph.ndata["l"] = torch.ones(2, dtype=torch.int32)
    graph.ndata["a"] = torch.ones(2, dtype=torch.int32)
    graph.ndata["f"] = torch.tensor([0, 1], dtype=torch.int32)
    graph.edata["x"] = torch.zeros((2, 5, 7), dtype=torch.float32)
    graph.edata["t"] = torch.zeros(2, dtype=torch.int32)
    graph.edata["l"] = torch.ones(2, dtype=torch.float32)
    graph.edata["a"] = torch.zeros(2, dtype=torch.float32)
    graph.edata["c"] = torch.zeros(2, dtype=torch.int32)
    return graph


def make_core_metadata():
    return {
        "spatial_pos": torch.tensor([[0, 1], [1, 0]], dtype=torch.int32),
        "edges_path": torch.full((2, 2, 6), -1, dtype=torch.int64),
        "d2_distance": torch.zeros((2, 2, 64), dtype=torch.float32),
        "angle_distance": torch.zeros((2, 2, 64), dtype=torch.float32),
        "instance": torch.eye(2, dtype=torch.int8),
    }


def make_geometry_batch(counts=(32, 96)):
    points = torch.zeros((2, 96, 6), dtype=torch.float32)
    mask = torch.zeros((2, 96), dtype=torch.bool)
    for face_index, count in enumerate(counts):
        points[face_index, :count, 0] = torch.arange(count, dtype=torch.float32)
        points[face_index, :count, 5] = 1.0
        mask[face_index, :count] = True
    return GeometryFeatureBatch(
        face_points=points.numpy(),
        face_point_mask=mask.numpy(),
        face_point_count=torch.tensor(counts, dtype=torch.int32).numpy(),
        cylindrical_pos=torch.tensor(
            [[0.0, 0.0, 1.0, 0.0], [1.0, 1.0, 0.0, 1.0]],
            dtype=torch.float32,
        ).numpy(),
    )


def write_geometry_bin(path, batch=None, options=None):
    graph = make_core_graph()
    options = options or ConversionOptions(
        seed=2026,
        point_sampling=PointSamplingConfig(),
        include_face_pointcloud=True,
        include_cylindrical_pos=True,
    )
    attach_geometry_features(graph, batch or make_geometry_batch(), options)
    metadata = make_core_metadata()
    metadata.update(geometry_metadata(options.point_sampling))
    save_graphs(str(path), [graph], metadata)
    return graph, metadata


def geometry_requirements():
    return BinRequirements(
        require_instance=True,
        require_pointcloud=True,
        require_cylindrical=True,
    )


def test_geometry_bin_has_exact_shapes_and_dtypes_and_allows_extra_fields(tmp_path):
    path = tmp_path / "sample.bin"
    graph, metadata = write_geometry_bin(path)
    graph.ndata["future_node_field"] = torch.zeros(2, dtype=torch.float32)
    metadata["future_metadata_field"] = torch.ones(1, dtype=torch.int16)
    save_graphs(str(path), [graph], metadata)

    validate_bin(
        path,
        expected_nodes=2,
        requirements=geometry_requirements(),
        point_sampling=PointSamplingConfig(),
    )

    assert graph.ndata["face_points"].shape == (2, 96, 6)
    assert graph.ndata["face_points"].dtype == torch.float32
    assert graph.ndata["face_point_mask"].shape == (2, 96)
    assert graph.ndata["face_point_mask"].dtype == torch.bool
    assert graph.ndata["face_point_count"].shape == (2,)
    assert graph.ndata["face_point_count"].dtype == torch.int32
    assert graph.ndata["cylindrical_pos"].shape == (2, 4)
    assert graph.ndata["cylindrical_pos"].dtype == torch.float32
    assert metadata["point_sampling_mode"].dtype == torch.int8
    assert metadata["point_sampling_config"].dtype == torch.float32
    assert metadata["point_candidate_grid"].dtype == torch.int32


@pytest.mark.parametrize(
    ("field", "index"),
    [
        ("face_points", (0, 0, 0)),
        ("cylindrical_pos", (0, 0)),
    ],
)
def test_geometry_bin_rejects_non_finite_values(tmp_path, field, index):
    path = tmp_path / "nan.bin"
    graph, metadata = write_geometry_bin(path)
    graph.ndata[field][index] = float("nan")
    save_graphs(str(path), [graph], metadata)

    with pytest.raises(ValueError, match="non-finite"):
        validate_bin(
            path,
            requirements=geometry_requirements(),
            point_sampling=PointSamplingConfig(),
        )


def test_geometry_bin_rejects_mask_count_mismatch(tmp_path):
    path = tmp_path / "mask.bin"
    graph, metadata = write_geometry_bin(path)
    graph.ndata["face_point_mask"][0, 0] = False
    save_graphs(str(path), [graph], metadata)

    with pytest.raises(ValueError, match="mask.*count"):
        validate_bin(
            path,
            requirements=geometry_requirements(),
            point_sampling=PointSamplingConfig(),
        )


@pytest.mark.parametrize(
    ("node_field", "requirement"),
    [
        ("face_points", BinRequirements(True, True, False)),
        ("face_point_mask", BinRequirements(True, True, False)),
        ("face_point_count", BinRequirements(True, True, False)),
        ("cylindrical_pos", BinRequirements(True, False, True)),
    ],
)
def test_geometry_bin_rejects_missing_requested_node_fields(
    tmp_path, node_field, requirement
):
    path = tmp_path / (node_field + ".bin")
    graph, metadata = write_geometry_bin(path)
    del graph.ndata[node_field]
    save_graphs(str(path), [graph], metadata)

    with pytest.raises(ValueError, match=node_field):
        validate_bin(
            path,
            requirements=requirement,
            point_sampling=PointSamplingConfig(),
        )


@pytest.mark.parametrize(
    "metadata_field",
    ["point_sampling_mode", "point_sampling_config", "point_candidate_grid"],
)
def test_geometry_bin_rejects_missing_sampling_metadata(tmp_path, metadata_field):
    path = tmp_path / (metadata_field + ".bin")
    graph, metadata = write_geometry_bin(path)
    del metadata[metadata_field]
    save_graphs(str(path), [graph], metadata)

    with pytest.raises(ValueError, match=metadata_field):
        validate_bin(
            path,
            requirements=BinRequirements(True, True, False),
            point_sampling=PointSamplingConfig(),
        )


def test_geometry_bin_rejects_invalid_adaptive_counts(tmp_path):
    path = tmp_path / "bad-count.bin"
    write_geometry_bin(path, batch=make_geometry_batch((40, 95)))

    with pytest.raises(ValueError, match="adaptive.*multiple of 8"):
        validate_bin(
            path,
            requirements=geometry_requirements(),
            point_sampling=PointSamplingConfig(),
        )


def test_geometry_bin_rejects_fixed_count_mismatch(tmp_path):
    path = tmp_path / "fixed.bin"
    fixed = PointSamplingConfig(mode="fixed", fixed_count=64)
    options = ConversionOptions(
        seed=2026,
        point_sampling=fixed,
        include_face_pointcloud=True,
        include_cylindrical_pos=True,
    )
    write_geometry_bin(path, batch=make_geometry_batch((32, 96)), options=options)

    with pytest.raises(ValueError, match="fixed"):
        validate_bin(
            path,
            requirements=geometry_requirements(),
            point_sampling=fixed,
        )


def test_geometry_bin_rejects_sampling_metadata_for_different_request(tmp_path):
    path = tmp_path / "metadata.bin"
    write_geometry_bin(path)
    changed = replace(PointSamplingConfig(), area_exponent=0.75)

    with pytest.raises(ValueError, match="point_sampling_config"):
        validate_bin(
            path,
            requirements=geometry_requirements(),
            point_sampling=changed,
        )


def test_legacy_bin_is_valid_when_optional_geometry_is_not_requested(tmp_path):
    path = tmp_path / "legacy.bin"
    save_graphs(str(path), [make_core_graph()], make_core_metadata())

    validate_bin(
        path,
        requirements=BinRequirements(
            require_instance=True,
            require_pointcloud=False,
            require_cylindrical=False,
        ),
    )


def test_selected_conversion_accepts_automatic_preparation_root_contract(
    tmp_path, monkeypatch
):
    prepared = tmp_path / "prepared"
    paths = ConversionPaths(
        dataset_root=tmp_path / "raw",
        subset_root=prepared,
        output_root=prepared,
    )
    options = ConversionOptions(
        seed=2026,
        point_sampling=PointSamplingConfig(),
        include_face_pointcloud=True,
        include_cylindrical_pos=True,
    )
    calls = []
    monkeypatch.setattr(mfrinstseg_to_bin, "load_schema", lambda path: {"schema": path})
    monkeypatch.setattr(
        mfrinstseg_to_bin,
        "load_extractor",
        lambda path: ("extractor", "topology"),
    )

    def fake_convert_one(
        sample_id,
        actual_paths,
        extractor,
        topology,
        schema,
        actual_options,
        overwrite,
    ):
        calls.append(
            (
                sample_id,
                actual_paths,
                extractor,
                topology,
                schema,
                actual_options,
                overwrite,
            )
        )
        return "converted"

    monkeypatch.setattr(mfrinstseg_to_bin, "convert_one", fake_convert_one)

    report = convert_selected_samples(
        {"train": ("sample_001",), "val": (), "test": ()},
        paths,
        tmp_path / "extractor.py",
        tmp_path / "schema.json",
        options,
        overwrite=False,
    )

    assert report.successful_ids == {
        "train": ["sample_001"],
        "val": [],
        "test": [],
    }
    assert report.failed_ids == []
    assert calls[0][1] == paths
    assert calls[0][5] == options
    assert (prepared / "train.txt").read_text(encoding="utf-8") == "sample_001\n"
    assert (prepared / "conversion_failures.json").read_text(
        encoding="utf-8"
    ) == "[]\n"
