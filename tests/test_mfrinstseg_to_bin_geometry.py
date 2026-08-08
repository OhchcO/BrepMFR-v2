from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import dgl
import numpy as np
import pytest
import torch
from dgl.data.utils import load_graphs, save_graphs

from tools import mfrinstseg_to_bin
from tools.geometry_features import (
    POINT_FEATURE_DIM,
    SURFACE_NEIGHBOR_COUNT,
    GeometryFeatureBatch,
    PointSamplingConfig,
)
from tools.mfrinstseg_to_bin import (
    BinRequirements,
    ConversionOptions,
    ConversionPaths,
    LabelData,
    attach_geometry_features,
    convert_one,
    convert_selected_samples,
    geometry_metadata,
    surface_geometry_metadata,
    validate_bin,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_repository_schema_matches_converter_contract():
    schema = mfrinstseg_to_bin.load_schema(PROJECT_ROOT / "tools" / "schema.json")

    assert schema["UV-grid"] == {
        "num_srf_u": 5,
        "num_srf_v": 5,
        "num_crv_u": 5,
    }


def test_repository_extractor_exposes_required_interfaces():
    extractor, topology_utils = mfrinstseg_to_bin.load_extractor(
        PROJECT_ROOT / "tools" / "AAGExtractor.py"
    )

    assert extractor.__name__ == "AAGExtractor"
    assert topology_utils.__name__.endswith("TopologyUtils")


def test_converter_defaults_to_repository_aag_assets():
    args = mfrinstseg_to_bin.parse_args([])

    assert args.extractor_file == PROJECT_ROOT / "tools" / "AAGExtractor.py"
    assert args.schema_file == PROJECT_ROOT / "tools" / "schema.json"


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


def make_geometry_batch(counts=(64, 64)):
    points = torch.zeros((2, 96, POINT_FEATURE_DIM), dtype=torch.float32)
    mask = torch.zeros((2, 96), dtype=torch.bool)
    for face_index, count in enumerate(counts):
        points[face_index, :count, 0] = torch.arange(count, dtype=torch.float32)
        points[face_index, :count, 5] = 1.0
        points[face_index, :count, 6] = torch.linspace(-1.0, 1.0, count)
        points[face_index, :count, 7] = torch.linspace(1.0, -1.0, count)
        points[face_index, :count, 8] = 0.25
        points[face_index, :count, 9] = -0.5
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

    assert graph.ndata["face_points"].shape == (2, 96, POINT_FEATURE_DIM)
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
    assert metadata["point_feature_contract"].tolist() == [2, 10, 1, 2]


def test_parametric_surface_bin_contract_roundtrip(tmp_path):
    path = tmp_path / "parametric.bin"
    batch = make_geometry_batch()
    metrics = np.zeros((2, 96, 3), dtype=np.float32)
    metrics[:, :64, 0] = 1.0
    metrics[:, :64, 2] = 1.0
    neighbors = np.full(
        (2, 96, SURFACE_NEIGHBOR_COUNT), -1, dtype=np.int16
    )
    neighbors[:, :64, 0] = (
        np.arange(64, dtype=np.int16)[None, :] + 1
    ) % 64
    batch = replace(
        batch,
        face_surface_metric=metrics,
        face_trim_neighbors=neighbors,
    )
    options = ConversionOptions(
        include_face_pointcloud=True,
        include_parametric_surface=True,
    )
    graph = make_core_graph()
    attach_geometry_features(graph, batch, options)
    metadata = make_core_metadata()
    metadata.update(geometry_metadata(options.point_sampling))
    metadata.update(surface_geometry_metadata())
    save_graphs(str(path), [graph], metadata)

    validate_bin(
        path,
        expected_nodes=2,
        requirements=BinRequirements(
            require_instance=True,
            require_pointcloud=True,
            require_parametric_surface=True,
        ),
    )
    loaded_graph = load_graphs(str(path))[0][0]
    assert loaded_graph.ndata["face_surface_metric"].shape == (2, 96, 3)
    assert loaded_graph.ndata["face_surface_metric"].dtype == torch.float32
    assert loaded_graph.ndata["face_trim_neighbors"].shape == (
        2,
        96,
        SURFACE_NEIGHBOR_COUNT,
    )
    assert loaded_graph.ndata["face_trim_neighbors"].dtype == torch.int16


def test_geometry_bin_accepts_dgl_roundtrip_uint8_logical_mask(tmp_path):
    path = tmp_path / "roundtrip-mask.bin"
    source_graph, _ = write_geometry_bin(path)

    loaded_graph = load_graphs(str(path))[0][0]

    assert source_graph.ndata["face_point_mask"].dtype == torch.bool
    assert loaded_graph.ndata["face_point_mask"].dtype == torch.uint8
    assert set(loaded_graph.ndata["face_point_mask"].unique().tolist()) == {0, 1}
    validate_bin(
        path,
        requirements=geometry_requirements(),
        point_sampling=PointSamplingConfig(),
    )


def test_geometry_bin_rejects_nonbinary_uint8_mask(tmp_path):
    path = tmp_path / "nonbinary-mask.bin"
    write_geometry_bin(path)
    graphs, metadata = load_graphs(str(path))
    assert graphs[0].ndata["face_point_mask"].dtype == torch.uint8
    graphs[0].ndata["face_point_mask"][0, 0] = 2
    save_graphs(str(path), graphs, metadata)

    with pytest.raises(ValueError, match="face_point_mask.*binary"):
        validate_bin(
            path,
            requirements=geometry_requirements(),
            point_sampling=PointSamplingConfig(),
        )


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
    [
        "point_sampling_mode",
        "point_sampling_config",
        "point_candidate_grid",
        "point_feature_contract",
    ],
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


def test_geometry_bin_rejects_wrong_but_divisible_adaptive_count(tmp_path):
    path = tmp_path / "wrong-adaptive-count.bin"
    graph = make_core_graph()
    graph.ndata["y"] = torch.tensor([1.0, 4.0], dtype=torch.float32)
    options = ConversionOptions(
        seed=2026,
        point_sampling=PointSamplingConfig(),
        include_face_pointcloud=True,
        include_cylindrical_pos=True,
    )
    attach_geometry_features(graph, make_geometry_batch((32, 80)), options)
    metadata = make_core_metadata()
    metadata.update(geometry_metadata(options.point_sampling))
    save_graphs(str(path), [graph], metadata)

    with pytest.raises(ValueError, match="adaptive.*face areas"):
        validate_bin(
            path,
            requirements=geometry_requirements(),
            point_sampling=PointSamplingConfig(),
        )


def test_geometry_bin_accepts_exact_area_derived_adaptive_counts(tmp_path):
    path = tmp_path / "exact-adaptive-count.bin"
    graph = make_core_graph()
    graph.ndata["y"] = torch.tensor([1.0, 4.0], dtype=torch.float32)
    options = ConversionOptions(
        seed=2026,
        point_sampling=PointSamplingConfig(),
        include_face_pointcloud=True,
        include_cylindrical_pos=True,
    )
    attach_geometry_features(graph, make_geometry_batch((40, 80)), options)
    metadata = make_core_metadata()
    metadata.update(geometry_metadata(options.point_sampling))
    save_graphs(str(path), [graph], metadata)

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


def test_geometry_bin_rejects_wrong_10d_feature_contract_values(tmp_path):
    path = tmp_path / "feature-contract.bin"
    graph, metadata = write_geometry_bin(path)
    metadata["point_feature_contract"] = torch.tensor(
        [1, 10, 2, 1], dtype=torch.int16
    )
    save_graphs(str(path), [graph], metadata)

    with pytest.raises(ValueError, match="point_feature_contract"):
        validate_bin(
            path,
            requirements=geometry_requirements(),
            point_sampling=PointSamplingConfig(),
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


def test_geometry_bin_rejects_non_unit_valid_normals(tmp_path):
    path = tmp_path / "normal.bin"
    graph, metadata = write_geometry_bin(path)
    graph.ndata["face_points"][0, 0, 3:6] = torch.tensor(
        [0.0, 0.0, 0.5], dtype=torch.float32
    )
    save_graphs(str(path), [graph], metadata)

    with pytest.raises(ValueError, match="unit normals"):
        validate_bin(
            path,
            requirements=geometry_requirements(),
            point_sampling=PointSamplingConfig(),
        )


@pytest.mark.parametrize("column", [6, 7])
def test_geometry_bin_rejects_uv_coordinates_outside_normalized_range(
    tmp_path, column
):
    path = tmp_path / ("uv-{0}.bin".format(column))
    graph, metadata = write_geometry_bin(path)
    graph.ndata["face_points"][0, 0, column] = 1.01
    save_graphs(str(path), [graph], metadata)

    with pytest.raises(ValueError, match="normalized UV"):
        validate_bin(
            path,
            requirements=geometry_requirements(),
            point_sampling=PointSamplingConfig(),
        )


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        (0, -0.01, "normalized radius"),
        (3, 1.01, "normalized z"),
        (1, 0.0, "sine/cosine"),
    ],
)
def test_geometry_bin_rejects_invalid_cylindrical_semantics(
    tmp_path, column, value, message
):
    path = tmp_path / ("cyl-{0}.bin".format(column))
    graph, metadata = write_geometry_bin(path)
    graph.ndata["cylindrical_pos"][0, column] = value
    if column == 1:
        graph.ndata["cylindrical_pos"][0, 2] = 0.0
    save_graphs(str(path), [graph], metadata)

    with pytest.raises(ValueError, match=message):
        validate_bin(
            path,
            requirements=geometry_requirements(),
            point_sampling=PointSamplingConfig(),
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


def test_conversion_paths_can_store_bins_under_a_shared_cache(tmp_path):
    paths = ConversionPaths(
        dataset_root=tmp_path / "raw",
        subset_root=tmp_path / "manifests",
        output_root=tmp_path / "reports",
        bin_cache_root=tmp_path / "shared",
    )

    assert paths.bins_root == tmp_path / "shared" / "bin"
    assert paths.bin_file("sample_001") == (
        tmp_path / "shared" / "bin" / "sample_001.bin"
    )


def test_conversion_paths_keep_legacy_output_bin_default(tmp_path):
    paths = ConversionPaths(
        dataset_root=tmp_path / "raw",
        subset_root=tmp_path / "manifests",
        output_root=tmp_path / "reports",
    )

    assert paths.bins_root == tmp_path / "reports" / "bin"


def test_converter_cli_accepts_separate_bin_cache_root(tmp_path):
    shared = tmp_path / "shared"

    args = mfrinstseg_to_bin.parse_args(
        ["--bin-cache-root", str(shared)]
    )

    assert args.bin_cache_root == shared


def make_convert_one_fixture(tmp_path, monkeypatch):
    paths = ConversionPaths(
        dataset_root=tmp_path / "raw",
        subset_root=tmp_path / "subset",
        output_root=tmp_path / "output",
    )
    step_file = paths.dataset_root / "steps" / "sample.step"
    step_file.parent.mkdir(parents=True)
    step_file.write_text("STEP", encoding="utf-8")
    label = LabelData(
        sample_id="sample",
        semantic=np.array([0, 1], dtype=np.int32),
        instance=np.eye(2, dtype=np.int8),
    )
    monkeypatch.setattr(mfrinstseg_to_bin, "load_label", lambda path: label)
    monkeypatch.setattr(
        mfrinstseg_to_bin, "build_graph", lambda *args: make_core_graph()
    )
    monkeypatch.setattr(
        mfrinstseg_to_bin,
        "shortest_path_metadata",
        lambda *args: {
            key: value
            for key, value in make_core_metadata().items()
            if key != "instance"
        },
    )

    class FakeExtractor:
        def __init__(self, step_file, attribute_schema):
            self.step_file = step_file
            self.attribute_schema = attribute_schema
            self._graph = object()

        def process(self):
            return {"extraction": True}

    return paths, FakeExtractor


def test_convert_one_retains_extractor_topology_in_memory(tmp_path, monkeypatch):
    paths, extractor_class = make_convert_one_fixture(tmp_path, monkeypatch)
    captured = {}

    def fake_build_graph(extraction, *args):
        captured["extractor"] = extraction["_extractor"]
        return make_core_graph()

    monkeypatch.setattr(mfrinstseg_to_bin, "build_graph", fake_build_graph)
    monkeypatch.setattr(
        mfrinstseg_to_bin,
        "save_graphs",
        lambda path, *args: type(paths.bin_file("sample"))(path).write_bytes(b"bin"),
    )
    monkeypatch.setattr(mfrinstseg_to_bin, "validate_bin", lambda *args, **kwargs: None)

    assert (
        convert_one(
            "sample",
            paths,
            extractor_class,
            topology_utils=object(),
            schema={},
            options=ConversionOptions(),
            overwrite=True,
        )
        == "converted"
    )
    assert captured["extractor"]._graph is not None


def test_convert_one_validates_temporary_before_atomic_replace(
    tmp_path, monkeypatch
):
    paths, extractor = make_convert_one_fixture(tmp_path, monkeypatch)
    target = paths.bin_file("sample")
    target.parent.mkdir(parents=True)
    target.write_bytes(b"old")
    events = []

    def fake_save(path, graphs, metadata):
        temporary = type(target)(path)
        events.append(("save", temporary.name))
        assert target.read_bytes() == b"old"
        temporary.write_bytes(b"new")

    def fake_validate(path, *args, **kwargs):
        events.append(("validate", path.name))
        assert path == target.with_suffix(".bin.tmp")
        assert path.read_bytes() == b"new"
        assert target.read_bytes() == b"old"

    monkeypatch.setattr(mfrinstseg_to_bin, "save_graphs", fake_save)
    monkeypatch.setattr(mfrinstseg_to_bin, "validate_bin", fake_validate)

    outcome = convert_one(
        "sample",
        paths,
        extractor,
        topology_utils=object(),
        schema={},
        options=ConversionOptions(),
        overwrite=True,
    )

    assert outcome == "converted"
    assert events == [("save", "sample.bin.tmp"), ("validate", "sample.bin.tmp")]
    assert target.read_bytes() == b"new"
    assert not target.with_suffix(".bin.tmp").exists()


@pytest.mark.parametrize("failure_stage", ["save", "validate"])
def test_convert_one_preserves_target_and_cleans_temporary_on_failure(
    tmp_path, monkeypatch, failure_stage
):
    paths, extractor = make_convert_one_fixture(tmp_path, monkeypatch)
    target = paths.bin_file("sample")
    target.parent.mkdir(parents=True)
    target.write_bytes(b"old")

    def fake_save(path, graphs, metadata):
        type(target)(path).write_bytes(b"partial")
        if failure_stage == "save":
            raise OSError("save failed")

    def fake_validate(path, *args, **kwargs):
        if failure_stage == "validate":
            raise ValueError("validation failed")

    monkeypatch.setattr(mfrinstseg_to_bin, "save_graphs", fake_save)
    monkeypatch.setattr(mfrinstseg_to_bin, "validate_bin", fake_validate)

    with pytest.raises((OSError, ValueError), match="failed"):
        convert_one(
            "sample",
            paths,
            extractor,
            topology_utils=object(),
            schema={},
            options=ConversionOptions(),
            overwrite=True,
        )

    assert target.read_bytes() == b"old"
    assert not target.with_suffix(".bin.tmp").exists()


def test_convert_one_rejects_incompatible_existing_bin_without_overwrite(
    tmp_path, monkeypatch
):
    paths, extractor = make_convert_one_fixture(tmp_path, monkeypatch)
    target = paths.bin_file("sample")
    target.parent.mkdir(parents=True)
    target.write_bytes(b"old")
    monkeypatch.setattr(
        mfrinstseg_to_bin,
        "validate_bin",
        lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("incompatible")),
    )
    monkeypatch.setattr(
        mfrinstseg_to_bin,
        "save_graphs",
        lambda *args: pytest.fail("incompatible target must not be overwritten"),
    )

    with pytest.raises(ValueError, match="new output directory|--overwrite"):
        convert_one(
            "sample",
            paths,
            extractor,
            topology_utils=object(),
            schema={},
            options=ConversionOptions(),
            overwrite=False,
        )

    assert target.read_bytes() == b"old"
    assert not target.with_suffix(".bin.tmp").exists()
