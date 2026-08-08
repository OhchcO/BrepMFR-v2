"""Verify that the ProcessPoolExecutor code path in convert_selected_samples works."""
from pathlib import Path
from unittest.mock import patch

import pytest

from tools import mfrinstseg_to_bin
from tools.mfrinstseg_to_bin import (
    ConversionOptions,
    ConversionPaths,
    _convert_one_worker,
    _init_worker,
    convert_selected_samples,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture()
def fake_dataset(tmp_path):
    """Create a minimal fake raw dataset with one sample."""
    steps = tmp_path / "steps"
    labels = tmp_path / "labels"
    steps.mkdir()
    labels.mkdir()
    # Create a dummy STEP file
    (steps / "sample_001.step").write_text("ISO-10303-21;\nEND-ISO-10303-21;\n")
    # Create a minimal label JSON
    import json
    label = {
        "sample_id": "sample_001",
        "seg": [0, 0],
        "inst": [0, 0],
        "adj": [[1], [0]],
    }
    (labels / "sample_001.json").write_text(json.dumps(label))
    return tmp_path


def test_init_worker_sets_globals(tmp_path):
    """_init_worker should populate module-level globals."""
    schema_file = PROJECT_ROOT / "tools" / "schema.json"
    extractor_file = PROJECT_ROOT / "tools" / "AAGExtractor.py"
    raw = tmp_path / "raw"
    prepared = tmp_path / "prepared"
    raw.mkdir()
    prepared.mkdir()

    options_dict = {
        "seed": 42,
        "include_face_pointcloud": False,
        "include_parametric_surface": False,
        "include_cylindrical_pos": False,
    }

    # Before init, globals should be None
    assert mfrinstseg_to_bin._worker_extractor_class is None

    _init_worker(
        str(extractor_file), str(schema_file),
        str(raw), str(prepared), "", options_dict,
    )

    assert mfrinstseg_to_bin._worker_extractor_class is not None
    assert mfrinstseg_to_bin._worker_topology_utils is not None
    assert mfrinstseg_to_bin._worker_schema is not None
    assert mfrinstseg_to_bin._worker_paths is not None
    assert mfrinstseg_to_bin._worker_options is not None

    # Clean up globals
    mfrinstseg_to_bin._worker_extractor_class = None
    mfrinstseg_to_bin._worker_topology_utils = None
    mfrinstseg_to_bin._worker_schema = None
    mfrinstseg_to_bin._worker_paths = None
    mfrinstseg_to_bin._worker_options = None


def test_convert_one_worker_delegates_to_convert_one(monkeypatch):
    """_convert_one_worker should call convert_one with globals."""
    calls = []

    def fake_convert_one(sample_id, paths, extractor, topology, schema, options, overwrite):
        calls.append({
            "sample_id": sample_id,
            "paths": paths,
            "overwrite": overwrite,
        })
        return "converted"

    monkeypatch.setattr(mfrinstseg_to_bin, "convert_one", fake_convert_one)
    monkeypatch.setattr(mfrinstseg_to_bin, "_worker_paths", "fake_paths")
    monkeypatch.setattr(mfrinstseg_to_bin, "_worker_extractor_class", "fake_extractor")
    monkeypatch.setattr(mfrinstseg_to_bin, "_worker_topology_utils", "fake_topology")
    monkeypatch.setattr(mfrinstseg_to_bin, "_worker_schema", "fake_schema")
    monkeypatch.setattr(mfrinstseg_to_bin, "_worker_options", "fake_options")

    result = _convert_one_worker("sample_001", overwrite=False)

    assert result == "converted"
    assert len(calls) == 1
    assert calls[0]["sample_id"] == "sample_001"
    assert calls[0]["paths"] == "fake_paths"
    assert calls[0]["overwrite"] is False


def test_convert_selected_samples_serial_fallback(tmp_path, monkeypatch):
    """num_workers=1 should use serial path and work as before."""
    calls = []

    def fake_convert_one(sample_id, paths, extractor, topology, schema, options, overwrite):
        calls.append(sample_id)
        return "converted"

    monkeypatch.setattr(mfrinstseg_to_bin, "convert_one", fake_convert_one)

    prepared = tmp_path / "prepared"
    prepared.mkdir()
    paths = ConversionPaths(
        dataset_root=tmp_path / "raw",
        subset_root=prepared,
        output_root=prepared,
    )
    options = ConversionOptions()

    report = convert_selected_samples(
        {"train": ("s1",), "val": (), "test": ()},
        paths,
        PROJECT_ROOT / "tools" / "AAGExtractor.py",
        PROJECT_ROOT / "tools" / "schema.json",
        options,
        overwrite=False,
        num_workers=1,
    )

    assert "s1" in calls
    assert report.successful_ids["train"] == ["s1"]
