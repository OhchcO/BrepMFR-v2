import json

import pytest

from brepmfr.inference.result import (
    ALLOWED_ERROR_STAGES,
    RESULT_SCHEMA_VERSION,
    build_error_document,
    build_success_document,
    write_json_atomic,
)


def _ready_mapping():
    faces = []
    for face_index in range(2):
        faces.append(
            {
                "face_index": face_index,
                "surface_type": "plane",
                "area": 1.0,
                "perimeter": 4.0,
                "centroid": [float(face_index), 0.0, 0.0],
                "bounding_box": [0.0, 0.0, 0.0, 1.0, 1.0, 0.0],
                "loop_count": 1,
                "edge_count": 4,
                "neighbor_count": 1,
                "surface_parameters": {
                    "origin": [0.0, 0.0, 0.0],
                    "axis": [0.0, 0.0, 1.0],
                },
                "sample_points": [[float(face_index), 0.0, 0.0]],
            }
        )
    return {
        "contract_version": "1.0",
        "status": "ready",
        "normalization": {
            "kind": "aag_unit_box",
            "box_side": 1.0,
            "scale_rule": "2_over_longest_bbox_extent",
        },
        "faces": faces,
        "adjacency_edges": [[0, 1]],
    }


def _success_document(tmp_path, timings=None, face_mapping=None, face_count=2):
    return build_success_document(
        step_path=tmp_path / "零件.step",
        step_sha256="step-hash",
        face_count=face_count,
        bin_path=tmp_path / "output" / "input.bin",
        metadata_path=tmp_path / "output" / "input.bin.meta.json",
        cache_hit=False,
        cache_fingerprint="cache-hash",
        checkpoint_path=tmp_path / "model.ckpt",
        checkpoint_sha256="checkpoint-hash",
        checkpoint_epoch=187,
        experiment_mode="dynamic_affinity",
        device="cpu",
        num_classes=25,
        affinity_threshold=0.5,
        faces=[
            {
                "face_index": 0,
                "class_id": 24,
                "feature_name": "stock",
                "confidence": 0.9,
                "instance_id": None,
            }
        ],
        instances=[],
        timings_seconds=(
            timings
            if timings is not None
            else {
                "checkpoint_load": 0.1,
                "conversion": 0.2,
                "inference": 0.3,
                "total": 0.6,
            }
        ),
        warnings=["保留 BIN 缓存"],
        face_mapping=(
            _ready_mapping() if face_mapping is None else face_mapping
        ),
    )


def test_success_document_has_stable_field_order_and_absolute_paths(tmp_path):
    document = _success_document(tmp_path)

    assert RESULT_SCHEMA_VERSION == "1.1"
    assert document["schema_version"] == "1.1"
    assert list(document) == [
        "schema_version",
        "status",
        "input",
        "cache",
        "model",
        "faces",
        "instances",
        "timings_seconds",
        "warnings",
        "face_mapping",
    ]
    assert list(document["input"]) == [
        "step_path",
        "step_sha256",
        "face_count",
    ]
    assert list(document["cache"]) == [
        "bin_path",
        "metadata_path",
        "hit",
        "fingerprint",
    ]
    assert list(document["model"]) == [
        "checkpoint_path",
        "checkpoint_sha256",
        "checkpoint_epoch",
        "experiment_mode",
        "device",
        "num_classes",
        "affinity_threshold",
    ]
    assert document["input"]["step_path"] == (
        tmp_path / "零件.step"
    ).resolve().as_posix()
    assert document["cache"]["bin_path"] == (
        tmp_path / "output" / "input.bin"
    ).resolve().as_posix()
    assert document["model"]["checkpoint_path"] == (
        tmp_path / "model.ckpt"
    ).resolve().as_posix()
    assert document["face_mapping"]["contract_version"] == "1.0"
    assert list(document)[-1] == "face_mapping"


def test_success_document_accepts_unavailable_face_mapping(tmp_path):
    unavailable = {
        "contract_version": "1.0",
        "status": "unavailable",
        "reason": "geometry signatures could not be extracted",
    }

    document = _success_document(tmp_path, face_mapping=unavailable)

    assert document["status"] == "success"
    assert document["face_mapping"] == unavailable


def test_success_document_rejects_nonfinite_face_mapping(tmp_path):
    mapping = _ready_mapping()
    mapping["faces"][0]["area"] = float("nan")

    with pytest.raises(ValueError, match="finite"):
        _success_document(tmp_path, face_mapping=mapping)


def test_success_document_rejects_face_mapping_count_mismatch(tmp_path):
    with pytest.raises(ValueError, match="face_count"):
        _success_document(tmp_path, face_count=3)


def test_result_documents_reject_nonfinite_floats(tmp_path):
    with pytest.raises(ValueError, match="finite"):
        _success_document(tmp_path, timings={"total": float("nan")})

    with pytest.raises(ValueError, match="finite"):
        write_json_atomic(tmp_path / "bad.json", {"value": float("inf")})


def test_error_document_accepts_only_known_stages(tmp_path):
    assert "conversion" in ALLOWED_ERROR_STAGES
    document = build_error_document(
        "conversion", ValueError("无法提取"), tmp_path / "part.step"
    )

    assert list(document) == ["schema_version", "status", "stage", "error", "input"]
    assert document["status"] == "error"
    assert document["schema_version"] == "1.1"
    assert document["error"] == {"type": "ValueError", "message": "无法提取"}
    assert document["input"]["step_path"] == (
        tmp_path / "part.step"
    ).resolve().as_posix()
    with pytest.raises(ValueError, match="stage"):
        build_error_document("unknown", RuntimeError("x"), tmp_path / "part.step")


def test_atomic_json_write_round_trips_utf8_and_ends_with_newline(tmp_path):
    path = tmp_path / "result.json"
    document = _success_document(tmp_path)

    write_json_atomic(path, document)

    content = path.read_text(encoding="utf-8")
    assert content.endswith("\n")
    assert "零件" in content
    assert json.loads(content) == document
    assert not path.with_suffix(".json.tmp").exists()


def test_atomic_json_write_removes_temporary_file_after_replace_error(
    tmp_path, monkeypatch
):
    path = tmp_path / "result.json"

    def fail_replace(_self, _target):
        raise OSError("replace failed")

    monkeypatch.setattr(type(path), "replace", fail_replace)

    with pytest.raises(OSError, match="replace failed"):
        write_json_atomic(path, {"message": "完整"})
    assert not path.with_suffix(".json.tmp").exists()
