import copy
import json
import math
from pathlib import Path

import numpy as np
import pytest
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Transform
from OCC.Core.gp import gp_Pnt, gp_Trsf, gp_Vec
from OCC.Core.STEPControl import (
    STEPControl_AsIs,
    STEPControl_Reader,
    STEPControl_Writer,
)

from brepmfr.inference.face_signature import (
    FACE_MAPPING_CONTRACT_VERSION,
    NORMALIZATION_KIND,
    extract_face_mapping,
    face_mapping_json_bytes,
    read_face_mapping,
    validate_face_mapping_document,
)
from tools.AAGExtractor import AAGExtractor
from tools.mfrinstseg_to_bin import DEFAULT_SCHEMA_FILE, load_schema


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
COMMITTED_STEP = (
    REPOSITORY_ROOT
    / "tests"
    / "assets"
    / "step_inference"
    / "20221121_154647_0.step"
)


def _extract_mapping(step_path):
    extractor = AAGExtractor(step_path, load_schema(DEFAULT_SCHEMA_FILE))
    extraction = extractor.process()
    return extract_face_mapping(extractor, extraction)


def _assert_all_numbers_are_finite(value):
    if isinstance(value, dict):
        for nested in value.values():
            _assert_all_numbers_are_finite(nested)
    elif isinstance(value, (list, tuple)):
        for nested in value:
            _assert_all_numbers_are_finite(nested)
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        assert math.isfinite(value)


def _ready_mapping():
    faces = []
    for face_index, centroid in enumerate(([0.0, 0.0, 0.0], [0.5, 0.0, 0.0])):
        faces.append(
            {
                "face_index": face_index,
                "surface_type": "plane",
                "area": 0.25,
                "perimeter": 2.0,
                "centroid": centroid,
                "bounding_box": [-0.5, -0.5, 0.0, 0.5, 0.5, 0.0],
                "loop_count": 1,
                "edge_count": 4,
                "neighbor_count": 1,
                "surface_parameters": {
                    "origin": [0.0, 0.0, 0.0],
                    "axis": [0.0, 0.0, 1.0],
                },
                "sample_points": [centroid],
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


@pytest.fixture(scope="module")
def golden_extraction():
    extractor = AAGExtractor(COMMITTED_STEP, load_schema(DEFAULT_SCHEMA_FILE))
    return extractor, extractor.process()


def test_golden_step_face_mapping_is_complete_and_aligned(golden_extraction):
    mapping = extract_face_mapping(*golden_extraction)

    assert FACE_MAPPING_CONTRACT_VERSION == "1.0"
    assert NORMALIZATION_KIND == "aag_unit_box"
    assert mapping["contract_version"] == "1.0"
    assert mapping["status"] == "ready"
    assert [face["face_index"] for face in mapping["faces"]] == list(range(41))
    assert all(face["sample_points"] for face in mapping["faces"])
    assert mapping["adjacency_edges"] == sorted(mapping["adjacency_edges"])
    assert all(left < right for left, right in mapping["adjacency_edges"])
    assert len(mapping["adjacency_edges"]) == len(
        {tuple(edge) for edge in mapping["adjacency_edges"]}
    )
    assert all(
        0 <= endpoint < 41
        for edge in mapping["adjacency_edges"]
        for endpoint in edge
    )
    _assert_all_numbers_are_finite(mapping)

    degrees = [0] * 41
    for left, right in mapping["adjacency_edges"]:
        degrees[left] += 1
        degrees[right] += 1
    assert [face["neighbor_count"] for face in mapping["faces"]] == degrees


def _read_step_shape(path):
    reader = STEPControl_Reader()
    assert reader.ReadFile(str(path)) == 1
    assert reader.TransferRoots() > 0
    return reader.OneShape()


def _write_transformed_step(source, target, transform):
    shape = BRepBuilderAPI_Transform(_read_step_shape(source), transform, True).Shape()
    writer = STEPControl_Writer()
    assert writer.Transfer(shape, STEPControl_AsIs) == 1
    assert writer.Write(str(target)) == 1


def _assert_mapping_geometry_close(actual, expected):
    assert actual["adjacency_edges"] == expected["adjacency_edges"]
    assert len(actual["faces"]) == len(expected["faces"])
    for actual_face, expected_face in zip(actual["faces"], expected["faces"]):
        assert actual_face["face_index"] == expected_face["face_index"]
        assert actual_face["surface_type"] == expected_face["surface_type"]
        assert actual_face["loop_count"] == expected_face["loop_count"]
        assert actual_face["edge_count"] == expected_face["edge_count"]
        assert actual_face["neighbor_count"] == expected_face["neighbor_count"]
        np.testing.assert_allclose(
            actual_face["area"], expected_face["area"], atol=1e-6, rtol=0.0
        )
        np.testing.assert_allclose(
            actual_face["perimeter"],
            expected_face["perimeter"],
            atol=1e-6,
            rtol=0.0,
        )
        np.testing.assert_allclose(
            actual_face["centroid"],
            expected_face["centroid"],
            atol=1e-6,
            rtol=0.0,
        )
        np.testing.assert_allclose(
            actual_face["bounding_box"],
            expected_face["bounding_box"],
            atol=1e-6,
            rtol=0.0,
        )
        assert actual_face["surface_parameters"].keys() == expected_face[
            "surface_parameters"
        ].keys()
        for name, expected_value in expected_face["surface_parameters"].items():
            np.testing.assert_allclose(
                actual_face["surface_parameters"][name],
                expected_value,
                atol=1e-6,
                rtol=0.0,
            )


def test_face_mapping_is_invariant_to_translation_and_uniform_scale(
    tmp_path, golden_extraction
):
    golden_mapping = extract_face_mapping(*golden_extraction)
    translation = gp_Trsf()
    translation.SetTranslation(gp_Vec(123.0, -456.0, 789.0))
    translated_step = tmp_path / "translated.step"
    _write_transformed_step(COMMITTED_STEP, translated_step, translation)

    scaling = gp_Trsf()
    scaling.SetScale(gp_Pnt(0.0, 0.0, 0.0), 25.4)
    scaled_step = tmp_path / "scaled.step"
    _write_transformed_step(COMMITTED_STEP, scaled_step, scaling)

    _assert_mapping_geometry_close(_extract_mapping(translated_step), golden_mapping)
    _assert_mapping_geometry_close(_extract_mapping(scaled_step), golden_mapping)


def test_face_mapping_round_trips_as_canonical_utf8(tmp_path):
    mapping = _ready_mapping()

    encoded = face_mapping_json_bytes(mapping)
    sidecar = tmp_path / "input.face_signatures.json"
    sidecar.write_bytes(encoded)

    assert encoded.endswith(b"\n")
    assert encoded == (
        json.dumps(mapping, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    assert read_face_mapping(sidecar, expected_face_count=2) == mapping
    assert validate_face_mapping_document(mapping, expected_face_count=2) == mapping


def test_face_mapping_accepts_explicit_unavailable_status():
    unavailable = {
        "contract_version": "1.0",
        "status": "unavailable",
        "reason": "geometry signatures could not be extracted",
    }

    assert validate_face_mapping_document(unavailable) == unavailable


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda value: value.update(contract_version="9.9"), "contract_version"),
        (lambda value: value.update(status="partial"), "status"),
        (lambda value: value["faces"][0].update(face_index=4), "continuous"),
        (lambda value: value["faces"][0].update(area=float("nan")), "finite"),
        (lambda value: value["faces"][0].update(centroid=[0.0, 0.0]), "centroid"),
        (lambda value: value["faces"][0].update(sample_points=[]), "sample_points"),
        (lambda value: value.update(adjacency_edges=[[0, 0]]), "self"),
        (lambda value: value.update(adjacency_edges=[[0, 1], [0, 1]]), "duplicate"),
        (lambda value: value["faces"][0].update(neighbor_count=0), "neighbor_count"),
    ],
)
def test_face_mapping_rejects_corrupt_ready_documents(mutate, message):
    mapping = copy.deepcopy(_ready_mapping())
    mutate(mapping)

    with pytest.raises(ValueError, match=message):
        validate_face_mapping_document(mapping, expected_face_count=2)


def test_face_mapping_rejects_expected_face_count_mismatch():
    with pytest.raises(ValueError, match="face_count"):
        validate_face_mapping_document(_ready_mapping(), expected_face_count=3)


@pytest.mark.parametrize(
    "unavailable",
    [
        {"contract_version": "1.0", "status": "unavailable"},
        {"contract_version": "1.0", "status": "unavailable", "reason": ""},
    ],
)
def test_face_mapping_rejects_unavailable_without_reason(unavailable):
    with pytest.raises(ValueError, match="reason"):
        validate_face_mapping_document(unavailable)
