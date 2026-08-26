import json

import pytest

from brepmfr.inference.cache import (
    INFERENCE_BIN_CONTRACT_VERSION,
    build_cache_identity,
    cache_fingerprint,
    inspect_cache,
    sha256_file,
    write_cache_metadata_atomic,
)


def _cache_inputs(tmp_path):
    step = tmp_path / "part.step"
    implementation = tmp_path / "converter.py"
    bin_path = tmp_path / "input.bin"
    signature_path = tmp_path / "input.face_signatures.json"
    metadata_path = tmp_path / "input.bin.meta.json"
    step.write_bytes(b"STEP-A")
    implementation.write_bytes(b"converter-v1")
    bin_path.write_bytes(b"valid-bin")
    signature_path.write_bytes(b"valid-signatures")
    identity = build_cache_identity(
        step,
        [implementation],
        42,
        {"mode": "fixed", "points_per_face": 96},
        {
            "pointcloud": True,
            "parametric_surface": True,
            "cylindrical": True,
        },
    )
    return step, implementation, bin_path, signature_path, metadata_path, identity


def test_sha256_file_hashes_file_content(tmp_path):
    path = tmp_path / "payload"
    path.write_bytes(b"abc")

    assert sha256_file(path) == (
        "ba7816bf8f01cfea414140de5dae2223"
        "b00361a396177a9cb410ff61f20015ad"
    )


def test_cache_hits_only_when_identity_and_bin_validation_match(tmp_path):
    _, _, bin_path, signature_path, metadata_path, identity = _cache_inputs(tmp_path)
    fingerprint = cache_fingerprint(identity)
    write_cache_metadata_atomic(
        metadata_path, identity, fingerprint, bin_path, signature_path
    )
    validated = []

    decision = inspect_cache(
        bin_path,
        signature_path,
        metadata_path,
        identity,
        lambda path: validated.append(("bin", path)),
        lambda path: validated.append(("signature", path)),
    )

    assert decision.hit is True
    assert decision.reason == "hit"
    assert decision.fingerprint == fingerprint
    assert decision.bin_path == bin_path
    assert decision.signature_path == signature_path
    assert decision.metadata_path == metadata_path
    assert validated == [("bin", bin_path), ("signature", signature_path)]
    assert json.loads(metadata_path.read_text(encoding="utf-8")) == {
        "schema_version": INFERENCE_BIN_CONTRACT_VERSION,
        "fingerprint": fingerprint,
        "identity": identity,
        "bin_sha256": sha256_file(bin_path),
        "face_signatures_sha256": sha256_file(signature_path),
    }
    assert INFERENCE_BIN_CONTRACT_VERSION == 3


@pytest.mark.parametrize(
    ("force_rebuild", "remove", "metadata", "expected_reason"),
    [
        (True, None, None, "forced"),
        (False, "bin", None, "missing-bin"),
        (False, "signatures", None, "missing-signatures"),
        (False, "metadata", None, "missing-metadata"),
        (False, None, "not-json", "invalid-metadata"),
        (False, None, "wrong-identity", "identity-mismatch"),
    ],
)
def test_cache_miss_reasons_are_stable(
    tmp_path, force_rebuild, remove, metadata, expected_reason
):
    _, _, bin_path, signature_path, metadata_path, identity = _cache_inputs(tmp_path)
    fingerprint = cache_fingerprint(identity)
    write_cache_metadata_atomic(
        metadata_path, identity, fingerprint, bin_path, signature_path
    )
    if remove == "bin":
        bin_path.unlink()
    elif remove == "signatures":
        signature_path.unlink()
    elif remove == "metadata":
        metadata_path.unlink()
    if metadata == "not-json":
        metadata_path.write_text("{", encoding="utf-8")
    elif metadata == "wrong-identity":
        changed = dict(identity)
        changed["seed"] = 7
        write_cache_metadata_atomic(
            metadata_path,
            changed,
            cache_fingerprint(changed),
            bin_path,
            signature_path,
        )

    decision = inspect_cache(
        bin_path,
        signature_path,
        metadata_path,
        identity,
        lambda path: None,
        lambda path: None,
        force_rebuild=force_rebuild,
    )

    assert decision.hit is False
    assert decision.reason == expected_reason
    assert decision.fingerprint == fingerprint


def test_cache_validator_failure_becomes_invalid_bin_miss(tmp_path):
    _, _, bin_path, signature_path, metadata_path, identity = _cache_inputs(tmp_path)
    write_cache_metadata_atomic(
        metadata_path,
        identity,
        cache_fingerprint(identity),
        bin_path,
        signature_path,
    )

    def reject(_path):
        raise ValueError("corrupt")

    decision = inspect_cache(
        bin_path, signature_path, metadata_path, identity, reject, lambda path: None
    )

    assert decision.hit is False
    assert decision.reason == "invalid-bin"


def test_cache_signature_validator_failure_becomes_invalid_signatures_miss(tmp_path):
    _, _, bin_path, signature_path, metadata_path, identity = _cache_inputs(tmp_path)
    write_cache_metadata_atomic(
        metadata_path,
        identity,
        cache_fingerprint(identity),
        bin_path,
        signature_path,
    )

    def reject(_path):
        raise ValueError("corrupt signatures")

    decision = inspect_cache(
        bin_path, signature_path, metadata_path, identity, lambda path: None, reject
    )

    assert decision.hit is False
    assert decision.reason == "invalid-signatures"


def test_cache_rejects_bin_from_a_different_publication_generation(tmp_path):
    _, _, bin_path, signature_path, metadata_path, identity = _cache_inputs(tmp_path)
    write_cache_metadata_atomic(
        metadata_path,
        identity,
        cache_fingerprint(identity),
        bin_path,
        signature_path,
    )
    bin_path.write_bytes(b"replacement-bin-after-sidecar-failure")
    validated = []

    decision = inspect_cache(
        bin_path,
        signature_path,
        metadata_path,
        identity,
        lambda path: validated.append(path),
        lambda path: validated.append(path),
    )

    assert decision.hit is False
    assert decision.reason == "invalid-bin"
    assert validated == []


def test_cache_rejects_signatures_from_a_different_publication_generation(tmp_path):
    _, _, bin_path, signature_path, metadata_path, identity = _cache_inputs(tmp_path)
    write_cache_metadata_atomic(
        metadata_path,
        identity,
        cache_fingerprint(identity),
        bin_path,
        signature_path,
    )
    signature_path.write_bytes(b"replacement-signatures-after-bin-publication")
    validated = []

    decision = inspect_cache(
        bin_path,
        signature_path,
        metadata_path,
        identity,
        lambda path: validated.append(path),
        lambda path: validated.append(path),
    )

    assert decision.hit is False
    assert decision.reason == "invalid-signatures"
    assert validated == []


def test_cache_identity_changes_with_every_conversion_input(tmp_path):
    step, implementation, _, _, _, identity = _cache_inputs(tmp_path)

    step.write_bytes(b"STEP-B")
    assert build_cache_identity(
        step,
        [implementation],
        42,
        {"mode": "fixed", "points_per_face": 96},
        {"pointcloud": True},
    )["step_sha256"] != identity["step_sha256"]

    step.write_bytes(b"STEP-A")
    implementation.write_bytes(b"converter-v2")
    changed_implementation = build_cache_identity(
        step,
        [implementation],
        42,
        {"mode": "fixed", "points_per_face": 96},
        {
            "pointcloud": True,
            "parametric_surface": True,
            "cylindrical": True,
        },
    )
    assert cache_fingerprint(changed_implementation) != cache_fingerprint(identity)

    implementation.write_bytes(b"converter-v1")
    for seed, point_sampling, geometry_fields in (
        (
            7,
            {"mode": "fixed", "points_per_face": 96},
            {
                "pointcloud": True,
                "parametric_surface": True,
                "cylindrical": True,
            },
        ),
        (
            42,
            {"mode": "fixed", "points_per_face": 64},
            {
                "pointcloud": True,
                "parametric_surface": True,
                "cylindrical": True,
            },
        ),
        (
            42,
            {"mode": "fixed", "points_per_face": 96},
            {
                "pointcloud": True,
                "parametric_surface": False,
                "cylindrical": True,
            },
        ),
    ):
        changed = build_cache_identity(
            step,
            [implementation],
            seed,
            point_sampling,
            geometry_fields,
        )
        assert cache_fingerprint(changed) != cache_fingerprint(identity)


def test_implementation_identity_is_portable_but_ordered(tmp_path):
    left = tmp_path / "left"
    right = tmp_path / "right"
    left.mkdir()
    right.mkdir()
    for directory in (left, right):
        (directory / "part.step").write_bytes(b"STEP")
        (directory / "schema.json").write_bytes(b"schema")
        (directory / "converter.py").write_bytes(b"converter")

    common_args = (
        42,
        {"points_per_face": 96},
        {"pointcloud": True},
    )
    left_identity = build_cache_identity(
        left / "part.step",
        [left / "schema.json", left / "converter.py"],
        *common_args,
    )
    right_identity = build_cache_identity(
        right / "part.step",
        [right / "schema.json", right / "converter.py"],
        *common_args,
    )

    assert left_identity == right_identity
    assert [item["name"] for item in left_identity["implementations"]] == [
        "schema.json",
        "converter.py",
    ]
    reversed_identity = build_cache_identity(
        left / "part.step",
        [left / "converter.py", left / "schema.json"],
        *common_args,
    )
    assert cache_fingerprint(reversed_identity) != cache_fingerprint(left_identity)


def test_cache_identity_normalizes_nested_sequences_for_json_round_trip(tmp_path):
    step = tmp_path / "part.step"
    implementation = tmp_path / "converter.py"
    step.write_bytes(b"STEP")
    implementation.write_bytes(b"converter")
    identity = build_cache_identity(
        step,
        [implementation],
        42,
        {"candidate_grids": (20, 32, 48)},
        {"pointcloud": True},
    )
    metadata_path = tmp_path / "input.bin.meta.json"
    bin_path = tmp_path / "input.bin"
    signature_path = tmp_path / "input.face_signatures.json"
    bin_path.write_bytes(b"valid-bin")
    signature_path.write_bytes(b"valid-signatures")

    write_cache_metadata_atomic(
        metadata_path,
        identity,
        cache_fingerprint(identity),
        bin_path,
        signature_path,
    )

    assert identity["point_sampling"]["candidate_grids"] == [20, 32, 48]
    assert json.loads(metadata_path.read_text(encoding="utf-8"))["identity"] == identity
