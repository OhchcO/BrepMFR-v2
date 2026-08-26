"""Versioned JSON documents for STEP inference results."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Dict, Mapping, Sequence

from brepmfr.inference.face_signature import validate_face_mapping_document


RESULT_SCHEMA_VERSION = "1.1"
ALLOWED_ERROR_STAGES = frozenset(
    {
        "arguments",
        "checkpoint",
        "cache",
        "conversion",
        "loading",
        "inference",
        "postprocess",
        "serialization",
    }
)


def _path_text(path: Path) -> str:
    return Path(path).resolve().as_posix()


def _reject_nonfinite(value: object) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("JSON result contains a non-finite float")
    if isinstance(value, Mapping):
        for nested in value.values():
            _reject_nonfinite(nested)
    elif isinstance(value, (list, tuple)):
        for nested in value:
            _reject_nonfinite(nested)


def build_success_document(
    step_path: Path,
    step_sha256: str,
    face_count: int,
    bin_path: Path,
    metadata_path: Path,
    cache_hit: bool,
    cache_fingerprint: str,
    checkpoint_path: Path,
    checkpoint_sha256: str,
    checkpoint_epoch: int,
    experiment_mode: str,
    device: str,
    num_classes: int,
    affinity_threshold: float,
    faces: Sequence[Mapping[str, object]],
    instances: Sequence[Mapping[str, object]],
    timings_seconds: Mapping[str, float],
    warnings: Sequence[str],
    face_mapping: Mapping[str, object],
) -> Dict[str, object]:
    """Build the canonical successful inference document."""
    validated_face_mapping = validate_face_mapping_document(
        face_mapping, expected_face_count=int(face_count)
    )
    document = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": "success",
        "input": {
            "step_path": _path_text(step_path),
            "step_sha256": str(step_sha256),
            "face_count": int(face_count),
        },
        "cache": {
            "bin_path": _path_text(bin_path),
            "metadata_path": _path_text(metadata_path),
            "hit": bool(cache_hit),
            "fingerprint": str(cache_fingerprint),
        },
        "model": {
            "checkpoint_path": _path_text(checkpoint_path),
            "checkpoint_sha256": str(checkpoint_sha256),
            "checkpoint_epoch": int(checkpoint_epoch),
            "experiment_mode": str(experiment_mode),
            "device": str(device),
            "num_classes": int(num_classes),
            "affinity_threshold": float(affinity_threshold),
        },
        "faces": [dict(face) for face in faces],
        "instances": [dict(instance) for instance in instances],
        "timings_seconds": {
            str(name): float(duration)
            for name, duration in timings_seconds.items()
        },
        "warnings": [str(warning) for warning in warnings],
        "face_mapping": validated_face_mapping,
    }
    _reject_nonfinite(document)
    return document


def build_error_document(
    stage: str,
    error: BaseException,
    step_path: Path,
) -> Dict[str, object]:
    """Build a concise error document without a traceback."""
    if stage not in ALLOWED_ERROR_STAGES:
        raise ValueError("Unsupported inference error stage: {0}".format(stage))
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": "error",
        "stage": stage,
        "error": {
            "type": type(error).__name__,
            "message": str(error),
        },
        "input": {"step_path": _path_text(step_path)},
    }


def write_json_atomic(path: Path, document: Mapping[str, object]) -> None:
    """Validate, read back, and atomically publish a UTF-8 JSON document."""
    _reject_nonfinite(document)
    intended = dict(document)
    payload = json.dumps(intended, ensure_ascii=False, indent=2) + "\n"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary_path.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        written = json.loads(temporary_path.read_text(encoding="utf-8"))
        if written != intended:
            raise ValueError("JSON result read-back did not match written content")
        temporary_path.replace(path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
