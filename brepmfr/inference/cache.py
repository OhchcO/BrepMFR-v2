"""Content-validated cache identity for inference BIN files."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Callable, Dict, Mapping, Sequence


INFERENCE_BIN_CONTRACT_VERSION = 1
_HASH_CHUNK_SIZE = 1024 * 1024


@dataclass(frozen=True)
class CacheDecision:
    """Result of checking whether an inference BIN can be reused."""

    hit: bool
    fingerprint: str
    bin_path: Path
    metadata_path: Path
    reason: str


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a file's bytes."""
    digest = hashlib.sha256()
    with Path(path).resolve().open("rb") as handle:
        while True:
            chunk = handle.read(_HASH_CHUNK_SIZE)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def build_cache_identity(
    step_path: Path,
    implementation_paths: Sequence[Path],
    seed: int,
    point_sampling: Mapping[str, object],
    geometry_fields: Mapping[str, bool],
) -> Dict[str, object]:
    """Describe every content or setting that affects an inference BIN."""
    point_sampling_json = json.loads(
        json.dumps(dict(point_sampling), ensure_ascii=False)
    )
    geometry_fields_json = json.loads(
        json.dumps(dict(geometry_fields), ensure_ascii=False)
    )
    implementations = [
        {
            "name": Path(path).name,
            "sha256": sha256_file(Path(path)),
        }
        for path in implementation_paths
    ]
    return {
        "contract_version": INFERENCE_BIN_CONTRACT_VERSION,
        "step_sha256": sha256_file(Path(step_path)),
        "implementations": implementations,
        "seed": int(seed),
        "point_sampling": point_sampling_json,
        "geometry_fields": geometry_fields_json,
    }


def cache_fingerprint(identity: Mapping[str, object]) -> str:
    """Hash an identity using canonical UTF-8 JSON."""
    payload = json.dumps(
        dict(identity),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _decision(
    hit: bool,
    fingerprint: str,
    bin_path: Path,
    metadata_path: Path,
    reason: str,
) -> CacheDecision:
    return CacheDecision(
        hit=hit,
        fingerprint=fingerprint,
        bin_path=Path(bin_path),
        metadata_path=Path(metadata_path),
        reason=reason,
    )


def inspect_cache(
    bin_path: Path,
    metadata_path: Path,
    identity: Mapping[str, object],
    validate_bin_file: Callable[[Path], None],
    force_rebuild: bool = False,
) -> CacheDecision:
    """Return a cache hit only after identity and BIN validation succeed."""
    bin_path = Path(bin_path)
    metadata_path = Path(metadata_path)
    fingerprint = cache_fingerprint(identity)
    if force_rebuild:
        return _decision(False, fingerprint, bin_path, metadata_path, "forced")
    if not bin_path.is_file():
        return _decision(False, fingerprint, bin_path, metadata_path, "missing-bin")
    if not metadata_path.is_file():
        return _decision(
            False, fingerprint, bin_path, metadata_path, "missing-metadata"
        )

    try:
        document = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return _decision(
            False, fingerprint, bin_path, metadata_path, "invalid-metadata"
        )
    if not isinstance(document, dict) or set(document) != {
        "schema_version",
        "fingerprint",
        "identity",
    }:
        return _decision(
            False, fingerprint, bin_path, metadata_path, "invalid-metadata"
        )
    if document["schema_version"] != INFERENCE_BIN_CONTRACT_VERSION:
        return _decision(
            False, fingerprint, bin_path, metadata_path, "invalid-metadata"
        )
    if document["fingerprint"] != fingerprint or document["identity"] != dict(
        identity
    ):
        return _decision(
            False, fingerprint, bin_path, metadata_path, "identity-mismatch"
        )

    try:
        validate_bin_file(bin_path)
    except Exception:
        return _decision(False, fingerprint, bin_path, metadata_path, "invalid-bin")
    return _decision(True, fingerprint, bin_path, metadata_path, "hit")


def write_cache_metadata_atomic(
    path: Path,
    identity: Mapping[str, object],
    fingerprint: str,
) -> None:
    """Write and read back a cache sidecar before atomically publishing it."""
    path = Path(path)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    document = {
        "schema_version": INFERENCE_BIN_CONTRACT_VERSION,
        "fingerprint": fingerprint,
        "identity": dict(identity),
    }
    try:
        with temporary_path.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(document, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        written = json.loads(temporary_path.read_text(encoding="utf-8"))
        if written != document:
            raise ValueError("Cache metadata read-back did not match written content")
        temporary_path.replace(path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
