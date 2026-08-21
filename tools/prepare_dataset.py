"""Deterministically prepare one reusable MFRInstSeg BIN cache.

This module deliberately does not import the STEP converter at module import
time.  Training configuration and unit tests can therefore inspect splitting
and cache decisions without requiring PyOCC.  The converter is loaded only
after a new cache has been selected for conversion.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXTRACTOR_FILE = PROJECT_ROOT / "tools" / "AAGExtractor.py"
DEFAULT_SCHEMA_FILE = PROJECT_ROOT / "tools" / "schema.json"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.sample_mfrinstseg_1pct import (
    select_balanced_replacement_id,
    select_balanced_sample_ids,
)
from tools.geometry_features import POINT_FEATURE_DIM, POINT_FEATURE_ORDER
from utils.reproducibility import DEFAULT_SEED, derive_seed


FORMAT_VERSION = 1
CONVERTER_FORMAT_VERSION = 4
SPLIT_NAMES = ("train", "val", "test")
DEFAULT_CANDIDATE_GRIDS = (20, 32, 48, 64, 80)
REPLACEMENT_POLICY_VERSION = 1
BIN_CACHE_FORMAT_VERSION = 1


class PreparationError(RuntimeError):
    """Raised when a selected cache cannot be prepared completely."""


class PreparationMismatchError(PreparationError):
    """Raised rather than silently reusing an incompatible cache."""


@dataclass(frozen=True)
class RawSample:
    """The exact STEP and JSON files that define one raw model."""

    sample_id: str
    step_file: Path
    label_file: Path


@dataclass(frozen=True)
class PreparationConfig:
    """Immutable cache identity; model-consumption switches are excluded."""

    raw_root: Path
    prepared_root: Path
    extractor_file: Path
    schema_file: Path
    seed: int = DEFAULT_SEED
    split_ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1)
    sample_percent: float = 0.01
    point_sampling_mode: str = "area_adaptive"
    fixed_point_count: int = 64
    min_point_count: int = 32
    max_point_count: int = 96
    area_sampling_exponent: float = 0.5
    candidate_grids: Tuple[int, ...] = DEFAULT_CANDIDATE_GRIDS
    face_pointcloud: bool = True
    cylindrical_pos: bool = True
    rebuild: bool = False
    shared_bin_cache_root: Optional[Path] = None
    num_workers: int = 1
    donor_prepared_roots: Tuple[Path, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "raw_root", Path(self.raw_root))
        object.__setattr__(self, "prepared_root", Path(self.prepared_root))
        object.__setattr__(self, "extractor_file", Path(self.extractor_file))
        object.__setattr__(self, "schema_file", Path(self.schema_file))
        if self.shared_bin_cache_root is not None:
            object.__setattr__(
                self,
                "shared_bin_cache_root",
                Path(self.shared_bin_cache_root),
            )
        object.__setattr__(
            self,
            "donor_prepared_roots",
            tuple(Path(path) for path in self.donor_prepared_roots),
        )
        object.__setattr__(self, "split_ratios", tuple(self.split_ratios))
        object.__setattr__(self, "candidate_grids", tuple(self.candidate_grids))
        _validate_split_ratios(self.split_ratios)
        if not 0.0 < self.sample_percent <= 1.0:
            raise ValueError("sample_percent must be in (0, 1]")
        if self.point_sampling_mode not in ("fixed", "area_adaptive"):
            raise ValueError("point_sampling_mode must be 'fixed' or 'area_adaptive'")
        if not 1 <= self.fixed_point_count <= 96:
            raise ValueError("fixed_point_count must be in [1, 96]")
        if not 1 <= self.min_point_count <= self.max_point_count <= 96:
            raise ValueError("point counts must satisfy 1 <= min <= max <= 96")
        if self.area_sampling_exponent <= 0.0:
            raise ValueError("area_sampling_exponent must be positive")
        if not self.candidate_grids or any(grid < 2 for grid in self.candidate_grids):
            raise ValueError("candidate_grids must contain positive grid resolutions")
        if not (self.face_pointcloud and self.cylindrical_pos):
            raise ValueError(
                "automatic preparation always requires face_pointcloud and cylindrical_pos"
            )


@dataclass(frozen=True)
class PreparedDataset:
    """A fully validated prepared directory and its selected model IDs."""

    root: Path
    bin_root: Path
    train_ids: Tuple[str, ...]
    val_ids: Tuple[str, ...]
    test_ids: Tuple[str, ...]
    reused: bool


def _validate_split_ratios(split_ratios: Sequence[float]) -> None:
    if len(split_ratios) != 3 or any(ratio <= 0.0 for ratio in split_ratios):
        raise ValueError("split_ratios must contain exactly three positive values")
    if not math.isclose(sum(split_ratios), 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("split_ratios must sum to 1.0")


def _validate_sample_ids(
    sample_ids: Sequence[str], allow_empty: bool = False
) -> Tuple[str, ...]:
    normalized = tuple(sample_ids)
    if (not allow_empty and not normalized) or any(
        not sample_id or sample_id != sample_id.strip() for sample_id in normalized
    ):
        raise ValueError("sample IDs must be non-empty")
    if len(set(normalized)) != len(normalized):
        raise ValueError("sample IDs must be unique")
    return normalized


def discover_raw_samples(raw_root: Path) -> Dict[str, RawSample]:
    """Scan and strictly pair ``steps/<id>.step`` with ``labels/<id>.json``."""
    raw_root = Path(raw_root)
    steps_root = raw_root / "steps"
    labels_root = raw_root / "labels"
    if not steps_root.is_dir() or not labels_root.is_dir():
        raise FileNotFoundError("raw dataset must contain steps/ and labels/ directories")
    step_files = {}
    for path in steps_root.iterdir():
        if not path.is_file() or path.suffix.lower() not in (".step", ".stp"):
            continue
        if path.stem in step_files:
            raise ValueError("Duplicate STEP stem: {0}".format(path.stem))
        step_files[path.stem] = path
    label_files = {}
    for path in labels_root.iterdir():
        if not path.is_file() or path.suffix.lower() != ".json":
            continue
        if path.stem in label_files:
            raise ValueError("Duplicate label stem: {0}".format(path.stem))
        label_files[path.stem] = path
    if not step_files:
        raise ValueError("No STEP files were found under {0}".format(steps_root))
    if set(step_files) != set(label_files):
        raise ValueError("STEP and label stems must match exactly")
    return {
        sample_id: RawSample(sample_id, step_files[sample_id], label_files[sample_id])
        for sample_id in sorted(step_files)
    }


def _read_official_manifest(path: Path) -> Tuple[str, ...]:
    if not path.is_file():
        raise FileNotFoundError("Split manifest not found: {0}".format(path))
    sample_ids = tuple(
        line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    )
    if any(not sample_id or sample_id != sample_id.strip() for sample_id in sample_ids):
        raise ValueError("sample IDs must be non-empty")
    return sample_ids


def _stable_unique(sample_ids: Sequence[str]) -> Tuple[str, ...]:
    seen = set()
    return tuple(
        sample_id
        for sample_id in sample_ids
        if not (sample_id in seen or seen.add(sample_id))
    )


def _official_splits(raw_root: Path, known_ids: Tuple[str, ...]) -> Dict[str, Tuple[str, ...]]:
    manifests = {name: Path(raw_root) / (name + ".txt") for name in SPLIT_NAMES}
    present = [name for name, path in manifests.items() if path.exists()]
    if present and len(present) != len(SPLIT_NAMES):
        raise ValueError("Official split manifests must include all three train/val/test files")
    if not present:
        return {}
    raw_splits = {
        name: _read_official_manifest(path) for name, path in manifests.items()
    }
    unique_splits = {
        name: _stable_unique(raw_splits[name]) for name in SPLIT_NAMES
    }

    # Some MFRInstSeg releases contain repeated lines and a few model IDs shared
    # across official manifests. Keep the highest-value evaluation membership
    # (test, then val, then train), while preserving manifest order otherwise.
    claimed = set()
    splits = {}
    cross_split_removed = 0
    for name in ("test", "val", "train"):
        retained = tuple(
            sample_id for sample_id in unique_splits[name] if sample_id not in claimed
        )
        cross_split_removed += len(unique_splits[name]) - len(retained)
        splits[name] = _validate_sample_ids(retained, allow_empty=True)
        claimed.update(retained)

    within_split_removed = sum(
        len(raw_splits[name]) - len(unique_splits[name]) for name in SPLIT_NAMES
    )
    if within_split_removed or cross_split_removed:
        warnings.warn(
            "Cleaned official split manifests: removed {0} repeated line(s) "
            "and {1} cross-split duplicate assignment(s) using "
            "test > val > train precedence.".format(
                within_split_removed, cross_split_removed
            ),
            RuntimeWarning,
            stacklevel=2,
        )

    all_ids = tuple(sample_id for name in SPLIT_NAMES for sample_id in splits[name])
    known_set = set(known_ids)
    listed_set = set(all_ids)
    unknown_ids = listed_set - known_set
    if unknown_ids:
        raise ValueError(
            "Official split manifests contain {0} unknown model ID(s)".format(
                len(unknown_ids)
            )
        )

    # An official evaluation membership should never be guessed. Models omitted
    # from the manifests are therefore added only to training, deterministically.
    missing_ids = tuple(sample_id for sample_id in known_ids if sample_id not in listed_set)
    if missing_ids:
        splits["train"] = splits["train"] + missing_ids
        warnings.warn(
            "Official split manifests omitted {0} discovered model(s); assigned "
            "them to train only.".format(len(missing_ids)),
            RuntimeWarning,
            stacklevel=2,
        )
    return splits


def _generated_splits(sample_ids: Tuple[str, ...], split_ratios: Sequence[float], seed: int) -> Dict[str, Tuple[str, ...]]:
    shuffled = list(sorted(sample_ids))
    random.Random(derive_seed(seed, "model_split")).shuffle(shuffled)
    expected = [len(shuffled) * ratio for ratio in split_ratios]
    counts = [int(math.floor(value)) for value in expected]
    for index in sorted(
        range(3), key=lambda item: (expected[item] - counts[item], -item), reverse=True
    )[: len(shuffled) - sum(counts)]:
        counts[index] += 1
    if len(shuffled) >= len(SPLIT_NAMES):
        for target in range(len(SPLIT_NAMES)):
            if counts[target] > 0:
                continue
            donor = max(range(len(SPLIT_NAMES)), key=lambda index: (counts[index], -index))
            counts[donor] -= 1
            counts[target] += 1
    boundary_one = counts[0]
    boundary_two = boundary_one + counts[1]
    return {
        "train": tuple(sorted(shuffled[:boundary_one])),
        "val": tuple(sorted(shuffled[boundary_one:boundary_two])),
        "test": tuple(sorted(shuffled[boundary_two:])),
    }


def resolve_model_splits(
    raw_root: Path,
    sample_ids: Sequence[str],
    split_ratios: Tuple[float, float, float],
    seed: int,
) -> Dict[str, Sequence[str]]:
    """Use complete official manifests, or deterministically split full models."""
    _validate_split_ratios(split_ratios)
    supplied_ids = _validate_sample_ids(sample_ids)
    discovered_ids = tuple(discover_raw_samples(raw_root))
    if set(supplied_ids) != set(discovered_ids):
        raise ValueError("sample_ids must contain every discovered raw model exactly once")
    official = _official_splits(raw_root, discovered_ids)
    return official or _generated_splits(supplied_ids, split_ratios, seed)


def select_split_subsets(
    raw_root: Path,
    splits: Mapping[str, Sequence[str]],
    sample_percent: float,
    seed: int,
) -> Dict[str, Sequence[str]]:
    """Balance train/val only; retain every test model exactly as assigned."""
    if not 0.0 < sample_percent <= 1.0:
        raise ValueError("sample_percent must be in (0, 1]")
    if set(splits) != set(SPLIT_NAMES):
        raise ValueError("splits must contain train, val, and test")
    normalized = {
        name: _validate_sample_ids(splits[name], allow_empty=True)
        for name in SPLIT_NAMES
    }
    combined = tuple(sample_id for name in SPLIT_NAMES for sample_id in normalized[name])
    if len(set(combined)) != len(combined):
        raise ValueError("splits must be model-level disjoint")
    known_ids = set(discover_raw_samples(raw_root))
    if set(combined) != known_ids:
        raise ValueError("splits must contain every discovered raw model exactly once")

    selected: Dict[str, Sequence[str]] = {}
    for split_name in ("train", "val"):
        ids = normalized[split_name]
        if sample_percent == 1.0:
            selected[split_name] = ids
        else:
            selected[split_name] = (
                select_balanced_sample_ids(
                    Path(raw_root),
                    ids,
                    sample_percent,
                    derive_seed(seed, "balanced_subset", split_name),
                    split_name,
                )
                if ids
                else ()
            )
    selected["test"] = normalized["test"]
    return selected


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative_path(root: Path, path: Path) -> str:
    return path.relative_to(root).as_posix()


def _external_file_fingerprint(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError("Required converter input is missing: {0}".format(path))
    return {"path": str(path), "sha256": _sha256(path)}


def _conversion_options(config: PreparationConfig) -> dict:
    return {
        "point_sampling_mode": config.point_sampling_mode,
        "fixed_point_count": config.fixed_point_count,
        "min_point_count": config.min_point_count,
        "max_point_count": config.max_point_count,
        "area_sampling_exponent": config.area_sampling_exponent,
        "candidate_grids": list(config.candidate_grids),
        "face_pointcloud": True,
        "parametric_surface": True,
        "cylindrical_pos": True,
        "base_seed": config.seed,
        "point_feature_dimension": POINT_FEATURE_DIM,
    }


def build_preparation_manifest(
    config: PreparationConfig,
    selected_ids: Mapping[str, Sequence[str]],
    samples: Optional[Mapping[str, RawSample]] = None,
) -> dict:
    """Build the stable request fingerprint; content hashes never use mtimes."""
    if set(selected_ids) != set(SPLIT_NAMES):
        raise ValueError("selected_ids must contain train, val, and test")
    samples = dict(samples or discover_raw_samples(config.raw_root))
    normalized = {
        name: list(_validate_sample_ids(selected_ids[name], allow_empty=True))
        for name in SPLIT_NAMES
    }
    return {
        "format_version": FORMAT_VERSION,
        "converter_format_version": CONVERTER_FORMAT_VERSION,
        "base_seed": config.seed,
        "split_ratios": list(config.split_ratios),
        "sample_percent": config.sample_percent,
        "source_files": {
            sample_id: {
                "step": {
                    "path": _relative_path(config.raw_root, sample.step_file),
                    "sha256": _sha256(sample.step_file),
                },
                "label": {
                    "path": _relative_path(config.raw_root, sample.label_file),
                    "sha256": _sha256(sample.label_file),
                },
            }
            for sample_id, sample in sorted(samples.items())
        },
        "converter_sources": {
            "extractor": _external_file_fingerprint(config.extractor_file),
            "schema": _external_file_fingerprint(config.schema_file),
        },
        "selected_ids": normalized,
        "output_features": {
            "face_pointcloud": True,
            "parametric_surface": True,
            "cylindrical_pos": True,
        },
        "point_feature_contract": {
            "dimension": POINT_FEATURE_DIM,
            "order": list(POINT_FEATURE_ORDER),
            "uv_range": [-1.0, 1.0],
            "characteristic_length": (
                "sampled_full_part_bounding_box_diagonal"
            ),
        },
        "conversion_options": _conversion_options(config),
        "effective_selected_ids": {
            name: list(normalized[name]) for name in SPLIT_NAMES
        },
        "replacement_policy_version": REPLACEMENT_POLICY_VERSION,
        "conversion_exclusions": [],
        "conversion": {},
        "successful_ids": {name: [] for name in SPLIT_NAMES},
        "failed_ids": [],
    }


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        temporary.write_text(text, encoding="utf-8")
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_write_json(path: Path, document: object) -> None:
    _atomic_write_text(path, json.dumps(document, indent=2, sort_keys=True) + "\n")


def _write_split_manifests(prepared_root: Path, selected_ids: Mapping[str, Sequence[str]]) -> None:
    for split_name in SPLIT_NAMES:
        sample_ids = tuple(selected_ids[split_name])
        _atomic_write_text(
            prepared_root / (split_name + ".txt"),
            "\n".join(sample_ids) + ("\n" if sample_ids else ""),
        )


def _request_fingerprint(manifest: Mapping[str, object]) -> dict:
    return {
        key: manifest[key]
        for key in (
            "format_version",
            "converter_format_version",
            "base_seed",
            "split_ratios",
            "sample_percent",
            "source_files",
            "converter_sources",
            "selected_ids",
            "output_features",
            "point_feature_contract",
            "conversion_options",
        )
    }


def _bin_cache_fingerprint(manifest: Mapping[str, object]) -> dict:
    """Return only inputs that can change one model's serialized BIN."""
    return {
        key: manifest[key]
        for key in (
            "converter_format_version",
            "base_seed",
            "source_files",
            "converter_sources",
            "output_features",
            "point_feature_contract",
            "conversion_options",
        )
    }


def _compatible_legacy_bin_cache_fingerprint(
    existing: object, expected: Mapping[str, object]
) -> bool:
    """Accept an old selected-source fingerprint only as a verified subset."""
    if not isinstance(existing, Mapping) or set(existing) != set(expected):
        return False
    for key, expected_value in expected.items():
        if key != "source_files" and existing[key] != expected_value:
            return False
    existing_sources = existing.get("source_files")
    expected_sources = expected.get("source_files")
    if not isinstance(existing_sources, Mapping) or not existing_sources:
        return False
    if not isinstance(expected_sources, Mapping):
        return False
    return all(
        sample_id in expected_sources
        and expected_sources[sample_id] == source_fingerprint
        for sample_id, source_fingerprint in existing_sources.items()
    )


def _active_bin_root(config: PreparationConfig) -> Path:
    return Path(config.shared_bin_cache_root or config.prepared_root)


def _ensure_shared_cache_manifest(
    shared_root: Path, request: Mapping[str, object]
) -> None:
    """Create or fail-closed validate the percentage-independent cache."""
    shared_root.mkdir(parents=True, exist_ok=True)
    manifest_path = shared_root / "bin_cache_manifest.json"
    expected = {
        "bin_cache_format_version": BIN_CACHE_FORMAT_VERSION,
        "fingerprint": _bin_cache_fingerprint(request),
    }
    if manifest_path.exists():
        existing = _read_manifest(manifest_path)
        if existing == expected:
            return
        compatible_legacy = (
            existing.get("bin_cache_format_version") == BIN_CACHE_FORMAT_VERSION
            and _compatible_legacy_bin_cache_fingerprint(
                existing.get("fingerprint"), expected["fingerprint"]
            )
        )
        if compatible_legacy:
            _atomic_write_json(manifest_path, expected)
            return
        raise PreparationMismatchError(
            "Shared BIN cache fingerprint differs; choose a new shared cache path"
        )
    _atomic_write_json(manifest_path, expected)


def _selected_ids_flat(
    selected_ids: Mapping[str, Sequence[str]]
) -> Tuple[str, ...]:
    return tuple(
        sample_id
        for split_name in SPLIT_NAMES
        for sample_id in selected_ids[split_name]
    )


def _import_compatible_donor_bins(
    config: PreparationConfig,
    request: Mapping[str, object],
    selected_ids: Mapping[str, Sequence[str]],
    converter: object,
    requirements: object,
    point_sampling: object,
) -> None:
    """Hard-link compatible selected donor BINs; never copy or overwrite."""
    if config.shared_bin_cache_root is None or not config.donor_prepared_roots:
        return
    destination = Path(config.shared_bin_cache_root) / "bin"
    destination.mkdir(parents=True, exist_ok=True)
    expected_fingerprint = _bin_cache_fingerprint(request)
    selected = _selected_ids_flat(selected_ids)
    for donor_root in config.donor_prepared_roots:
        donor_manifest_path = donor_root / "preparation_manifest.json"
        try:
            donor_manifest = _read_manifest(donor_manifest_path)
        except (OSError, PreparationMismatchError) as error:
            warnings.warn(
                "Ignoring unreadable BIN donor {0}: {1}".format(
                    donor_root, error
                ),
                RuntimeWarning,
                stacklevel=2,
            )
            continue
        if _bin_cache_fingerprint(donor_manifest) != expected_fingerprint:
            warnings.warn(
                "Ignoring incompatible BIN donor: {0}".format(donor_root),
                RuntimeWarning,
                stacklevel=2,
            )
            continue
        for sample_id in selected:
            target = destination / (sample_id + ".bin")
            if target.exists():
                continue
            source = donor_root / "bin" / (sample_id + ".bin")
            if not source.is_file():
                continue
            try:
                converter.validate_bin(
                    source,
                    expected_nodes=None,
                    requirements=requirements,
                    point_sampling=point_sampling,
                )
                os.link(str(source), str(target))
            except (OSError, ValueError) as error:
                warnings.warn(
                    "Could not import donor BIN {0}: {1}".format(source, error),
                    RuntimeWarning,
                    stacklevel=2,
                )


def _effective_selected_ids(manifest: Mapping[str, object]) -> dict:
    """Return effective IDs, falling back to legacy requested selections."""
    source = (
        manifest["effective_selected_ids"]
        if "effective_selected_ids" in manifest
        else manifest["selected_ids"]
    )
    if not isinstance(source, Mapping) or set(source) != set(SPLIT_NAMES):
        raise ValueError(
            "effective selected IDs must contain train, val, and test"
        )
    return {
        name: _validate_sample_ids(source[name], allow_empty=True)
        for name in SPLIT_NAMES
    }


def _conversion_exclusions_path(prepared_root: Path) -> Path:
    return prepared_root / "conversion_exclusions.json"


def _read_manifest(path: Path) -> dict:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise PreparationMismatchError("Prepared manifest cannot be read: {0}".format(error)) from error
    if not isinstance(document, dict):
        raise PreparationMismatchError("Prepared manifest must be a JSON object")
    return document


def _validate_prepared_outputs(
    prepared_root: Path,
    manifest: Mapping[str, object],
    converter: object,
    requirements: object,
    point_sampling: object,
    bin_root: Optional[Path] = None,
) -> list:
    """Return structured output failures using Task 3's OCC-free BIN validator.

    Task 3 extends its published ``validate_bin`` as
    ``validate_bin(path, expected_nodes=None, requirements=...,
    point_sampling=...)``. It may import DGL/Torch but must not import PyOCC;
    cache reuse needs this feature-aware validation before conversion starts.
    """
    failures = []
    try:
        successful = manifest["successful_ids"]
        selected = _effective_selected_ids(manifest)
        failed_ids = manifest["failed_ids"]
        if failed_ids or not isinstance(successful, Mapping) or not isinstance(selected, Mapping):
            return [{"sample_id": None, "reason": "incomplete_manifest"}]
        for split_name in SPLIT_NAMES:
            selected_ids = tuple(selected[split_name])
            if tuple(successful[split_name]) != selected_ids:
                failures.append(
                    {"split": split_name, "sample_id": None, "reason": "incomplete_success_manifest"}
                )
                continue
            split_file = prepared_root / (split_name + ".txt")
            actual_ids = tuple(
                line.strip() for line in split_file.read_text(encoding="utf-8").splitlines() if line.strip()
            )
            if actual_ids != selected_ids:
                failures.append(
                    {"split": split_name, "sample_id": None, "reason": "split_manifest_mismatch"}
                )
                continue
            for sample_id in selected_ids:
                bin_file = Path(bin_root or prepared_root) / "bin" / (
                    sample_id + ".bin"
                )
                if not bin_file.is_file():
                    failures.append(
                        {"split": split_name, "sample_id": sample_id, "reason": "missing_or_invalid_bin"}
                    )
                    continue
                try:
                    converter.validate_bin(
                        bin_file,
                        expected_nodes=None,
                        requirements=requirements,
                        point_sampling=point_sampling,
                    )
                except Exception as error:
                    failures.append(
                        {
                            "split": split_name,
                            "sample_id": sample_id,
                            "reason": "missing_or_invalid_bin",
                            "exception_type": type(error).__name__,
                            "message": str(error),
                        }
                    )
    except (KeyError, OSError, TypeError, UnicodeError) as error:
        failures.append(
            {"sample_id": None, "reason": "incomplete_manifest", "message": str(error)}
        )
    return failures


def _validate_successful_outputs(
    prepared_root: Path,
    successful_ids: Mapping[str, Sequence[str]],
    converter: object,
    requirements: object,
    point_sampling: object,
    bin_root: Optional[Path] = None,
) -> list:
    """Validate only the successful portion of a recoverable failed cache."""
    failures = []
    try:
        if not isinstance(successful_ids, Mapping) or set(successful_ids) != set(
            SPLIT_NAMES
        ):
            return [{"sample_id": None, "reason": "incomplete_success_manifest"}]
        normalized = {
            name: _validate_sample_ids(successful_ids[name], allow_empty=True)
            for name in SPLIT_NAMES
        }
        for split_name in SPLIT_NAMES:
            split_file = prepared_root / (split_name + ".txt")
            actual_ids = tuple(
                line.strip()
                for line in split_file.read_text(encoding="utf-8").splitlines()
                if line.strip()
            )
            if actual_ids != normalized[split_name]:
                failures.append(
                    {
                        "split": split_name,
                        "sample_id": None,
                        "reason": "split_manifest_mismatch",
                    }
                )
                continue
            for sample_id in normalized[split_name]:
                bin_file = Path(bin_root or prepared_root) / "bin" / (sample_id + ".bin")
                if not bin_file.is_file():
                    failures.append(
                        {
                            "split": split_name,
                            "sample_id": sample_id,
                            "reason": "missing_or_invalid_bin",
                        }
                    )
                    continue
                try:
                    converter.validate_bin(
                        bin_file,
                        expected_nodes=None,
                        requirements=requirements,
                        point_sampling=point_sampling,
                    )
                except Exception as error:
                    failures.append(
                        {
                            "split": split_name,
                            "sample_id": sample_id,
                            "reason": "missing_or_invalid_bin",
                            "exception_type": type(error).__name__,
                            "message": str(error),
                        }
                    )
    except (KeyError, OSError, TypeError, UnicodeError, ValueError) as error:
        failures.append(
            {
                "sample_id": None,
                "reason": "incomplete_success_manifest",
                "message": str(error),
            }
        )
    return failures


def _prepared_dataset(
    prepared_root: Path,
    bin_root: Path,
    selected_ids: Mapping[str, Sequence[str]],
    reused: bool,
) -> PreparedDataset:
    return PreparedDataset(
        root=prepared_root,
        bin_root=bin_root,
        train_ids=tuple(selected_ids["train"]),
        val_ids=tuple(selected_ids["val"]),
        test_ids=tuple(selected_ids["test"]),
        reused=reused,
    )


def _load_converter():
    """Load the future PyOCC-dependent converter only when conversion is needed."""
    from tools import mfrinstseg_to_bin

    return mfrinstseg_to_bin


def _converter_contract(config: PreparationConfig, converter: object) -> tuple[object, object, object]:
    """Construct Task 3's typed options and feature requirements lazily."""
    point_sampling = converter.PointSamplingConfig(
        mode=config.point_sampling_mode,
        fixed_count=config.fixed_point_count,
        min_count=config.min_point_count,
        max_count=config.max_point_count,
        area_exponent=config.area_sampling_exponent,
        candidate_grids=config.candidate_grids,
    )
    options = converter.ConversionOptions(
        seed=config.seed,
        point_sampling=point_sampling,
        include_face_pointcloud=True,
        include_cylindrical_pos=True,
        include_parametric_surface=True,
    )
    requirements = converter.BinRequirements(
        require_instance=True,
        require_pointcloud=True,
        require_cylindrical=True,
        require_parametric_surface=True,
    )
    return point_sampling, options, requirements


def _normalise_conversion_result(result: object, selected_ids: Mapping[str, Sequence[str]]) -> tuple[dict, list, dict]:
    if isinstance(result, Mapping):
        successful = result.get("successful_ids", {})
        failed = result.get("failed_ids", [])
        conversion = result.get("conversion", {})
    else:
        successful = getattr(result, "successful_ids", {})
        failed = getattr(result, "failed_ids", [])
        conversion = getattr(result, "conversion", {})
    if not isinstance(successful, Mapping) or not isinstance(failed, (list, tuple)):
        raise PreparationError("convert_selected_samples returned an invalid result")
    normalized_success = {
        name: list(successful.get(name, ())) for name in SPLIT_NAMES
    }
    expected_success = {name: list(selected_ids[name]) for name in SPLIT_NAMES}
    if not failed and normalized_success != expected_success:
        failed = [{"sample_id": None, "message": "converter did not report every selected sample"}]
    return normalized_success, list(failed), dict(conversion) if isinstance(conversion, Mapping) else {}


def _replaceable_failure(
    failure: Mapping[str, object],
    requested_ids: Mapping[str, Sequence[str]],
) -> bool:
    """Return whether one structured failure may be replaced safely."""
    if not isinstance(failure, Mapping):
        return False
    split_name = failure.get("split")
    sample_id = failure.get("sample_id")
    return (
        split_name in ("train", "val")
        and isinstance(sample_id, str)
        and bool(sample_id)
        and sample_id in requested_ids.get(split_name, ())
    )


def _merge_conversion_counts(
    accumulated: Optional[Mapping[str, object]],
    addition: Optional[Mapping[str, object]],
) -> dict:
    """Accumulate numeric converter counters without inventing a schema."""
    merged = dict(accumulated or {})
    for key, value in dict(addition or {}).items():
        previous = merged.get(key)
        if (
            isinstance(previous, (int, float))
            and not isinstance(previous, bool)
            and isinstance(value, (int, float))
            and not isinstance(value, bool)
        ):
            merged[key] = previous + value
        elif key not in merged:
            merged[key] = value
    return merged


def _objective_document(objective: Sequence[object]) -> dict:
    return {
        "uncovered": int(objective[0]),
        "coefficient_of_variation": float(objective[1]),
        "spread": int(objective[2]),
    }


def _recover_replaceable_failures(
    config: PreparationConfig,
    model_splits: Mapping[str, Sequence[str]],
    requested_ids: Mapping[str, Sequence[str]],
    successful_ids: Mapping[str, Sequence[str]],
    failures: Sequence[Mapping[str, object]],
    converter: object,
    paths: object,
    options: object,
    existing_conversion: Optional[Mapping[str, object]] = None,
) -> tuple[dict, list, list, dict]:
    """Replace deterministic Train/Val conversion failures one model at a time."""
    requested = {
        name: tuple(_validate_sample_ids(requested_ids[name], allow_empty=True))
        for name in SPLIT_NAMES
    }
    effective = {
        name: set(_validate_sample_ids(successful_ids.get(name, ()), allow_empty=True))
        for name in SPLIT_NAMES
    }
    unresolved = list(failures)
    conversion = dict(existing_conversion or {})
    audit = []

    # Fail closed before converting anything if the initial result is malformed,
    # affects Test, or reports IDs outside the submitted request.
    if any(not _replaceable_failure(failure, requested) for failure in unresolved):
        return (
            {name: sorted(effective[name]) for name in SPLIT_NAMES},
            unresolved,
            audit,
            conversion,
        )
    for split_name in SPLIT_NAMES:
        if not effective[split_name].issubset(requested[split_name]):
            unresolved.append(
                {
                    "split": split_name,
                    "sample_id": None,
                    "reason": "unexpected_success_id",
                }
            )
    failed_by_split = {
        name: {
            failure["sample_id"]
            for failure in unresolved
            if failure.get("split") == name
        }
        for name in SPLIT_NAMES
    }
    for split_name in SPLIT_NAMES:
        accounted = effective[split_name] | failed_by_split[split_name]
        if accounted != set(requested[split_name]):
            unresolved.append(
                {
                    "split": split_name,
                    "sample_id": None,
                    "reason": "incomplete_conversion_result",
                }
            )
    if any(not _replaceable_failure(failure, requested) for failure in unresolved):
        return (
            {name: sorted(effective[name]) for name in SPLIT_NAMES},
            unresolved,
            audit,
            conversion,
        )

    pending = list(unresolved)
    unresolved = []
    excluded = set(
        sample_id
        for name in SPLIT_NAMES
        for sample_id in requested[name]
    )
    excluded.update(
        failure["sample_id"] for failure in pending if failure.get("sample_id")
    )
    attempts = {"train": 0, "val": 0}

    while pending:
        current = pending.pop(0)
        split_name = current["split"]
        attempts[split_name] += 1
        candidates = [
            sample_id
            for sample_id in model_splits[split_name]
            if sample_id not in excluded and sample_id not in effective[split_name]
        ]
        if not candidates:
            unresolved.append(
                dict(current, reason="replacement_candidates_exhausted")
            )
            unresolved.extend(pending)
            break

        selection = select_balanced_replacement_id(
            config.raw_root,
            successful_ids=sorted(effective[split_name]),
            candidate_ids=candidates,
            base_seed=config.seed,
            split_name=split_name,
            attempt=attempts[split_name],
        )
        replacement_id = selection.sample_id
        excluded.add(replacement_id)
        replacement_batch = {name: [] for name in SPLIT_NAMES}
        replacement_batch[split_name] = [replacement_id]
        converter_error = None
        try:
            result = converter.convert_selected_samples(
                replacement_batch,
                paths,
                config.extractor_file,
                config.schema_file,
                options,
                False,
            )
            replacement_success, replacement_failures, added_conversion = (
                _normalise_conversion_result(result, replacement_batch)
            )
            conversion = _merge_conversion_counts(conversion, added_conversion)
        except Exception as error:
            converter_error = {
                "sample_id": None,
                "exception_type": type(error).__name__,
                "message": str(error),
            }
            replacement_success = {name: [] for name in SPLIT_NAMES}
            replacement_failures = [converter_error]

        succeeded = (
            replacement_success[split_name] == [replacement_id]
            and all(
                not replacement_success[name]
                for name in SPLIT_NAMES
                if name != split_name
            )
            and not replacement_failures
        )
        audit.append(
            {
                "split": split_name,
                "failed_sample_id": current["sample_id"],
                "replacement_sample_id": replacement_id,
                "exception_type": current.get("exception_type"),
                "message": current.get("message"),
                "attempt": attempts[split_name],
                "balance_before": _objective_document(selection.balance_before),
                "balance_after": _objective_document(selection.balance_after),
                "replacement_succeeded": succeeded,
            }
        )
        if succeeded:
            effective[split_name].add(replacement_id)
            continue

        if converter_error is not None:
            unresolved.append(converter_error)
            unresolved.extend(pending)
            break

        if (
            not any(replacement_success[name] for name in SPLIT_NAMES)
            and len(replacement_failures) == 1
            and _replaceable_failure(replacement_failures[0], replacement_batch)
        ):
            pending.insert(0, replacement_failures[0])
            continue

        unresolved.extend(replacement_failures)
        if not replacement_failures:
            unresolved.append(
                {
                    "split": split_name,
                    "sample_id": replacement_id,
                    "reason": "invalid_replacement_result",
                }
            )
        unresolved.extend(pending)
        break

    for split_name in SPLIT_NAMES:
        if len(effective[split_name]) != len(requested[split_name]):
            if not unresolved:
                unresolved.append(
                    {
                        "split": split_name,
                        "sample_id": None,
                        "reason": "replacement_count_mismatch",
                    }
                )

    return (
        {name: sorted(effective[name]) for name in SPLIT_NAMES},
        unresolved,
        audit,
        conversion,
    )


def ensure_prepared_dataset(config: PreparationConfig) -> PreparedDataset:
    """Reuse a verified exact cache or atomically prepare it in one converter call."""
    print("[prepare] Discovering raw samples...", flush=True)
    samples = discover_raw_samples(config.raw_root)
    print("[prepare] Found {0} raw samples. Resolving splits...".format(len(samples)), flush=True)
    model_splits = resolve_model_splits(config.raw_root, tuple(samples), config.split_ratios, config.seed)
    print("[prepare] Selecting subsets ({0:.0%})...".format(config.sample_percent), flush=True)
    selected_ids = select_split_subsets(config.raw_root, model_splits, config.sample_percent, config.seed)
    total_selected = sum(len(v) for v in selected_ids.values())
    print("[prepare] Selected {0} samples. Building manifest...".format(total_selected), flush=True)
    request = build_preparation_manifest(config, selected_ids, samples=samples)
    manifest_path = config.prepared_root / "preparation_manifest.json"
    bin_root = _active_bin_root(config)
    if config.shared_bin_cache_root is not None:
        _ensure_shared_cache_manifest(bin_root, request)

    if manifest_path.exists():
        try:
            existing = _read_manifest(manifest_path)
            request_matches = _request_fingerprint(existing) == _request_fingerprint(request)
        except (KeyError, TypeError, PreparationMismatchError):
            request_matches = False
        if request_matches:
            print("[prepare] Manifest matches. Loading converter...", flush=True)
            converter = _load_converter()
            print("[prepare] Converter loaded. Validating cache...", flush=True)
            point_sampling, options, requirements = _converter_contract(config, converter)
            cache_failures = _validate_prepared_outputs(
                config.prepared_root,
                existing,
                converter,
                requirements,
                point_sampling,
                bin_root=bin_root,
            )
            if not cache_failures:
                return _prepared_dataset(
                    config.prepared_root,
                    bin_root,
                    _effective_selected_ids(existing),
                    reused=True,
                )
            if not config.rebuild:
                existing_failures = existing.get("failed_ids", [])
                existing_successful = existing.get("successful_ids", {})
                recoverable = (
                    isinstance(existing_failures, list)
                    and bool(existing_failures)
                    and all(
                        _replaceable_failure(failure, selected_ids)
                        for failure in existing_failures
                    )
                    and not _validate_successful_outputs(
                        config.prepared_root,
                        existing_successful,
                        converter,
                        requirements,
                        point_sampling,
                        bin_root=bin_root,
                    )
                )
                if recoverable:
                    paths = converter.ConversionPaths(
                        config.raw_root,
                        config.prepared_root,
                        config.prepared_root,
                        config.shared_bin_cache_root,
                    )
                    (
                        effective_ids,
                        unresolved,
                        exclusions,
                        conversion,
                    ) = _recover_replaceable_failures(
                        config,
                        model_splits,
                        selected_ids,
                        existing_successful,
                        existing_failures,
                        converter,
                        paths,
                        options,
                        existing_conversion=existing.get("conversion", {}),
                    )
                    recovered = dict(existing)
                    recovered["replacement_policy_version"] = (
                        REPLACEMENT_POLICY_VERSION
                    )
                    recovered["effective_selected_ids"] = {
                        name: list(effective_ids[name]) for name in SPLIT_NAMES
                    }
                    recovered["successful_ids"] = effective_ids
                    recovered["failed_ids"] = unresolved
                    previous_exclusions = existing.get("conversion_exclusions", [])
                    if not isinstance(previous_exclusions, list):
                        previous_exclusions = []
                    recovered["conversion_exclusions"] = (
                        list(previous_exclusions) + exclusions
                    )
                    recovered["conversion"] = conversion
                    _write_split_manifests(config.prepared_root, effective_ids)
                    if not unresolved:
                        unresolved.extend(
                            _validate_prepared_outputs(
                                config.prepared_root,
                                recovered,
                                converter,
                                requirements,
                                point_sampling,
                                bin_root=bin_root,
                            )
                        )
                        recovered["failed_ids"] = unresolved
                    _atomic_write_json(
                        config.prepared_root / "conversion_failures.json",
                        unresolved,
                    )
                    _atomic_write_json(
                        _conversion_exclusions_path(config.prepared_root),
                        recovered["conversion_exclusions"],
                    )
                    _atomic_write_json(manifest_path, recovered)
                    if unresolved:
                        raise PreparationError(
                            "Prepared conversion failed; inspect conversion_failures.json"
                        )
                    return _prepared_dataset(
                        config.prepared_root,
                        bin_root,
                        effective_ids,
                        reused=False,
                    )
                raise PreparationMismatchError(
                    "Prepared cache is incomplete; choose a new prepared directory or rebuild it"
                )
        elif not config.rebuild:
            raise PreparationMismatchError(
                "Prepared cache fingerprint differs; choose a new prepared directory or rebuild it"
            )

    config.prepared_root.mkdir(parents=True, exist_ok=True)
    _write_split_manifests(config.prepared_root, selected_ids)
    _atomic_write_json(config.prepared_root / "conversion_failures.json", [])
    _atomic_write_json(_conversion_exclusions_path(config.prepared_root), [])
    _atomic_write_json(manifest_path, request)

    failures: list = []
    exclusions: list = []
    try:
        print("[prepare] Loading converter (PyOCC import, may take a moment)...", flush=True)
        converter = _load_converter()
        print("[prepare] Converter loaded. Preparing options...", flush=True)
        point_sampling, options, requirements = _converter_contract(config, converter)
        print("[prepare] Importing donor bins...", flush=True)
        _import_compatible_donor_bins(
            config,
            request,
            selected_ids,
            converter,
            requirements,
            point_sampling,
        )
        print("[prepare] Starting BIN conversion ({0} workers)...".format(config.num_workers), flush=True)
        paths = converter.ConversionPaths(
            config.raw_root,
            config.prepared_root,
            config.prepared_root,
            config.shared_bin_cache_root,
        )
        result = converter.convert_selected_samples(
            selected_ids,
            paths,
            config.extractor_file,
            config.schema_file,
            options,
            config.rebuild and config.shared_bin_cache_root is None,
            num_workers=config.num_workers,
        )
        successful_ids, failures, conversion = _normalise_conversion_result(result, selected_ids)
        if failures:
            successful_ids, failures, exclusions, conversion = (
                _recover_replaceable_failures(
                    config,
                    model_splits,
                    selected_ids,
                    successful_ids,
                    failures,
                    converter,
                    paths,
                    options,
                    existing_conversion=conversion,
                )
            )
        request["effective_selected_ids"] = {
            name: list(successful_ids[name]) for name in SPLIT_NAMES
        }
        request["successful_ids"] = successful_ids
        request["failed_ids"] = failures
        _write_split_manifests(config.prepared_root, successful_ids)
        if not failures:
            validation_failures = _validate_prepared_outputs(
                config.prepared_root,
                request,
                converter,
                requirements,
                point_sampling,
                bin_root=bin_root,
            )
            failures.extend(validation_failures)
    except Exception as error:
        successful_ids = {name: [] for name in SPLIT_NAMES}
        conversion = {}
        failures = [{"sample_id": None, "exception_type": type(error).__name__, "message": str(error)}]

    request["conversion"] = conversion
    request["successful_ids"] = successful_ids
    request["failed_ids"] = failures
    request["conversion_exclusions"] = exclusions
    _atomic_write_json(config.prepared_root / "conversion_failures.json", failures)
    _atomic_write_json(_conversion_exclusions_path(config.prepared_root), exclusions)
    _atomic_write_json(manifest_path, request)
    if failures:
        raise PreparationError("Prepared conversion failed; inspect conversion_failures.json")
    return _prepared_dataset(
        config.prepared_root, bin_root, successful_ids, reused=False
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the standalone cache-preparation CLI used by launch scripts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dataset", type=Path, required=True)
    parser.add_argument("--prepared-dataset", type=Path, required=True)
    parser.add_argument("--shared-bin-cache", type=Path)
    parser.add_argument(
        "--donor-prepared-dataset",
        type=Path,
        nargs="*",
        default=[],
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Number of parallel threads for BIN conversion (default: 1, serial)",
    )
    parser.add_argument(
        "--extractor-file", type=Path, default=DEFAULT_EXTRACTOR_FILE
    )
    parser.add_argument("--schema-file", type=Path, default=DEFAULT_SCHEMA_FILE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--split-ratios",
        type=float,
        nargs=3,
        default=[0.8, 0.1, 0.1],
        metavar=("TRAIN", "VAL", "TEST"),
    )
    parser.add_argument("--sample-percent", type=float, default=0.01)
    parser.add_argument(
        "--point-sampling-mode",
        choices=("fixed", "area_adaptive"),
        default="area_adaptive",
    )
    parser.add_argument("--fixed-point-count", type=int, default=64)
    parser.add_argument("--min-point-count", type=int, default=32)
    parser.add_argument("--max-point-count", type=int, default=96)
    parser.add_argument("--area-sampling-exponent", type=float, default=0.5)
    parser.add_argument(
        "--candidate-grids",
        type=int,
        nargs="+",
        default=list(DEFAULT_CANDIDATE_GRIDS),
    )
    parser.add_argument("--rebuild", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    prepared = ensure_prepared_dataset(
        PreparationConfig(
            raw_root=args.raw_dataset,
            prepared_root=args.prepared_dataset,
            extractor_file=args.extractor_file,
            schema_file=args.schema_file,
            seed=args.seed,
            split_ratios=tuple(args.split_ratios),
            sample_percent=args.sample_percent,
            point_sampling_mode=args.point_sampling_mode,
            fixed_point_count=args.fixed_point_count,
            min_point_count=args.min_point_count,
            max_point_count=args.max_point_count,
            area_sampling_exponent=args.area_sampling_exponent,
            candidate_grids=tuple(args.candidate_grids),
            face_pointcloud=True,
            cylindrical_pos=True,
            rebuild=args.rebuild,
            shared_bin_cache_root=args.shared_bin_cache,
            num_workers=getattr(args, 'num_workers', 1),
            donor_prepared_roots=tuple(args.donor_prepared_dataset),
        )
    )
    print(
        json.dumps(
            {
                "prepared_dataset_path": str(prepared.root.resolve()),
                "bin_cache_path": str(prepared.bin_root.resolve()),
                "preparation_reuse_status": (
                    "reused" if prepared.reused else "built"
                ),
                "selected_split_counts": {
                    "train": len(prepared.train_ids),
                    "val": len(prepared.val_ids),
                    "test": len(prepared.test_ids),
                },
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
