"""Deterministically prepare one reusable MFRInstSeg BIN cache.

This module deliberately does not import the STEP converter at module import
time.  Training configuration and unit tests can therefore inspect splitting
and cache decisions without requiring PyOCC.  The converter is loaded only
after a new cache has been selected for conversion.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Tuple

from tools.sample_mfrinstseg_1pct import select_balanced_sample_ids
from utils.reproducibility import DEFAULT_SEED, derive_seed


FORMAT_VERSION = 1
CONVERTER_FORMAT_VERSION = 2
SPLIT_NAMES = ("train", "val", "test")
DEFAULT_CANDIDATE_GRIDS = (20, 32, 48, 64, 80)


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

    def __post_init__(self) -> None:
        object.__setattr__(self, "raw_root", Path(self.raw_root))
        object.__setattr__(self, "prepared_root", Path(self.prepared_root))
        object.__setattr__(self, "extractor_file", Path(self.extractor_file))
        object.__setattr__(self, "schema_file", Path(self.schema_file))
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
    return _validate_sample_ids(sample_ids, allow_empty=True)


def _official_splits(raw_root: Path, known_ids: Tuple[str, ...]) -> Dict[str, Tuple[str, ...]]:
    manifests = {name: Path(raw_root) / (name + ".txt") for name in SPLIT_NAMES}
    present = [name for name, path in manifests.items() if path.exists()]
    if present and len(present) != len(SPLIT_NAMES):
        raise ValueError("Official split manifests must include all three train/val/test files")
    if not present:
        return {}
    splits = {name: _read_official_manifest(path) for name, path in manifests.items()}
    all_ids = tuple(sample_id for name in SPLIT_NAMES for sample_id in splits[name])
    if len(set(all_ids)) != len(all_ids):
        raise ValueError("Official split manifests must be model-level disjoint")
    if set(all_ids) != set(known_ids):
        raise ValueError("Official split manifests must contain every discovered raw model exactly once")
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
        "cylindrical_pos": True,
        "base_seed": config.seed,
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
        "output_features": {"face_pointcloud": True, "cylindrical_pos": True},
        "conversion_options": _conversion_options(config),
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
            "conversion_options",
        )
    }


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
        selected = manifest["selected_ids"]
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
                bin_file = prepared_root / "bin" / (sample_id + ".bin")
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


def _prepared_dataset(prepared_root: Path, selected_ids: Mapping[str, Sequence[str]], reused: bool) -> PreparedDataset:
    return PreparedDataset(
        root=prepared_root,
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
    )
    requirements = converter.BinRequirements(
        require_instance=True,
        require_pointcloud=True,
        require_cylindrical=True,
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


def ensure_prepared_dataset(config: PreparationConfig) -> PreparedDataset:
    """Reuse a verified exact cache or atomically prepare it in one converter call."""
    samples = discover_raw_samples(config.raw_root)
    model_splits = resolve_model_splits(config.raw_root, tuple(samples), config.split_ratios, config.seed)
    selected_ids = select_split_subsets(config.raw_root, model_splits, config.sample_percent, config.seed)
    request = build_preparation_manifest(config, selected_ids, samples=samples)
    manifest_path = config.prepared_root / "preparation_manifest.json"

    if manifest_path.exists():
        try:
            existing = _read_manifest(manifest_path)
            request_matches = _request_fingerprint(existing) == _request_fingerprint(request)
        except (KeyError, TypeError, PreparationMismatchError):
            request_matches = False
        if request_matches:
            converter = _load_converter()
            point_sampling, _, requirements = _converter_contract(config, converter)
            cache_failures = _validate_prepared_outputs(
                config.prepared_root, existing, converter, requirements, point_sampling
            )
            if not cache_failures:
                return _prepared_dataset(config.prepared_root, selected_ids, reused=True)
            if not config.rebuild:
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
    _atomic_write_json(manifest_path, request)

    failures: list = []
    try:
        converter = _load_converter()
        point_sampling, options, requirements = _converter_contract(config, converter)
        # Direct automatic preparation deliberately uses one prepared root for
        # generated manifests and BINs.  Task 3 keeps the old source/output
        # rejection as a CLI-only guard; this API receives selected IDs and
        # never reads a separate subset directory.
        paths = converter.ConversionPaths(config.raw_root, config.prepared_root, config.prepared_root)
        result = converter.convert_selected_samples(
            selected_ids,
            paths,
            config.extractor_file,
            config.schema_file,
            options,
            config.rebuild,
        )
        successful_ids, failures, conversion = _normalise_conversion_result(result, selected_ids)
        if not failures:
            request["successful_ids"] = successful_ids
            validation_failures = _validate_prepared_outputs(
                config.prepared_root, request, converter, requirements, point_sampling
            )
            failures.extend(validation_failures)
    except Exception as error:
        successful_ids = {name: [] for name in SPLIT_NAMES}
        conversion = {}
        failures = [{"sample_id": None, "exception_type": type(error).__name__, "message": str(error)}]

    request["conversion"] = conversion
    request["successful_ids"] = successful_ids
    request["failed_ids"] = failures
    _atomic_write_json(config.prepared_root / "conversion_failures.json", failures)
    _atomic_write_json(manifest_path, request)
    if failures:
        raise PreparationError("Prepared conversion failed; inspect conversion_failures.json")
    return _prepared_dataset(config.prepared_root, selected_ids, reused=False)
