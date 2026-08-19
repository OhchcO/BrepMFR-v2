#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Derive a deterministic MFRInstSeg subset from a prepared parent subset.

Only train and validation manifests are sampled.  The complete parent test
manifest is copied unchanged, and all geometry BIN files are reused from the
shared cache.  No STEP conversion is performed by this tool.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Mapping, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.sample_mfrinstseg_1pct import select_balanced_sample_ids
from utils.reproducibility import derive_seed


SPLIT_NAMES = ("train", "val", "test")
AUDIT_FILES = ("conversion_exclusions.json", "conversion_failures.json")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_ids(path: Path) -> Tuple[str, ...]:
    if not path.is_file():
        raise FileNotFoundError("Split manifest not found: {0}".format(path))
    values = tuple(
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    if not values:
        raise ValueError("Split manifest is empty: {0}".format(path))
    if len(set(values)) != len(values):
        raise ValueError("Split manifest contains duplicate model IDs: {0}".format(path))
    return values


def _read_json_object(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError("JSON file not found: {0}".format(path))
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("Cannot read JSON file {0}: {1}".format(path, error)) from error
    if not isinstance(document, dict):
        raise ValueError("JSON document must be an object: {0}".format(path))
    return document


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(text)
    os.replace(str(temporary), str(path))


def _write_ids(path: Path, sample_ids: Sequence[str]) -> None:
    _write_text_atomic(path, "\n".join(sample_ids) + "\n")


def _validate_disjoint(splits: Mapping[str, Sequence[str]]) -> None:
    seen = set()
    for split_name in SPLIT_NAMES:
        overlap = seen.intersection(splits[split_name])
        if overlap:
            raise ValueError(
                "Parent split manifests are not model-level disjoint; first overlap: {0}".format(
                    sorted(overlap)[0]
                )
            )
        seen.update(splits[split_name])


def _validate_shared_bins(
    shared_bin_cache: Path, selected_ids: Mapping[str, Sequence[str]]
) -> None:
    bins_root = shared_bin_cache / "bin"
    if not bins_root.is_dir():
        raise FileNotFoundError("Shared BIN directory not found: {0}".format(bins_root))
    missing = []
    for split_name in SPLIT_NAMES:
        for sample_id in selected_ids[split_name]:
            if not (bins_root / (sample_id + ".bin")).is_file():
                missing.append(sample_id)
                if len(missing) == 10:
                    break
        if len(missing) == 10:
            break
    if missing:
        raise FileNotFoundError(
            "Shared BIN cache is missing selected model(s): {0}".format(
                ", ".join(missing)
            )
        )


def derive_nested_subset(
    raw_dataset: Path,
    parent_prepared_dataset: Path,
    output_prepared_dataset: Path,
    shared_bin_cache: Path,
    sample_percent: float,
    seed: int,
    rebuild: bool = False,
) -> dict:
    raw_dataset = Path(raw_dataset).resolve()
    parent_prepared_dataset = Path(parent_prepared_dataset).resolve()
    output_prepared_dataset = Path(output_prepared_dataset).resolve()
    shared_bin_cache = Path(shared_bin_cache).resolve()

    if parent_prepared_dataset == output_prepared_dataset:
        raise ValueError("Parent and output prepared dataset paths must differ")
    if not 0.0 < sample_percent < 1.0:
        raise ValueError("sample_percent must be in (0, 1)")
    labels_root = raw_dataset / "labels"
    if not labels_root.is_dir():
        raise FileNotFoundError("Raw label directory not found: {0}".format(labels_root))

    parent_manifest_path = parent_prepared_dataset / "preparation_manifest.json"
    parent_manifest = _read_json_object(parent_manifest_path)
    try:
        parent_percent = float(parent_manifest["sample_percent"])
        parent_seed = int(parent_manifest["base_seed"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            "Parent preparation manifest must record sample_percent and base_seed"
        ) from error
    if parent_seed != seed:
        raise ValueError(
            "Parent seed {0} does not match requested seed {1}".format(parent_seed, seed)
        )
    if not 0.0 < sample_percent < parent_percent:
        raise ValueError(
            "Child sample_percent must be smaller than parent sample_percent ({0})".format(
                parent_percent
            )
        )

    parent_ids = {
        split_name: _read_ids(parent_prepared_dataset / (split_name + ".txt"))
        for split_name in SPLIT_NAMES
    }
    _validate_disjoint(parent_ids)
    nested_ratio = sample_percent / parent_percent
    selected_ids: Dict[str, Tuple[str, ...]] = {}
    for split_name in ("train", "val"):
        selected_ids[split_name] = select_balanced_sample_ids(
            raw_dataset,
            parent_ids[split_name],
            nested_ratio,
            derive_seed(seed, "nested_subset", split_name),
            split_name,
        )
    selected_ids["test"] = tuple(parent_ids["test"])

    for split_name in ("train", "val"):
        expected_count = max(
            1, int(math.floor(len(parent_ids[split_name]) * nested_ratio))
        )
        if len(selected_ids[split_name]) != expected_count:
            raise RuntimeError(
                "Unexpected {0} subset size: expected {1}, got {2}".format(
                    split_name, expected_count, len(selected_ids[split_name])
                )
            )
        if not set(selected_ids[split_name]).issubset(parent_ids[split_name]):
            raise RuntimeError("Derived {0} IDs escaped the parent subset".format(split_name))
    _validate_shared_bins(shared_bin_cache, selected_ids)

    manifest = {
        "manifest_type": "nested_mfrinstseg_subset",
        "manifest_version": 1,
        "base_seed": seed,
        "sample_percent": sample_percent,
        "parent_sample_percent": parent_percent,
        "nested_ratio": nested_ratio,
        "parent_prepared_dataset_path": str(parent_prepared_dataset),
        "parent_manifest_sha256": _sha256(parent_manifest_path),
        "shared_bin_cache_path": str(shared_bin_cache),
        "raw_dataset_path": str(raw_dataset),
        "selection_policy": "balanced_train_val_from_parent_full_parent_test",
        "selection_seed_namespace": "nested_subset",
        "test_policy": "full_parent_test_split",
        "source_split_counts": {
            name: len(parent_ids[name]) for name in SPLIT_NAMES
        },
        "selected_split_counts": {
            name: len(selected_ids[name]) for name in SPLIT_NAMES
        },
        "effective_selected_ids": {
            name: list(selected_ids[name]) for name in SPLIT_NAMES
        },
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    output_manifest_path = output_prepared_dataset / "preparation_manifest.json"
    if output_manifest_path.exists() and not rebuild:
        existing = _read_json_object(output_manifest_path)
        comparable_keys = (
            "manifest_type",
            "manifest_version",
            "base_seed",
            "sample_percent",
            "parent_sample_percent",
            "parent_manifest_sha256",
            "shared_bin_cache_path",
            "selection_policy",
            "test_policy",
            "effective_selected_ids",
        )
        manifests_match = all(
            existing.get(key) == manifest.get(key) for key in comparable_keys
        )
        split_files_match = all(
            _read_ids(output_prepared_dataset / (name + ".txt"))
            == selected_ids[name]
            for name in SPLIT_NAMES
        )
        if manifests_match and split_files_match:
            return {**existing, "reused": True}
        raise ValueError(
            "Existing nested subset does not match the requested configuration; "
            "use --rebuild to replace its generated manifests"
        )

    output_prepared_dataset.mkdir(parents=True, exist_ok=True)
    for split_name in SPLIT_NAMES:
        _write_ids(
            output_prepared_dataset / (split_name + ".txt"),
            selected_ids[split_name],
        )
    for filename in AUDIT_FILES:
        _write_text_atomic(output_prepared_dataset / filename, "[]\n")
    _write_text_atomic(
        output_manifest_path,
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    return {**manifest, "reused": False}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dataset", type=Path, required=True)
    parser.add_argument("--parent-prepared-dataset", type=Path, required=True)
    parser.add_argument("--output-prepared-dataset", type=Path, required=True)
    parser.add_argument("--shared-bin-cache", type=Path, required=True)
    parser.add_argument("--sample-percent", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rebuild", action="store_true")
    return parser


def main(argv: Sequence[str] = None) -> int:
    args = _build_parser().parse_args(argv)
    manifest = derive_nested_subset(
        raw_dataset=args.raw_dataset,
        parent_prepared_dataset=args.parent_prepared_dataset,
        output_prepared_dataset=args.output_prepared_dataset,
        shared_bin_cache=args.shared_bin_cache,
        sample_percent=args.sample_percent,
        seed=args.seed,
        rebuild=args.rebuild,
    )
    counts = manifest["selected_split_counts"]
    print(
        "Nested subset {0}: train={1}, val={2}, test={3}".format(
            "reused" if manifest.get("reused") else "created",
            counts["train"],
            counts["val"],
            counts["test"],
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
