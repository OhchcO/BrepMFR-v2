#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Create deterministic, instance-balanced train and validation subsets.

The script reads the official ``train.txt`` and ``val.txt`` manifests from
MFRInstSeg. It never changes the source manifests or label files. Classes
0..23 are balanced by feature-instance count. Semantic class 24 stays in a
selected CAD model but is excluded from the balancing objective.

Outputs:
    train.txt
    val.txt
    train_class_statistics.csv
    val_class_statistics.csv
    selection_report.json
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")


DEFAULT_CLASSES = tuple(range(24))
Objective = Tuple[int, float, int]


@dataclass(frozen=True)
class SampleRecord:
    sample_id: str
    instance_counts: np.ndarray
    face_counts: np.ndarray


@dataclass(frozen=True)
class SplitResult:
    split_name: str
    source_entries: int
    duplicate_entries: int
    records: Tuple[SampleRecord, ...]
    selected_indices: np.ndarray
    random_indices: np.ndarray
    objective_before_swaps: Objective
    objective_after_swaps: Objective

    @property
    def instance_matrix(self) -> np.ndarray:
        return np.stack([record.instance_counts for record in self.records])

    @property
    def face_matrix(self) -> np.ndarray:
        return np.stack([record.face_counts for record in self.records])

    @property
    def selected_instance_counts(self) -> np.ndarray:
        return counts_for_indices(self.instance_matrix, self.selected_indices)

    @property
    def random_instance_counts(self) -> np.ndarray:
        return counts_for_indices(self.instance_matrix, self.random_indices)

    @property
    def selected_face_counts(self) -> np.ndarray:
        return counts_for_indices(self.face_matrix, self.selected_indices)


def read_unique_ids(path: Path) -> Tuple[List[str], int, int]:
    """Return unique IDs, raw entry count, and duplicate entry count."""
    if not path.is_file():
        raise FileNotFoundError("Split manifest not found: {0}".format(path))

    raw_ids = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not raw_ids:
        raise ValueError("Split manifest is empty: {0}".format(path))

    seen = set()
    unique_ids = []
    for sample_id in raw_ids:
        if sample_id not in seen:
            seen.add(sample_id)
            unique_ids.append(sample_id)
    return unique_ids, len(raw_ids), len(raw_ids) - len(unique_ids)


def unwrap_label(document: object, path: Path) -> Tuple[str, dict]:
    """Read the official ``[[sample_id, annotation]]`` label wrapper."""
    if (
        isinstance(document, list)
        and len(document) == 1
        and isinstance(document[0], list)
        and len(document[0]) == 2
        and isinstance(document[0][1], dict)
    ):
        return str(document[0][0]), document[0][1]
    if isinstance(document, dict) and {"seg", "inst"}.issubset(document):
        return path.stem, document
    raise ValueError("Unsupported label format: {0}".format(path))


def semantic_labels(seg: object, path: Path) -> np.ndarray:
    """Convert a face-indexed semantic dictionary into a dense array."""
    if isinstance(seg, dict):
        try:
            indexed = {int(key): int(value) for key, value in seg.items()}
        except (TypeError, ValueError) as error:
            raise ValueError(
                "Non-integer key or value in seg: {0}".format(path)
            ) from error
        if set(indexed) != set(range(len(indexed))):
            raise ValueError(
                "Face indices in seg are not contiguous from zero: {0}".format(
                    path
                )
            )
        return np.asarray(
            [indexed[index] for index in range(len(indexed))], dtype=np.int32
        )
    if isinstance(seg, list):
        return np.asarray(seg, dtype=np.int32)
    raise ValueError("seg must be a dictionary or list: {0}".format(path))


def count_instances(
    labels: np.ndarray,
    affinity: object,
    class_ids: Sequence[int],
    path: Path,
) -> np.ndarray:
    """Count same-semantic connected components as feature instances.

    The affinity matrix is treated as an undirected graph. Traversal is
    restricted to faces with the same semantic class. An all-zero row is an
    isolated vertex and therefore counts as one single-face instance when its
    semantic class is in ``class_ids``.
    """
    matrix = np.asarray(affinity)
    node_count = len(labels)
    if matrix.shape != (node_count, node_count):
        raise ValueError(
            "inst shape {0} does not match {1} faces: {2}".format(
                tuple(matrix.shape), node_count, path
            )
        )

    linked = (matrix > 0) | (matrix.T > 0)
    class_to_column = {
        class_id: column for column, class_id in enumerate(class_ids)
    }
    counts = np.zeros(len(class_ids), dtype=np.int32)
    visited = np.zeros(node_count, dtype=bool)

    for start in range(node_count):
        if visited[start]:
            continue

        class_id = int(labels[start])
        column = class_to_column.get(class_id)
        if column is None:
            visited[start] = True
            continue

        stack = [start]
        visited[start] = True
        while stack:
            node = stack.pop()
            neighbours = np.flatnonzero(
                linked[node] & ~visited & (labels == class_id)
            )
            if len(neighbours):
                visited[neighbours] = True
                stack.extend(int(value) for value in neighbours)
        counts[column] += 1

    return counts


def load_record(
    labels_dir: Path, sample_id: str, class_ids: Sequence[int]
) -> SampleRecord:
    path = labels_dir / (sample_id + ".json")
    if not path.is_file():
        raise FileNotFoundError("Label file not found: {0}".format(path))

    try:
        with path.open("r", encoding="utf-8") as stream:
            document = json.load(stream)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(
            "Cannot read label file {0}: {1}".format(path, error)
        ) from error

    _, annotation = unwrap_label(document, path)
    labels = semantic_labels(annotation.get("seg"), path)
    instance_counts = count_instances(
        labels, annotation.get("inst"), class_ids, path
    )
    face_counts = np.asarray(
        [
            int(np.count_nonzero(labels == class_id))
            for class_id in class_ids
        ],
        dtype=np.int32,
    )
    return SampleRecord(sample_id, instance_counts, face_counts)


def load_records(
    labels_dir: Path,
    sample_ids: Sequence[str],
    class_ids: Sequence[int],
    split_name: str,
) -> Tuple[SampleRecord, ...]:
    records = []
    total = len(sample_ids)
    for index, sample_id in enumerate(sample_ids, start=1):
        records.append(load_record(labels_dir, sample_id, class_ids))
        if index == total or index % 1000 == 0:
            print(
                "\rLoading {0}: {1}/{2}".format(split_name, index, total),
                end="" if index < total else "\n",
                flush=True,
            )
    return tuple(records)


def balance_objective(counts: np.ndarray) -> Objective:
    """Return (uncovered classes, coefficient of variation, spread)."""
    values = np.asarray(counts, dtype=np.float64)
    uncovered = int(np.count_nonzero(values == 0))
    mean = float(values.mean())
    coefficient = (
        float(values.std() / mean) if mean > 0.0 else float("inf")
    )
    spread = int(values.max() - values.min()) if len(values) else 0
    return uncovered, coefficient, spread


def select_balanced(
    matrix: np.ndarray,
    selection_size: int,
    seed: int,
    swap_attempts: int,
    split_name: str,
) -> Tuple[np.ndarray, Objective, Objective]:
    """Select a fixed number of rows with balanced column totals."""
    model_count, class_count = matrix.shape
    if not 0 < selection_size <= model_count:
        raise ValueError(
            "Selection size for {0} must be in [1, {1}]".format(
                split_name, model_count
            )
        )
    if class_count == 0:
        raise ValueError("No classes are configured for balancing")

    available = matrix.sum(axis=0)
    if np.any(available == 0):
        missing_columns = np.flatnonzero(available == 0)
        missing_classes = [
            DEFAULT_CLASSES[int(column)] for column in missing_columns
        ]
        raise ValueError(
            "Classes absent from {0}: {1}".format(split_name, missing_classes)
        )

    rng = np.random.default_rng(seed)
    tie_rank = np.empty(model_count, dtype=np.int64)
    tie_rank[rng.permutation(model_count)] = np.arange(model_count)
    rarity_score = matrix.dot(1.0 / available.astype(np.float64))

    selected = []
    selected_mask = np.zeros(model_count, dtype=bool)
    counts = np.zeros(class_count, dtype=np.int64)

    for step in range(selection_size):
        candidates = np.flatnonzero(~selected_mask)
        proposed = counts[None, :] + matrix[candidates]
        uncovered = np.count_nonzero(proposed == 0, axis=1)
        means = proposed.mean(axis=1)
        coefficients = np.divide(
            proposed.std(axis=1),
            means,
            out=np.full_like(means, np.inf, dtype=np.float64),
            where=means > 0,
        )
        spreads = proposed.max(axis=1) - proposed.min(axis=1)
        order = np.lexsort(
            (
                tie_rank[candidates],
                -rarity_score[candidates],
                spreads,
                coefficients,
                uncovered,
            )
        )
        chosen = int(candidates[int(order[0])])
        selected.append(chosen)
        selected_mask[chosen] = True
        counts += matrix[chosen]

        if step + 1 == selection_size or (step + 1) % 50 == 0:
            print(
                "\rSelecting {0}: {1}/{2}".format(
                    split_name, step + 1, selection_size
                ),
                end="" if step + 1 < selection_size else "\n",
                flush=True,
            )

    before_swaps = balance_objective(counts)
    current_objective = before_swaps
    unselected = np.flatnonzero(~selected_mask).tolist()
    random_generator = random.Random(seed)

    for _ in range(swap_attempts):
        if not selected or not unselected:
            break
        selected_position = random_generator.randrange(len(selected))
        unselected_position = random_generator.randrange(len(unselected))
        removed = selected[selected_position]
        added = unselected[unselected_position]
        proposed_counts = counts - matrix[removed] + matrix[added]
        proposed_objective = balance_objective(proposed_counts)
        if proposed_objective < current_objective:
            selected[selected_position] = added
            unselected[unselected_position] = removed
            counts = proposed_counts
            current_objective = proposed_objective

    return (
        np.asarray(selected, dtype=np.int64),
        before_swaps,
        current_objective,
    )


def counts_for_indices(
    matrix: np.ndarray, indices: np.ndarray
) -> np.ndarray:
    if len(indices) == 0:
        return np.zeros(matrix.shape[1], dtype=np.int64)
    return matrix[indices].sum(axis=0).astype(np.int64)


def sample_split(
    dataset_root: Path,
    split_name: str,
    ratio: float,
    seed: int,
    swap_attempts: int,
) -> SplitResult:
    split_file = dataset_root / (split_name + ".txt")
    sample_ids, source_entries, duplicate_entries = read_unique_ids(split_file)
    records = load_records(
        dataset_root / "labels",
        sample_ids,
        DEFAULT_CLASSES,
        split_name,
    )
    matrix = np.stack([record.instance_counts for record in records])
    selection_size = max(1, int(math.floor(len(records) * ratio)))
    selected, before_swaps, after_swaps = select_balanced(
        matrix,
        selection_size,
        seed,
        swap_attempts,
        split_name,
    )
    random_indices = np.random.default_rng(seed).choice(
        len(records), size=selection_size, replace=False
    )
    return SplitResult(
        split_name=split_name,
        source_entries=source_entries,
        duplicate_entries=duplicate_entries,
        records=records,
        selected_indices=selected,
        random_indices=random_indices,
        objective_before_swaps=before_swaps,
        objective_after_swaps=after_swaps,
    )


def selected_records(result: SplitResult) -> List[SampleRecord]:
    records = [result.records[int(index)] for index in result.selected_indices]
    return sorted(records, key=lambda record: record.sample_id)


def write_manifest(path: Path, result: SplitResult) -> None:
    text = "\n".join(
        record.sample_id for record in selected_records(result)
    ) + "\n"
    path.write_text(text, encoding="utf-8")


def write_class_statistics(path: Path, result: SplitResult) -> None:
    full_counts = result.instance_matrix.sum(axis=0)
    selected_counts = result.selected_instance_counts
    random_counts = result.random_instance_counts
    selected_faces = result.selected_face_counts

    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "class_id",
                "full_split_instances",
                "balanced_subset_instances",
                "random_subset_instances",
                "balanced_subset_faces",
            ]
        )
        for column, class_id in enumerate(DEFAULT_CLASSES):
            writer.writerow(
                [
                    class_id,
                    int(full_counts[column]),
                    int(selected_counts[column]),
                    int(random_counts[column]),
                    int(selected_faces[column]),
                ]
            )


def split_report(result: SplitResult) -> dict:
    full_counts = result.instance_matrix.sum(axis=0)
    selected_counts = result.selected_instance_counts
    random_counts = result.random_instance_counts
    return {
        "official_entries": result.source_entries,
        "unique_models": len(result.records),
        "duplicate_entries_ignored": result.duplicate_entries,
        "selected_models": int(len(result.selected_indices)),
        "actual_unique_model_ratio": (
            len(result.selected_indices) / len(result.records)
        ),
        "objective_before_swaps": list(result.objective_before_swaps),
        "objective_after_swaps": list(result.objective_after_swaps),
        "balanced_objective": list(balance_objective(selected_counts)),
        "random_objective": list(balance_objective(random_counts)),
        "full_instance_counts": {
            str(class_id): int(full_counts[column])
            for column, class_id in enumerate(DEFAULT_CLASSES)
        },
        "balanced_instance_counts": {
            str(class_id): int(selected_counts[column])
            for column, class_id in enumerate(DEFAULT_CLASSES)
        },
        "random_instance_counts": {
            str(class_id): int(random_counts[column])
            for column, class_id in enumerate(DEFAULT_CLASSES)
        },
    }


def validate_output_dir(dataset_root: Path, output_dir: Path) -> None:
    source_root = dataset_root.resolve()
    destination_root = output_dir.resolve()
    if source_root == destination_root:
        raise ValueError(
            "--output-dir must not be the dataset root because train.txt "
            "and val.txt would overwrite the official manifests"
        )

    source_manifests = {
        (source_root / "train.txt").resolve(),
        (source_root / "val.txt").resolve(),
    }
    output_manifests = {
        (destination_root / "train.txt").resolve(),
        (destination_root / "val.txt").resolve(),
    }
    if source_manifests & output_manifests:
        raise ValueError("--output-dir would overwrite an official manifest")


def write_outputs(
    output_dir: Path,
    ratio: float,
    seed: int,
    swap_attempts: int,
    results: Dict[str, SplitResult],
    dataset_root: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for split_name in ("train", "val"):
        result = results[split_name]
        write_manifest(output_dir / (split_name + ".txt"), result)
        write_class_statistics(
            output_dir / (split_name + "_class_statistics.csv"), result
        )

    report = {
        "dataset_root": str(dataset_root.resolve()),
        "output_dir": str(output_dir.resolve()),
        "ratio": ratio,
        "seed": seed,
        "swap_attempts": swap_attempts,
        "balanced_class_ids": list(DEFAULT_CLASSES),
        "excluded_semantic_class": 24,
        "train": split_report(results["train"]),
        "val": split_report(results["val"]),
    }
    with (output_dir / "selection_report.json").open(
        "w", encoding="utf-8"
    ) as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path(r"D:\Code\Datasets\MFRInstSeg"),
        help="MFRInstSeg root directory",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help=(
            "Output directory; default: "
            "<dataset-root>/subsets/train_val_balanced"
        ),
    )
    parser.add_argument(
        "--ratio",
        type=float,
        default=0.01,
        help="Sampling ratio applied independently to train and val",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2026,
        help="Fixed random seed (default: 2026)",
    )
    parser.add_argument(
        "--swap-attempts",
        type=int,
        default=20000,
        help="Number of post-greedy random swap attempts",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)
    try:
        if not 0.0 < args.ratio <= 1.0:
            raise ValueError("--ratio must be in (0, 1]")
        if args.swap_attempts < 0:
            raise ValueError("--swap-attempts must be non-negative")

        dataset_root = args.dataset_root
        output_dir = args.output_dir or (
            dataset_root / "subsets" / "train_val_balanced"
        )
        validate_output_dir(dataset_root, output_dir)

        results = {
            split_name: sample_split(
                dataset_root=dataset_root,
                split_name=split_name,
                ratio=args.ratio,
                seed=args.seed,
                swap_attempts=args.swap_attempts,
            )
            for split_name in ("train", "val")
        }
        write_outputs(
            output_dir=output_dir,
            ratio=args.ratio,
            seed=args.seed,
            swap_attempts=args.swap_attempts,
            results=results,
            dataset_root=dataset_root,
        )

        for split_name in ("train", "val"):
            result = results[split_name]
            print(
                "{0}: selected {1}/{2}, balanced={3}, random={4}".format(
                    split_name,
                    len(result.selected_indices),
                    len(result.records),
                    balance_objective(result.selected_instance_counts),
                    balance_objective(result.random_instance_counts),
                )
            )
        print("Output: {0}".format(output_dir))
        return 0
    except (FileNotFoundError, ValueError, OSError) as error:
        print("Error: {0}".format(error), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
