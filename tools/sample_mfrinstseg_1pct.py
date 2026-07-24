#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""从 MFRInstSeg 官方训练集抽取类别尽量均衡的小样本清单。

默认行为：
1. 只读取官方 train.txt，不修改原数据集；
2. 抽取唯一训练模型数量的 1%；
3. 根据 ``inst`` 实例邻接矩阵统计每个模型中 0~23 类特征的实例数；
4. 用多标签贪心选择和随机交换优化，使 24 类实例数量尽量接近；
5. 输出 train_1pct.txt、class_statistics.csv 和 selection_report.json。

标签 24 是非加工特征/背景面，默认不参与特征均衡，但仍会保留在被选模型中。
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
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np


DEFAULT_CLASSES = tuple(range(24))


@dataclass(frozen=True)
class SampleRecord:
    """一个 CAD 模型及其中各语义类别的特征实例数量。"""

    sample_id: str
    instance_counts: np.ndarray
    face_counts: np.ndarray


@dataclass(frozen=True)
class SplitResult:
    """Sampling inputs and results for one official split."""

    split_file: Path
    raw_model_count: int
    duplicate_count: int
    records: Sequence[SampleRecord]
    selected_indices: np.ndarray
    random_indices: np.ndarray
    before_swaps: Tuple[int, float, int]
    after_swaps: Tuple[int, float, int]


def parse_class_spec(text: str) -> Tuple[int, ...]:
    """解析 ``0-23`` 或 ``0,1,5-8`` 形式的类别集合。"""
    values = set()
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            start_text, end_text = item.split("-", 1)
            start, end = int(start_text), int(end_text)
            if end < start:
                raise ValueError("类别范围终点不能小于起点：{0}".format(item))
            values.update(range(start, end + 1))
        else:
            values.add(int(item))
    if not values:
        raise ValueError("类别集合不能为空")
    if min(values) < 0:
        raise ValueError("类别编号不能为负数")
    return tuple(sorted(values))


def read_unique_ids(path: Path) -> Tuple[List[str], int]:
    """读取划分清单并保持首次出现的顺序，同时返回重复项数量。"""
    if not path.is_file():
        raise FileNotFoundError("找不到训练集清单：{0}".format(path))
    raw_ids = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    seen = set()
    unique_ids = []
    for sample_id in raw_ids:
        if sample_id not in seen:
            seen.add(sample_id)
            unique_ids.append(sample_id)
    return unique_ids, len(raw_ids) - len(unique_ids)


def unwrap_label(document: object, path: Path) -> Tuple[str, dict]:
    """兼容官方 ``[[sample_id, annotation]]`` 标签包装格式。"""
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
    raise ValueError("无法识别标签格式：{0}".format(path))


def semantic_labels(seg: object, path: Path) -> np.ndarray:
    """将语义标签字典按面编号转换成连续数组。"""
    if isinstance(seg, dict):
        try:
            indexed = {int(key): int(value) for key, value in seg.items()}
        except (TypeError, ValueError) as error:
            raise ValueError("seg 中存在非整数键值：{0}".format(path)) from error
        expected = set(range(len(indexed)))
        if set(indexed) != expected:
            raise ValueError("seg 面编号不是从 0 开始的连续编号：{0}".format(path))
        return np.asarray([indexed[index] for index in range(len(indexed))], dtype=np.int32)
    if isinstance(seg, list):
        return np.asarray(seg, dtype=np.int32)
    raise ValueError("seg 必须是字典或列表：{0}".format(path))


def count_instances(
    labels: np.ndarray,
    affinity: object,
    class_ids: Sequence[int],
    path: Path,
) -> np.ndarray:
    """按语义类别统计实例连通分量数量。

    ``inst[i][j] > 0`` 表示面 i、j 属于同一实例。对矩阵做无向化后，
    每个连通分量计为一个特征实例；全零行不计为实例。
    """
    matrix = np.asarray(affinity)
    node_count = len(labels)
    if matrix.shape != (node_count, node_count):
        raise ValueError(
            "inst 形状 {0} 与面数 {1} 不一致：{2}".format(
                tuple(matrix.shape), node_count, path
            )
        )
    linked = (matrix > 0) | (matrix.T > 0)
    class_to_column = {class_id: index for index, class_id in enumerate(class_ids)}
    counts = np.zeros(len(class_ids), dtype=np.int32)
    visited = np.zeros(node_count, dtype=bool)

    for start in range(node_count):
        if visited[start] or not linked[start].any():
            continue
        stack = [start]
        visited[start] = True
        component = []
        while stack:
            node = stack.pop()
            component.append(node)
            neighbours = np.flatnonzero(linked[node] & ~visited)
            if len(neighbours):
                visited[neighbours] = True
                stack.extend(int(value) for value in neighbours)

        component_labels = labels[np.asarray(component, dtype=np.int64)]
        values, frequencies = np.unique(component_labels, return_counts=True)
        class_id = int(values[int(np.argmax(frequencies))])
        column = class_to_column.get(class_id)
        if column is not None:
            counts[column] += 1
    return counts


def load_record(
    labels_dir: Path, sample_id: str, class_ids: Sequence[int]
) -> SampleRecord:
    path = labels_dir / (sample_id + ".json")
    if not path.is_file():
        raise FileNotFoundError("找不到标签文件：{0}".format(path))
    try:
        with path.open("r", encoding="utf-8") as stream:
            document = json.load(stream)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("读取标签失败：{0} ({1})".format(path, error)) from error

    _, annotation = unwrap_label(document, path)
    labels = semantic_labels(annotation.get("seg"), path)
    instance_counts = count_instances(
        labels, annotation.get("inst"), class_ids, path
    )
    face_counts = np.asarray(
        [int(np.count_nonzero(labels == class_id)) for class_id in class_ids],
        dtype=np.int32,
    )
    return SampleRecord(sample_id, instance_counts, face_counts)


def load_records(
    labels_dir: Path, sample_ids: Sequence[str], class_ids: Sequence[int]
) -> List[SampleRecord]:
    records = []
    total = len(sample_ids)
    for index, sample_id in enumerate(sample_ids, start=1):
        records.append(load_record(labels_dir, sample_id, class_ids))
        if index == total or index % 1000 == 0:
            print(
                "\r读取标签：{0}/{1}".format(index, total),
                end="" if index < total else "\n",
                flush=True,
            )
    return records


def balance_objective(counts: np.ndarray) -> Tuple[int, float, int]:
    """返回越小越好的 (未覆盖类别数, 变异系数, 极差)。"""
    counts = np.asarray(counts, dtype=np.float64)
    uncovered = int(np.count_nonzero(counts == 0))
    mean = float(counts.mean())
    coefficient_of_variation = (
        float(counts.std() / mean) if mean > 0.0 else float("inf")
    )
    spread = int(counts.max() - counts.min()) if len(counts) else 0
    return uncovered, coefficient_of_variation, spread


def select_balanced(
    matrix: np.ndarray,
    selection_size: int,
    seed: int,
    swap_attempts: int,
) -> Tuple[np.ndarray, Tuple[int, float, int], Tuple[int, float, int]]:
    """在固定模型数约束下选择实例分布尽量均衡的行。"""
    model_count, class_count = matrix.shape
    if not 0 < selection_size <= model_count:
        raise ValueError("抽样数量必须位于 1 到训练模型总数之间")
    if class_count == 0:
        raise ValueError("没有用于均衡的类别")
    available = matrix.sum(axis=0)
    if np.any(available == 0):
        missing = np.flatnonzero(available == 0).tolist()
        raise ValueError("下列类别在训练集中没有实例：{0}".format(missing))

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
        if (step + 1) % 50 == 0 or step + 1 == selection_size:
            print(
                "\r均衡选择：{0}/{1}".format(step + 1, selection_size),
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


def counts_for_indices(matrix: np.ndarray, indices: np.ndarray) -> np.ndarray:
    if len(indices) == 0:
        return np.zeros(matrix.shape[1], dtype=np.int64)
    return matrix[indices].sum(axis=0).astype(np.int64)


def sample_split(
    *,
    dataset_root: Path,
    split_file: Path,
    class_ids: Sequence[int],
    ratio: float,
    seed: int,
    swap_attempts: int,
) -> SplitResult:
    sample_ids, duplicate_count = read_unique_ids(split_file)
    records = load_records(dataset_root / "labels", sample_ids, class_ids)
    selection_size = max(1, int(math.floor(len(records) * ratio)))
    matrix = np.stack([record.instance_counts for record in records])
    selected_indices, before_swaps, after_swaps = select_balanced(
        matrix, selection_size, seed, swap_attempts
    )
    random_indices = np.random.default_rng(seed).choice(
        len(records), size=selection_size, replace=False
    )
    return SplitResult(
        split_file,
        len(sample_ids) + duplicate_count,
        duplicate_count,
        records,
        selected_indices,
        random_indices,
        before_swaps,
        after_swaps,
    )


def write_manifest(path: Path, records: Sequence[SampleRecord]) -> None:
    path.write_text(
        "\n".join(sorted(record.sample_id for record in records)) + "\n",
        encoding="utf-8",
    )


def write_class_statistics(
    path: Path, result: SplitResult, class_ids: Sequence[int]
) -> None:
    matrix = np.stack([record.instance_counts for record in result.records])
    face_matrix = np.stack([record.face_counts for record in result.records])
    balanced = counts_for_indices(matrix, result.selected_indices)
    random_counts = counts_for_indices(matrix, result.random_indices)
    balanced_faces = counts_for_indices(face_matrix, result.selected_indices)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow([
            "class_id", "full_split_instances", "balanced_subset_instances",
            "random_subset_instances", "balanced_subset_faces",
        ])
        for column, class_id in enumerate(class_ids):
            writer.writerow([
                class_id, int(matrix[:, column].sum()), int(balanced[column]),
                int(random_counts[column]), int(balanced_faces[column]),
            ])


def build_split_report(result: SplitResult, class_ids: Sequence[int]) -> dict:
    matrix = np.stack([record.instance_counts for record in result.records])
    balanced = counts_for_indices(matrix, result.selected_indices)
    random_counts = counts_for_indices(matrix, result.random_indices)
    return {
        "source_split": str(result.split_file.resolve()),
        "official_entries": result.raw_model_count,
        "unique_models": len(result.records),
        "duplicate_entries_ignored": result.duplicate_count,
        "selected_models": int(len(result.selected_indices)),
        "actual_unique_model_ratio": len(result.selected_indices) / len(result.records),
        "objective_before_swaps": list(result.before_swaps),
        "objective_after_swaps": list(result.after_swaps),
        "balanced_objective": list(balance_objective(balanced)),
        "random_objective": list(balance_objective(random_counts)),
        "full_instance_counts": {
            str(class_id): int(matrix[:, column].sum())
            for column, class_id in enumerate(class_ids)
        },
        "balanced_instance_counts": {
            str(class_id): int(balanced[column])
            for column, class_id in enumerate(class_ids)
        },
        "random_instance_counts": {
            str(class_id): int(random_counts[column])
            for column, class_id in enumerate(class_ids)
        },
    }


def validate_output_dir(dataset_root: Path, output_dir: Path) -> None:
    """Reject output destinations that would replace official split manifests."""
    if output_dir.resolve() == dataset_root.resolve():
        raise ValueError(
            "--output-dir must not be the dataset root because it would overwrite "
            "official train.txt and val.txt"
        )


def write_outputs(
    output_dir: Path,
    records: Sequence[SampleRecord],
    selected_indices: np.ndarray,
    class_ids: Sequence[int],
    full_instance_counts: np.ndarray,
    random_indices: np.ndarray,
    report: dict,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_records = [records[int(index)] for index in selected_indices]
    selected_records.sort(key=lambda record: record.sample_id)
    manifest = "\n".join(record.sample_id for record in selected_records) + "\n"
    (output_dir / "train_1pct.txt").write_text(manifest, encoding="utf-8")

    balanced_instance_counts = counts_for_indices(
        np.stack([record.instance_counts for record in records]), selected_indices
    )
    random_instance_counts = counts_for_indices(
        np.stack([record.instance_counts for record in records]), random_indices
    )
    balanced_face_counts = counts_for_indices(
        np.stack([record.face_counts for record in records]), selected_indices
    )

    with (output_dir / "class_statistics.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "class_id",
                "full_train_instances",
                "balanced_1pct_instances",
                "random_1pct_instances",
                "balanced_1pct_faces",
            ]
        )
        for column, class_id in enumerate(class_ids):
            writer.writerow(
                [
                    class_id,
                    int(full_instance_counts[column]),
                    int(balanced_instance_counts[column]),
                    int(random_instance_counts[column]),
                    int(balanced_face_counts[column]),
                ]
            )

    with (output_dir / "selection_report.json").open(
        "w", encoding="utf-8"
    ) as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def build_report(
    *,
    dataset_root: Path,
    split_file: Path,
    output_dir: Path,
    ratio: float,
    seed: int,
    class_ids: Sequence[int],
    raw_model_count: int,
    duplicate_count: int,
    records: Sequence[SampleRecord],
    selected_indices: np.ndarray,
    random_indices: np.ndarray,
    before_swaps: Tuple[int, float, int],
    after_swaps: Tuple[int, float, int],
) -> dict:
    matrix = np.stack([record.instance_counts for record in records])
    balanced_counts = counts_for_indices(matrix, selected_indices)
    random_counts = counts_for_indices(matrix, random_indices)
    return {
        "dataset_root": str(dataset_root.resolve()),
        "source_split": str(split_file.resolve()),
        "output_dir": str(output_dir.resolve()),
        "requested_ratio": ratio,
        "seed": seed,
        "balanced_class_ids": list(class_ids),
        "class_24_note": "默认不参与均衡；它是背景/非加工特征面。",
        "official_train_entries": raw_model_count,
        "unique_train_models": len(records),
        "duplicate_entries_ignored": duplicate_count,
        "selected_models": int(len(selected_indices)),
        "actual_unique_model_ratio": len(selected_indices) / len(records),
        "objective_before_swaps": list(before_swaps),
        "objective_after_swaps": list(after_swaps),
        "balanced_objective": list(balance_objective(balanced_counts)),
        "random_objective": list(balance_objective(random_counts)),
        "full_train_instance_counts": {
            str(class_id): int(matrix[:, column].sum())
            for column, class_id in enumerate(class_ids)
        },
        "balanced_instance_counts": {
            str(class_id): int(balanced_counts[column])
            for column, class_id in enumerate(class_ids)
        },
        "random_instance_counts": {
            str(class_id): int(random_counts[column])
            for column, class_id in enumerate(class_ids)
        },
    }


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create deterministic balanced train and validation subsets."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path(r"D:\Code\Datasets\MFRInstSeg"),
        help="MFRInstSeg 根目录，默认：D:\\Code\\Datasets\\MFRInstSeg",
    )
    parser.add_argument(
        "--split-file",
        type=Path,
        help="候选训练清单，默认使用 <dataset-root>/train.txt",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="输出目录，默认使用 <dataset-root>/subsets/train_1pct_balanced",
    )
    parser.add_argument("--ratio", type=float, default=0.01, help="抽样比例")
    parser.add_argument("--seed", type=int, default=2026, help="随机种子")
    parser.add_argument(
        "--classes",
        default="0-23",
        choices=("0-23",),
        help="参与均衡的类别，例如 0-23 或 0,1,5-8；默认不含背景类 24",
    )
    parser.add_argument(
        "--swap-attempts",
        type=int,
        default=20000,
        help="贪心选择后的随机交换优化次数",
    )
    for action in parser._actions:
        if action.dest == "classes":
            action.help = argparse.SUPPRESS
        elif action.dest == "split_file":
            action.help = argparse.SUPPRESS
            for option in action.option_strings:
                del parser._option_string_actions[option]
        elif action.dest == "output_dir":
            action.help = (
                "Output directory (default: "
                "<dataset-root>/subsets/train_val_1pct_balanced)"
            )
    return parser.parse_args(argv)


def legacy_main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if not 0.0 < args.ratio <= 1.0:
            raise ValueError("--ratio 必须位于 (0, 1] 区间")
        if args.swap_attempts < 0:
            raise ValueError("--swap-attempts 不能为负数")

        dataset_root = args.dataset_root
        split_file = args.split_file or (dataset_root / "train.txt")
        output_dir = args.output_dir or (
            dataset_root / "subsets" / "train_1pct_balanced"
        )
        class_ids = parse_class_spec(args.classes)
        sample_ids, duplicate_count = read_unique_ids(split_file)
        records = load_records(dataset_root / "labels", sample_ids, class_ids)
        selection_size = max(1, int(math.floor(len(records) * args.ratio)))
        matrix = np.stack([record.instance_counts for record in records])

        selected_indices, before_swaps, after_swaps = select_balanced(
            matrix,
            selection_size=selection_size,
            seed=args.seed,
            swap_attempts=args.swap_attempts,
        )
        random_indices = np.random.default_rng(args.seed).choice(
            len(records), size=selection_size, replace=False
        )
        report = build_report(
            dataset_root=dataset_root,
            split_file=split_file,
            output_dir=output_dir,
            ratio=args.ratio,
            seed=args.seed,
            class_ids=class_ids,
            raw_model_count=len(sample_ids) + duplicate_count,
            duplicate_count=duplicate_count,
            records=records,
            selected_indices=selected_indices,
            random_indices=random_indices,
            before_swaps=before_swaps,
            after_swaps=after_swaps,
        )
        write_outputs(
            output_dir=output_dir,
            records=records,
            selected_indices=selected_indices,
            class_ids=class_ids,
            full_instance_counts=matrix.sum(axis=0),
            random_indices=random_indices,
            report=report,
        )

        balanced_counts = np.asarray(
            [report["balanced_instance_counts"][str(value)] for value in class_ids]
        )
        random_counts = np.asarray(
            [report["random_instance_counts"][str(value)] for value in class_ids]
        )
        print("完成：从 {0} 个唯一训练模型中选择 {1} 个。".format(
            len(records), selection_size
        ))
        print("均衡抽样每类实例数：{0}".format(balanced_counts.tolist()))
        print("随机抽样每类实例数：{0}".format(random_counts.tolist()))
        print("均衡目标值：{0}".format(balance_objective(balanced_counts)))
        print("随机目标值：{0}".format(balance_objective(random_counts)))
        print("输出清单：{0}".format(output_dir / "train_1pct.txt"))
        return 0
    except (FileNotFoundError, ValueError, OSError) as error:
        print("错误：{0}".format(error), file=sys.stderr)
        return 2


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if not 0.0 < args.ratio <= 1.0:
            raise ValueError("--ratio must be in (0, 1]")
        if args.swap_attempts < 0:
            raise ValueError("--swap-attempts cannot be negative")

        dataset_root = args.dataset_root
        output_dir = args.output_dir or (
            dataset_root / "subsets" / "train_val_1pct_balanced"
        )
        validate_output_dir(dataset_root, output_dir)
        class_ids = DEFAULT_CLASSES
        split_results = {
            name: sample_split(
                dataset_root=dataset_root,
                split_file=dataset_root / (name + ".txt"),
                class_ids=class_ids,
                ratio=args.ratio,
                seed=args.seed,
                swap_attempts=args.swap_attempts,
            )
            for name in ("train", "val")
        }
        output_dir.mkdir(parents=True, exist_ok=True)
        for split_name, result in split_results.items():
            selected_records = [
                result.records[int(index)] for index in result.selected_indices
            ]
            write_manifest(output_dir / (split_name + ".txt"), selected_records)
            write_class_statistics(
                output_dir / (split_name + "_class_statistics.csv"),
                result,
                class_ids,
            )

        report = {
            "dataset_root": str(dataset_root.resolve()),
            "output_dir": str(output_dir.resolve()),
            "requested_ratio": args.ratio,
            "seed": args.seed,
            "balanced_class_ids": list(class_ids),
            "class_24_note": "Background class 24 is not balanced.",
            "train": build_split_report(split_results["train"], class_ids),
            "val": build_split_report(split_results["val"], class_ids),
        }
        with (output_dir / "selection_report.json").open(
            "w", encoding="utf-8"
        ) as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2)
            stream.write("\n")

        for split_name, result in split_results.items():
            matrix = np.stack([record.instance_counts for record in result.records])
            balanced = counts_for_indices(matrix, result.selected_indices)
            random_counts = counts_for_indices(matrix, result.random_indices)
            print(
                "Completed {0}: selected {1} of {2} unique models.".format(
                    split_name, len(result.selected_indices), len(result.records)
                )
            )
            print("Balanced objective: {0}".format(balance_objective(balanced)))
            print("Random objective: {0}".format(balance_objective(random_counts)))
        print("Output directory: {0}".format(output_dir))
        return 0
    except (FileNotFoundError, ValueError, OSError) as error:
        print("Error: {0}".format(error), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
