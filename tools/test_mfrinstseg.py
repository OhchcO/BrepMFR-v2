#!/usr/bin/env python
"""Evaluate a trained BrepMFR checkpoint on an existing BIN split."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import sys
import time
from typing import Mapping

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.dataset import CADSynth
from models.brepseg_model import BrepSeg


@dataclass(frozen=True)
class GeometryRequirements:
    """Strict BIN fields selected by checkpoint-saved model switches."""

    use_face_uv: bool
    require_face_pointcloud: bool
    require_parametric_surface: bool
    require_cylindrical_pos: bool


def _configuration_value(configuration, name, default):
    if isinstance(configuration, Mapping):
        return configuration.get(name, default)
    return getattr(configuration, name, default)


def geometry_requirements(hparams):
    """Translate checkpoint hyperparameters into strict dataset requirements."""
    return GeometryRequirements(
        use_face_uv=bool(_configuration_value(hparams, "use_face_uv", True)),
        require_face_pointcloud=bool(
            _configuration_value(hparams, "use_face_pointcloud", False)
        ),
        require_parametric_surface=(
            _configuration_value(
                hparams, "face_geometry_encoder", "mini_point"
            )
            == "param_surface"
        ),
        require_cylindrical_pos=bool(
            _configuration_value(hparams, "use_cylindrical_pos", False)
        ),
    )


def requires_instance_for_test(experiment_mode):
    """Only oracle inference consumes ground-truth instance annotations."""
    return experiment_mode == "oracle"


def supports_instance_evaluation(experiment_mode):
    """Dynamic modes expose predicted affinity logits at inference."""
    return experiment_mode in ("dynamic", "dynamic_affinity")


def seed_everything(seed):
    """Use the same project-wide deterministic seed setup as training."""
    from utils.reproducibility import seed_everything as seed_project

    seed_project(seed)


def _model_configuration_value(model, name, default):
    if hasattr(model, name):
        return getattr(model, name)
    return _configuration_value(getattr(model, "hparams", {}), name, default)


def apply_instance_bias_scale_override(model, override):
    """Apply an optional inference-only scale to the loaded relation bias."""
    encoder = getattr(model, "brep_encoder", None)
    relation_bias = getattr(encoder, "instance_relation_bias", None)
    if relation_bias is None:
        raise AttributeError(
            "Loaded model does not expose brep_encoder.instance_relation_bias"
        )

    checkpoint_scale = float(relation_bias.bias_scale)
    if override is None:
        return checkpoint_scale, checkpoint_scale, False

    effective_scale = float(override)
    if not math.isfinite(effective_scale) or effective_scale < 0.0:
        raise ValueError(
            "--instance-bias-scale-override must be finite and non-negative"
        )
    relation_bias.bias_scale = effective_scale
    return checkpoint_scale, effective_scale, True


def resolve_output_directory(
    checkpoint_path,
    requested_output_dir,
    scale_override,
):
    """Keep ordinary outputs stable and isolate scale-override comparisons."""
    if requested_output_dir is not None:
        return requested_output_dir
    if scale_override is None:
        return checkpoint_path.parent
    scale_label = format(float(scale_override), ".12g")
    return checkpoint_path.parent / ("test_bias_scale_" + scale_label)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=Path(r"D:\Code\Datasets\MFRInstSeg_BrepMFR"),
    )
    parser.add_argument("--shared-bin-cache-path", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--split",
        choices=("test", "val"),
        default="test",
        help="Use val only for a smoke test; formal evaluation uses test",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--num-classes", type=int, default=25)
    parser.add_argument(
        "--evaluate-instances",
        action="store_true",
        help="Evaluate dynamic affinity and topology-constrained instances",
    )
    parser.add_argument("--affinity-threshold", type=float, default=0.5)
    parser.add_argument("--background-class-id", type=int, default=24)
    parser.add_argument(
        "--instance-bias-scale-override",
        type=float,
        default=None,
        help=(
            "Inference-only instance bias scale; omit to use the value saved "
            "in the checkpoint and use 0 to disable bias injection"
        ),
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Default: checkpoint directory",
    )
    return parser.parse_args()


def select_device(requested):
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def move_batch(batch, device):
    moved = {}
    for name, value in batch.items():
        if hasattr(value, "to"):
            moved[name] = value.to(device)
        else:
            moved[name] = value
    return moved


def optional_float(value):
    if value is None or not np.isfinite(value):
        return None
    return float(value)


def metrics_from_confusion(confusion):
    confusion = confusion.astype(np.float64, copy=False)
    true_positive = np.diag(confusion)
    support = confusion.sum(axis=1)
    predicted = confusion.sum(axis=0)
    union = support + predicted - true_positive
    f1_denominator = support + predicted

    class_accuracy = np.divide(
        true_positive,
        support,
        out=np.full_like(true_positive, np.nan),
        where=support > 0,
    )
    class_iou = np.divide(
        true_positive,
        union,
        out=np.full_like(true_positive, np.nan),
        where=union > 0,
    )
    class_f1 = np.divide(
        2.0 * true_positive,
        f1_denominator,
        out=np.full_like(true_positive, np.nan),
        where=f1_denominator > 0,
    )
    return support, class_accuracy, class_iou, class_f1


def _validate_square_matrix(matrix, name):
    matrix = np.asarray(matrix)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("{} must be a square matrix".format(name))
    return matrix


def pairwise_counts(predicted, target, valid_nodes, threshold=0.5):
    """Count unordered valid face pairs for one graph."""
    predicted = _validate_square_matrix(predicted, "predicted affinity")
    target = _validate_square_matrix(target, "target affinity")
    if predicted.shape != target.shape:
        raise ValueError("predicted and target affinities must have equal shape")
    valid_nodes = np.asarray(valid_nodes, dtype=bool)
    if valid_nodes.shape != (predicted.shape[0],):
        raise ValueError("valid_nodes must have one entry per face")

    pair_mask = np.triu(
        valid_nodes[:, None] & valid_nodes[None, :],
        k=1,
    )
    predicted_pairs = np.asarray(predicted >= threshold, dtype=bool)[pair_mask]
    target_pairs = np.asarray(target > 0, dtype=bool)[pair_mask]
    return {
        "true_positive": int(np.count_nonzero(predicted_pairs & target_pairs)),
        "false_positive": int(np.count_nonzero(predicted_pairs & ~target_pairs)),
        "false_negative": int(np.count_nonzero(~predicted_pairs & target_pairs)),
    }


def pairwise_metrics(counts):
    """Convert accumulated TP/FP/FN counts into stable pairwise metrics."""
    true_positive = int(counts["true_positive"])
    false_positive = int(counts["false_positive"])
    false_negative = int(counts["false_negative"])
    precision_denominator = true_positive + false_positive
    recall_denominator = true_positive + false_negative
    precision = (
        true_positive / float(precision_denominator)
        if precision_denominator
        else 0.0
    )
    recall = (
        true_positive / float(recall_denominator)
        if recall_denominator
        else 0.0
    )
    f1_denominator = precision + recall
    return {
        "precision": precision,
        "recall": recall,
        "f1": (
            2.0 * precision * recall / f1_denominator
            if f1_denominator
            else 0.0
        ),
    }


def cluster_instances(
    affinity,
    semantic,
    threshold=0.5,
    background_class_id=24,
    adjacency=None,
):
    """Cluster same-semantic faces with deterministic union-find."""
    affinity = _validate_square_matrix(affinity, "affinity")
    semantic = np.asarray(semantic)
    node_count = affinity.shape[0]
    if semantic.shape != (node_count,):
        raise ValueError("semantic must have one entry per face")
    if adjacency is None:
        adjacency = np.ones_like(affinity, dtype=bool)
        np.fill_diagonal(adjacency, False)
    else:
        adjacency = _validate_square_matrix(adjacency, "adjacency")
        if adjacency.shape != affinity.shape:
            raise ValueError("adjacency and affinity must have equal shape")
        adjacency = np.asarray(adjacency, dtype=bool)

    parent = np.arange(node_count, dtype=np.int64)

    def find(node):
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = int(parent[node])
        return node

    def union(left, right):
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[max(left_root, right_root)] = min(left_root, right_root)

    active = semantic != background_class_id
    eligible = (
        (affinity >= threshold)
        & (adjacency | adjacency.T)
        & active[:, None]
        & active[None, :]
        & (semantic[:, None] == semantic[None, :])
    )
    for left, right in zip(*np.where(np.triu(eligible, k=1))):
        union(int(left), int(right))

    clusters = np.full(node_count, -1, dtype=np.int64)
    root_to_cluster = {}
    for node in range(node_count):
        if not active[node]:
            continue
        root = find(node)
        if root not in root_to_cluster:
            root_to_cluster[root] = len(root_to_cluster)
        clusters[node] = root_to_cluster[root]
    return clusters


def _cluster_affinity(clusters):
    clusters = np.asarray(clusters)
    return (
        (clusters[:, None] == clusters[None, :])
        & (clusters[:, None] >= 0)
        & (clusters[None, :] >= 0)
    )


def adjusted_rand_index(truth, predicted):
    """Compute ARI without adding a scikit-learn runtime dependency."""
    truth = np.asarray(truth)
    predicted = np.asarray(predicted)
    if truth.shape != predicted.shape or truth.ndim != 1:
        raise ValueError("truth and predicted must be equal-length vectors")
    sample_count = truth.size
    if sample_count < 2:
        return 1.0

    _, truth_inverse = np.unique(truth, return_inverse=True)
    _, predicted_inverse = np.unique(predicted, return_inverse=True)
    contingency = np.zeros(
        (
            int(truth_inverse.max()) + 1,
            int(predicted_inverse.max()) + 1,
        ),
        dtype=np.int64,
    )
    np.add.at(contingency, (truth_inverse, predicted_inverse), 1)

    def combinations_of_two(values):
        values = np.asarray(values, dtype=np.float64)
        return float(np.sum(values * (values - 1.0) / 2.0))

    index = combinations_of_two(contingency)
    truth_pairs = combinations_of_two(contingency.sum(axis=1))
    predicted_pairs = combinations_of_two(contingency.sum(axis=0))
    total_pairs = sample_count * (sample_count - 1.0) / 2.0
    expected = truth_pairs * predicted_pairs / total_pairs
    maximum = 0.5 * (truth_pairs + predicted_pairs)
    denominator = maximum - expected
    if abs(denominator) < 1e-12:
        return 1.0
    return float((index - expected) / denominator)


def clustering_metrics(
    predicted_clusters,
    target_affinity,
    ground_truth_semantic,
    background_class_id=24,
):
    """Compare one predicted partition with the annotation partition."""
    target_affinity = _validate_square_matrix(
        target_affinity, "target affinity"
    )
    ground_truth_semantic = np.asarray(ground_truth_semantic)
    predicted_clusters = np.asarray(predicted_clusters, dtype=np.int64)
    node_count = target_affinity.shape[0]
    if ground_truth_semantic.shape != (node_count,):
        raise ValueError("ground_truth_semantic must have one entry per face")
    if predicted_clusters.shape != (node_count,):
        raise ValueError("predicted_clusters must have one entry per face")

    valid_nodes = ground_truth_semantic != background_class_id
    ground_truth_clusters = cluster_instances(
        target_affinity,
        ground_truth_semantic,
        threshold=0.5,
        background_class_id=background_class_id,
    )
    counts = pairwise_counts(
        _cluster_affinity(predicted_clusters),
        _cluster_affinity(ground_truth_clusters),
        valid_nodes,
        threshold=0.5,
    )
    pairwise = dict(counts)
    pairwise.update(pairwise_metrics(counts))

    true_labels = ground_truth_clusters[valid_nodes]
    predicted_labels = predicted_clusters[valid_nodes].copy()
    next_label = int(predicted_labels[predicted_labels >= 0].max()) + 1 if bool(
        np.any(predicted_labels >= 0)
    ) else 0
    for index in np.flatnonzero(predicted_labels < 0):
        predicted_labels[index] = next_label
        next_label += 1

    predicted_count = int(np.unique(predicted_clusters[predicted_clusters >= 0]).size)
    ground_truth_count = int(
        np.unique(ground_truth_clusters[ground_truth_clusters >= 0]).size
    )
    return {
        "pairwise": pairwise,
        "adjusted_rand_index": adjusted_rand_index(
            true_labels, predicted_labels
        ),
        "predicted_instance_count": predicted_count,
        "ground_truth_instance_count": ground_truth_count,
        "instance_count_absolute_error": abs(
            predicted_count - ground_truth_count
        ),
    }


def evaluate_instance_graph(
    sample_id,
    affinity_logits,
    predicted_semantic,
    ground_truth_semantic,
    target_affinity,
    spatial_pos,
    threshold=0.5,
    background_class_id=24,
):
    """Evaluate and serialize one graph's predicted feature instances."""
    affinity_logits = _validate_square_matrix(
        affinity_logits, "affinity logits"
    )
    probability = 1.0 / (
        1.0 + np.exp(-np.clip(affinity_logits, -60.0, 60.0))
    )
    predicted_semantic = np.asarray(predicted_semantic, dtype=np.int64)
    ground_truth_semantic = np.asarray(
        ground_truth_semantic, dtype=np.int64
    )
    spatial_pos = _validate_square_matrix(spatial_pos, "spatial_pos")
    if spatial_pos.shape != affinity_logits.shape:
        raise ValueError("spatial_pos and affinity logits must have equal shape")

    valid_nodes = ground_truth_semantic != background_class_id
    raw_counts = pairwise_counts(
        probability,
        target_affinity,
        valid_nodes,
        threshold=threshold,
    )
    raw_pairwise = dict(raw_counts)
    raw_pairwise.update(pairwise_metrics(raw_counts))

    predicted_clusters = cluster_instances(
        probability,
        predicted_semantic,
        threshold=threshold,
        background_class_id=background_class_id,
        # collator.pad_spatial_pos_unsqueeze reserves zero for padding by
        # adding one to every real graph distance.  Direct neighbours are
        # therefore encoded as two in an evaluation batch.
        adjacency=spatial_pos == 2,
    )
    cluster_result = clustering_metrics(
        predicted_clusters,
        target_affinity,
        ground_truth_semantic,
        background_class_id=background_class_id,
    )
    instances = []
    for instance_id in np.unique(
        predicted_clusters[predicted_clusters >= 0]
    ):
        faces = np.flatnonzero(predicted_clusters == instance_id)
        instances.append(
            {
                "instance_id": int(instance_id),
                "semantic_class": int(predicted_semantic[faces[0]]),
                "face_indices": [int(face) for face in faces],
            }
        )
    return {
        "sample_id": str(sample_id),
        "raw_pairwise": raw_pairwise,
        "clustering": cluster_result,
        "instances": instances,
    }


def summarize_instance_results(records):
    """Aggregate per-graph instance metrics without graph-size weighting."""
    if not records:
        raise ValueError("at least one instance evaluation record is required")

    def accumulated_pairwise(key):
        counts = {
            "true_positive": 0,
            "false_positive": 0,
            "false_negative": 0,
        }
        for record in records:
            source = (
                record["raw_pairwise"]
                if key == "raw_pairwise"
                else record["clustering"]["pairwise"]
            )
            for name in counts:
                counts[name] += int(source[name])
        result = dict(counts)
        result.update(pairwise_metrics(counts))
        return result

    clustering = [record["clustering"] for record in records]
    return {
        "evaluated_graphs": len(records),
        "raw_pairwise": accumulated_pairwise("raw_pairwise"),
        "clustering_pairwise": accumulated_pairwise("clustering_pairwise"),
        "mean_adjusted_rand_index": float(
            np.mean(
                [item["adjusted_rand_index"] for item in clustering]
            )
        ),
        "instance_count_mae": float(
            np.mean(
                [
                    item["instance_count_absolute_error"]
                    for item in clustering
                ]
            )
        ),
        "predicted_instance_count": int(
            sum(item["predicted_instance_count"] for item in clustering)
        ),
        "ground_truth_instance_count": int(
            sum(item["ground_truth_instance_count"] for item in clustering)
        ),
    }


def write_results(output_dir, result):
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "test_metrics.json"
    log_path = output_dir / "test_log.txt"

    json_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    lines = [
        "BrepMFR test result",
        "checkpoint={}".format(result["checkpoint"]),
        "checkpoint_epoch={}".format(result["checkpoint_epoch"]),
        "experiment_mode={}".format(result["experiment_mode"]),
        "seed={}".format(result["seed"]),
        "prepared_dataset_path={}".format(result["prepared_dataset_path"]),
        "use_face_uv={}".format(result["use_face_uv"]),
        "use_face_pointcloud={}".format(result["use_face_pointcloud"]),
        "use_cylindrical_pos={}".format(result["use_cylindrical_pos"]),
        "face_geometry_encoder={}".format(
            result.get("face_geometry_encoder", "mini_point")
        ),
        "surface_neighbor_metric={}".format(
            result.get("surface_neighbor_metric", "intrinsic")
        ),
        "surface_trim_aware={}".format(
            result.get("surface_trim_aware", False)
        ),
        "surface_fusion={}".format(result.get("surface_fusion", "add")),
        "pointcloud_scale={}".format(result["pointcloud_scale"]),
        "cylindrical_scale={}".format(result["cylindrical_scale"]),
        "instance_bias_strategy={}".format(
            result["instance_bias_strategy"]
        ),
        "instance_relation_mapping={}".format(
            result["instance_relation_mapping"]
        ),
        "instance_bias_last_layers={}".format(
            result["instance_bias_last_layers"]
        ),
        "instance_bias_confidence_power={}".format(
            result["instance_bias_confidence_power"]
        ),
        "instance_bias_gate_init={}".format(
            result["instance_bias_gate_init"]
        ),
        "instance_bias_max_gate_scale={}".format(
            result["instance_bias_max_gate_scale"]
        ),
        "instance_bias_delay_epochs={}".format(
            result["instance_bias_delay_epochs"]
        ),
        "instance_bias_ramp_epochs={}".format(
            result["instance_bias_ramp_epochs"]
        ),
        "checkpoint_instance_bias_scale={}".format(
            result["checkpoint_instance_bias_scale"]
        ),
        "effective_instance_bias_scale={}".format(
            result["effective_instance_bias_scale"]
        ),
        "instance_bias_scale_overridden={}".format(
            result["instance_bias_scale_overridden"]
        ),
        "require_instance={}".format(result["require_instance"]),
        "instance_evaluation_requested={}".format(
            result.get("instance_evaluation_requested", False)
        ),
        "instance_evaluation_enabled={}".format(
            result.get("instance_evaluation_enabled", False)
        ),
        "instance_labels_used_by_model={}".format(
            result.get("instance_labels_used_by_model", False)
        ),
        "instance_labels_used_for_metrics={}".format(
            result.get("instance_labels_used_for_metrics", False)
        ),
        "batch_size={}".format(result["batch_size"]),
        "num_workers={}".format(result["num_workers"]),
        "device={}".format(result["device"]),
        "split={}".format(result["split"]),
        "samples={}".format(result["samples"]),
        "faces={}".format(result["faces"]),
        "elapsed_seconds={:.3f}".format(result["elapsed_seconds"]),
        "test_loss={:.8f}".format(result["test_loss"]),
        "per_face_accuracy={:.8f}".format(result["per_face_accuracy"]),
        "mean_iou={:.8f}".format(result["mean_iou"]),
        "macro_f1={:.8f}".format(result["macro_f1"]),
    ]
    instance_evaluation = result.get("instance_evaluation")
    if instance_evaluation is not None:
        raw = instance_evaluation["raw_pairwise"]
        clustered = instance_evaluation["clustering_pairwise"]
        lines.extend(
            [
                "affinity_threshold={:.8f}".format(
                    result["affinity_threshold"]
                ),
                "background_class_id={}".format(
                    result["background_class_id"]
                ),
                "affinity_pairwise_precision={:.8f}".format(
                    raw["precision"]
                ),
                "affinity_pairwise_recall={:.8f}".format(raw["recall"]),
                "affinity_pairwise_f1={:.8f}".format(raw["f1"]),
                "cluster_pairwise_precision={:.8f}".format(
                    clustered["precision"]
                ),
                "cluster_pairwise_recall={:.8f}".format(
                    clustered["recall"]
                ),
                "cluster_pairwise_f1={:.8f}".format(clustered["f1"]),
                "mean_adjusted_rand_index={:.8f}".format(
                    instance_evaluation["mean_adjusted_rand_index"]
                ),
                "instance_count_mae={:.8f}".format(
                    instance_evaluation["instance_count_mae"]
                ),
                "predicted_instance_count={}".format(
                    instance_evaluation["predicted_instance_count"]
                ),
                "ground_truth_instance_count={}".format(
                    instance_evaluation["ground_truth_instance_count"]
                ),
            ]
        )
    lines.extend(["", "per_class_metrics:"])
    for item in result["per_class"]:
        lines.append(
            "class_{class_id:02d} support={support} accuracy={accuracy} "
            "iou={iou} f1={f1}".format(
                class_id=item["class_id"],
                support=item["support"],
                accuracy=(
                    "{:.8f}".format(item["accuracy"])
                    if item["accuracy"] is not None
                    else "NA"
                ),
                iou=(
                    "{:.8f}".format(item["iou"])
                    if item["iou"] is not None
                    else "NA"
                ),
                f1=(
                    "{:.8f}".format(item["f1"])
                    if item["f1"] is not None
                    else "NA"
                ),
            )
        )
    log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return log_path, json_path


def write_instance_predictions(output_dir, records):
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "predicted_instances.json"
    document = [
        {
            "sample_id": record["sample_id"],
            "instances": record["instances"],
        }
        for record in records
    ]
    path.write_text(
        json.dumps(document, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def main():
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    if args.num_workers < 0:
        raise ValueError("--num-workers must be non-negative")
    if not 0.0 <= args.affinity_threshold <= 1.0:
        raise ValueError("--affinity-threshold must be between 0 and 1")
    if not 0 <= args.background_class_id < args.num_classes:
        raise ValueError(
            "--background-class-id must be within the semantic class range"
        )
    if not args.checkpoint.is_file():
        raise FileNotFoundError("Checkpoint not found: {}".format(args.checkpoint))
    manifest = args.dataset_path / (args.split + ".txt")
    if not manifest.is_file():
        raise FileNotFoundError(
            "Prepared BIN manifest not found: {}. Prepare the shared dataset cache first.".format(
                manifest
            )
        )

    seed_everything(args.seed)
    device = select_device(args.device)

    checkpoint_document = torch.load(args.checkpoint, map_location="cpu")
    checkpoint_epoch = (
        int(checkpoint_document.get("epoch", -1)) + 1
        if isinstance(checkpoint_document, Mapping)
        else -1
    )
    model = BrepSeg.load_from_checkpoint(
        str(args.checkpoint),
        map_location="cpu",
    )
    (
        checkpoint_instance_bias_scale,
        effective_instance_bias_scale,
        instance_bias_scale_overridden,
    ) = apply_instance_bias_scale_override(
        model,
        args.instance_bias_scale_override,
    )
    mode = _model_configuration_value(model, "experiment_mode", "baseline")
    requirements = geometry_requirements(model)
    pointcloud_scale = float(
        _model_configuration_value(model, "pointcloud_scale", 1.0)
    )
    cylindrical_scale = float(
        _model_configuration_value(model, "cylindrical_scale", 1.0)
    )
    face_geometry_encoder = str(
        _model_configuration_value(
            model, "face_geometry_encoder", "mini_point"
        )
    )
    surface_neighbor_metric = str(
        _model_configuration_value(
            model, "surface_neighbor_metric", "intrinsic"
        )
    )
    surface_trim_aware = bool(
        _model_configuration_value(model, "surface_trim_aware", False)
    )
    surface_fusion = str(
        _model_configuration_value(model, "surface_fusion", "add")
    )
    instance_bias_strategy = str(
        _model_configuration_value(model, "instance_bias_strategy", "legacy")
    )
    instance_relation_mapping = str(
        _model_configuration_value(
            model,
            "instance_relation_mapping",
            "signed",
        )
    )
    instance_bias_last_layers = int(
        _model_configuration_value(model, "instance_bias_last_layers", 2)
    )
    instance_bias_confidence_power = float(
        _model_configuration_value(
            model,
            "instance_bias_confidence_power",
            1.0,
        )
    )
    instance_bias_gate_init = float(
        _model_configuration_value(model, "instance_bias_gate_init", 0.05)
    )
    instance_bias_max_gate_scale = float(
        _model_configuration_value(
            model,
            "instance_bias_max_gate_scale",
            1.0,
        )
    )
    instance_bias_delay_epochs = int(
        _model_configuration_value(
            model,
            "instance_bias_delay_epochs",
            10,
        )
    )
    instance_bias_ramp_epochs = int(
        _model_configuration_value(
            model,
            "instance_bias_ramp_epochs",
            10,
        )
    )
    instance_evaluation_enabled = (
        args.evaluate_instances and supports_instance_evaluation(mode)
    )
    if args.evaluate_instances and not instance_evaluation_enabled:
        print(
            "Instance evaluation skipped: checkpoint mode '{}' does not "
            "produce dynamic affinity logits.".format(mode)
        )
    require_instance = (
        requires_instance_for_test(mode) or instance_evaluation_enabled
    )
    dataset = CADSynth(
        root_dir=args.dataset_path,
        bin_root=args.shared_bin_cache_path or args.dataset_path,
        split=args.split,
        random_rotate=False,
        num_class=args.num_classes,
        require_instance=require_instance,
        require_face_pointcloud=requirements.require_face_pointcloud,
        require_parametric_surface=requirements.require_parametric_surface,
        require_cylindrical_pos=requirements.require_cylindrical_pos,
        seed=args.seed,
    )
    if len(dataset) == 0:
        raise RuntimeError("No BIN files were loaded for split {}".format(args.split))
    loader = dataset.get_dataloader(
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        seed_purpose="test_loader",
    )

    model.to(device)
    model.eval()
    confusion = torch.zeros(
        (args.num_classes, args.num_classes), dtype=torch.long
    )
    loss_sum = 0.0
    face_count = 0
    instance_records = []
    started_at = time.perf_counter()

    with torch.inference_mode():
        for batch in loader:
            batch = move_batch(batch, device)
            probabilities, affinity_logits = model.predict_with_affinity(batch)
            labels = batch["label_feature"].long()
            predictions = probabilities.argmax(dim=-1)
            loss_sum += float(
                (
                    -torch.log(
                        probabilities.gather(1, labels.unsqueeze(1)).clamp_min(1e-12)
                    )
                ).sum().item()
            )
            face_count += int(labels.numel())
            indices = (
                labels * args.num_classes + predictions
            ).detach().cpu()
            confusion += torch.bincount(
                indices,
                minlength=args.num_classes * args.num_classes,
            ).reshape(args.num_classes, args.num_classes)

            if instance_evaluation_enabled:
                if affinity_logits is None:
                    raise RuntimeError(
                        "Dynamic checkpoint did not produce affinity logits"
                    )
                counts = (~batch["padding_mask"]).sum(dim=1).tolist()
                offset = 0
                for graph_index, node_count in enumerate(counts):
                    node_count = int(node_count)
                    next_offset = offset + node_count
                    instance_records.append(
                        evaluate_instance_graph(
                            sample_id=batch["data_ids"][graph_index],
                            affinity_logits=affinity_logits[
                                graph_index, :node_count, :node_count
                            ]
                            .detach()
                            .cpu()
                            .numpy(),
                            predicted_semantic=predictions[
                                offset:next_offset
                            ]
                            .detach()
                            .cpu()
                            .numpy(),
                            ground_truth_semantic=labels[
                                offset:next_offset
                            ]
                            .detach()
                            .cpu()
                            .numpy(),
                            target_affinity=batch["instance"][
                                graph_index, :node_count, :node_count
                            ]
                            .detach()
                            .cpu()
                            .numpy(),
                            spatial_pos=batch["spatial_pos"][
                                graph_index, :node_count, :node_count
                            ]
                            .detach()
                            .cpu()
                            .numpy(),
                            threshold=args.affinity_threshold,
                            background_class_id=args.background_class_id,
                        )
                    )
                    offset = next_offset

    elapsed_seconds = time.perf_counter() - started_at
    confusion_np = confusion.numpy()
    support, class_accuracy, class_iou, class_f1 = metrics_from_confusion(
        confusion_np
    )
    correct = float(np.diag(confusion_np).sum())
    per_class = []
    for class_id in range(args.num_classes):
        per_class.append(
            {
                "class_id": class_id,
                "support": int(support[class_id]),
                "accuracy": optional_float(class_accuracy[class_id]),
                "iou": optional_float(class_iou[class_id]),
                "f1": optional_float(class_f1[class_id]),
            }
        )

    result = {
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_epoch": checkpoint_epoch,
        "experiment_mode": mode,
        "seed": args.seed,
        "prepared_dataset_path": str(args.dataset_path.resolve()),
        "bin_cache_path": str(
            (args.shared_bin_cache_path or args.dataset_path).resolve()
        ),
        "use_face_uv": requirements.use_face_uv,
        "use_face_pointcloud": requirements.require_face_pointcloud,
        "use_cylindrical_pos": requirements.require_cylindrical_pos,
        "face_geometry_encoder": face_geometry_encoder,
        "surface_neighbor_metric": surface_neighbor_metric,
        "surface_trim_aware": surface_trim_aware,
        "surface_fusion": surface_fusion,
        "pointcloud_scale": pointcloud_scale,
        "cylindrical_scale": cylindrical_scale,
        "instance_bias_strategy": instance_bias_strategy,
        "instance_relation_mapping": instance_relation_mapping,
        "instance_bias_last_layers": instance_bias_last_layers,
        "instance_bias_confidence_power": (
            instance_bias_confidence_power
        ),
        "instance_bias_gate_init": instance_bias_gate_init,
        "instance_bias_max_gate_scale": instance_bias_max_gate_scale,
        "instance_bias_delay_epochs": instance_bias_delay_epochs,
        "instance_bias_ramp_epochs": instance_bias_ramp_epochs,
        "checkpoint_instance_bias_scale": checkpoint_instance_bias_scale,
        "effective_instance_bias_scale": effective_instance_bias_scale,
        "instance_bias_scale_overridden": instance_bias_scale_overridden,
        "require_instance": require_instance,
        "instance_evaluation_requested": args.evaluate_instances,
        "instance_evaluation_enabled": instance_evaluation_enabled,
        "instance_labels_used_by_model": mode == "oracle",
        "instance_labels_used_for_metrics": instance_evaluation_enabled,
        "affinity_threshold": args.affinity_threshold,
        "background_class_id": args.background_class_id,
        "instance_evaluation": (
            summarize_instance_results(instance_records)
            if instance_evaluation_enabled
            else None
        ),
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "device": str(device),
        "split": args.split,
        "samples": len(dataset),
        "faces": face_count,
        "elapsed_seconds": elapsed_seconds,
        "test_loss": loss_sum / max(face_count, 1),
        "per_face_accuracy": correct / max(face_count, 1),
        "mean_iou": float(np.nanmean(class_iou)),
        "macro_f1": float(np.nanmean(class_f1)),
        "per_class": per_class,
        "confusion_matrix": confusion_np.astype(int).tolist(),
    }
    output_dir = resolve_output_directory(
        args.checkpoint,
        args.output_dir,
        args.instance_bias_scale_override,
    )
    prediction_path = None
    if instance_evaluation_enabled:
        prediction_path = write_instance_predictions(
            output_dir, instance_records
        )
        result["predicted_instances_path"] = str(prediction_path.resolve())
    log_path, json_path = write_results(output_dir, result)
    print("Test completed: {}".format(args.split))
    print("Samples: {} Faces: {}".format(result["samples"], result["faces"]))
    print("Accuracy: {:.4%}".format(result["per_face_accuracy"]))
    print("mIoU: {:.4%}".format(result["mean_iou"]))
    print("Macro-F1: {:.4%}".format(result["macro_f1"]))
    if result["instance_evaluation"] is not None:
        instance_metrics = result["instance_evaluation"]
        print(
            "Affinity Pairwise-F1: {:.4%}".format(
                instance_metrics["raw_pairwise"]["f1"]
            )
        )
        print(
            "Cluster Pairwise-F1: {:.4%} ARI: {:.4f} Count-MAE: {:.4f}".format(
                instance_metrics["clustering_pairwise"]["f1"],
                instance_metrics["mean_adjusted_rand_index"],
                instance_metrics["instance_count_mae"],
            )
        )
        print("Instances: {}".format(prediction_path))
    print("Log: {}".format(log_path))
    print("JSON: {}".format(json_path))


if __name__ == "__main__":
    main()
