from types import SimpleNamespace

import numpy as np
import pytest
import sys

from tools import test_mfrinstseg
from tools.test_mfrinstseg import (
    adjusted_rand_index,
    cluster_instances,
    clustering_metrics,
    evaluate_instance_graph,
    pairwise_counts,
    pairwise_metrics,
    semantic_metrics_from_confusion,
    summarize_instance_results,
)


def _model_with_instance_bias_scale(scale):
    relation_bias = SimpleNamespace(bias_scale=scale)
    encoder = SimpleNamespace(instance_relation_bias=relation_bias)
    return SimpleNamespace(brep_encoder=encoder), relation_bias


def test_semantic_metrics_include_overall_and_mean_class_accuracy():
    metrics = semantic_metrics_from_confusion(
        np.array(
            [
                [1, 1, 0],
                [0, 2, 0],
                [0, 0, 0],
            ]
        )
    )

    assert metrics["per_face_accuracy"] == pytest.approx(0.75)
    assert metrics["mean_class_accuracy"] == pytest.approx(0.75)
    assert metrics["mean_iou"] == pytest.approx((0.5 + 2.0 / 3.0) / 2.0)
    assert metrics["macro_f1"] == pytest.approx((2.0 / 3.0 + 0.8) / 2.0)


def test_instance_bias_scale_override_none_preserves_checkpoint_value():
    model, relation_bias = _model_with_instance_bias_scale(1.0)

    values = test_mfrinstseg.apply_instance_bias_scale_override(model, None)

    assert values == (1.0, 1.0, False)
    assert relation_bias.bias_scale == 1.0


def test_evaluation_parser_accepts_separate_shared_bin_root(monkeypatch, tmp_path):
    checkpoint = tmp_path / "best.ckpt"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "test_mfrinstseg.py",
            "--checkpoint",
            str(checkpoint),
            "--dataset-path",
            str(tmp_path / "ten_pct"),
            "--shared-bin-cache-path",
            str(tmp_path / "shared"),
        ],
    )

    args = test_mfrinstseg.parse_args()

    assert args.dataset_path == tmp_path / "ten_pct"
    assert args.shared_bin_cache_path == tmp_path / "shared"


def test_instance_bias_scale_override_zero_disables_loaded_bias():
    model, relation_bias = _model_with_instance_bias_scale(1.0)

    values = test_mfrinstseg.apply_instance_bias_scale_override(model, 0.0)

    assert values == (1.0, 0.0, True)
    assert relation_bias.bias_scale == 0.0


def test_instance_bias_scale_override_accepts_positive_value():
    model, relation_bias = _model_with_instance_bias_scale(0.5)

    values = test_mfrinstseg.apply_instance_bias_scale_override(model, 0.25)

    assert values == (0.5, 0.25, True)
    assert relation_bias.bias_scale == 0.25


@pytest.mark.parametrize("override", [-0.1, float("nan"), float("inf")])
def test_instance_bias_scale_override_rejects_invalid_value(override):
    model, relation_bias = _model_with_instance_bias_scale(1.0)

    with pytest.raises(ValueError, match="finite and non-negative"):
        test_mfrinstseg.apply_instance_bias_scale_override(model, override)

    assert relation_bias.bias_scale == 1.0


def test_instance_bias_scale_override_requires_relation_bias_module():
    model = SimpleNamespace(brep_encoder=SimpleNamespace())

    with pytest.raises(AttributeError, match="instance_relation_bias"):
        test_mfrinstseg.apply_instance_bias_scale_override(model, 0.0)


def test_bias_override_uses_scale_specific_output_directory(tmp_path):
    checkpoint = tmp_path / "best.ckpt"

    output_dir = test_mfrinstseg.resolve_output_directory(
        checkpoint,
        requested_output_dir=None,
        scale_override=0.0,
    )

    assert output_dir == tmp_path / "test_bias_scale_0"


def test_saved_scale_evaluation_preserves_legacy_output_directory(tmp_path):
    checkpoint = tmp_path / "best.ckpt"

    output_dir = test_mfrinstseg.resolve_output_directory(
        checkpoint,
        requested_output_dir=None,
        scale_override=None,
    )

    assert output_dir == tmp_path


def test_explicit_output_directory_takes_priority(tmp_path):
    checkpoint = tmp_path / "best.ckpt"
    requested = tmp_path / "comparison"

    output_dir = test_mfrinstseg.resolve_output_directory(
        checkpoint,
        requested_output_dir=requested,
        scale_override=0.25,
    )

    assert output_dir == requested


def test_pairwise_metrics_exclude_background_and_use_upper_triangle_once():
    target = np.array(
        [
            [0, 1, 0, 0],
            [1, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ],
        dtype=bool,
    )
    probability = np.array(
        [
            [0.0, 0.9, 0.8, 0.9],
            [0.9, 0.0, 0.1, 0.9],
            [0.8, 0.1, 0.0, 0.9],
            [0.9, 0.9, 0.9, 0.0],
        ]
    )
    semantic = np.array([1, 1, 2, 24])

    counts = pairwise_counts(
        probability,
        target,
        semantic != 24,
        threshold=0.5,
    )

    assert counts == {"true_positive": 1, "false_positive": 1, "false_negative": 0}
    assert pairwise_metrics(counts) == {
        "precision": 0.5,
        "recall": 1.0,
        "f1": 2.0 / 3.0,
    }


def test_predicted_clustering_requires_topology_semantics_and_affinity():
    probability = np.array(
        [
            [0.0, 0.9, 0.95, 0.0],
            [0.9, 0.0, 0.9, 0.0],
            [0.95, 0.9, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ]
    )
    adjacency = np.array(
        [
            [0, 1, 0, 0],
            [1, 0, 1, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 0],
        ],
        dtype=bool,
    )
    predicted_semantic = np.array([3, 3, 4, 24])

    clusters = cluster_instances(
        probability,
        predicted_semantic,
        threshold=0.5,
        background_class_id=24,
        adjacency=adjacency,
    )

    assert clusters.tolist() == [0, 0, 1, -1]


def test_clustering_metrics_report_perfect_partition_and_instance_count():
    target_affinity = np.array(
        [
            [0, 1, 0, 0],
            [1, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ],
        dtype=bool,
    )
    gt_semantic = np.array([3, 3, 4, 24])
    predicted_clusters = np.array([0, 0, 1, -1])

    result = clustering_metrics(
        predicted_clusters,
        target_affinity,
        gt_semantic,
        background_class_id=24,
    )

    assert result["pairwise"] == {
        "true_positive": 1,
        "false_positive": 0,
        "false_negative": 0,
        "precision": 1.0,
        "recall": 1.0,
        "f1": 1.0,
    }
    assert result["adjusted_rand_index"] == 1.0
    assert result["predicted_instance_count"] == 2
    assert result["ground_truth_instance_count"] == 2
    assert result["instance_count_absolute_error"] == 0


def test_adjusted_rand_index_penalizes_merged_instances():
    truth = np.array([0, 0, 1, 1])
    predicted = np.array([0, 0, 0, 0])

    assert adjusted_rand_index(truth, predicted) == 0.0


def test_instance_graph_uses_collated_direct_neighbor_distance():
    logits = np.array(
        [
            [0.0, 4.0, -4.0],
            [4.0, 0.0, -4.0],
            [-4.0, -4.0, 0.0],
        ]
    )
    target = np.array(
        [
            [0, 1, 0],
            [1, 0, 0],
            [0, 0, 0],
        ]
    )
    spatial_pos = np.array(
        [
            [1, 2, 3],
            [2, 1, 2],
            [3, 2, 1],
        ]
    )

    result = evaluate_instance_graph(
        sample_id="part-1",
        affinity_logits=logits,
        predicted_semantic=np.array([5, 5, 24]),
        ground_truth_semantic=np.array([5, 5, 24]),
        target_affinity=target,
        spatial_pos=spatial_pos,
        threshold=0.5,
        background_class_id=24,
    )

    assert result["sample_id"] == "part-1"
    assert result["instances"] == [
        {
            "instance_id": 0,
            "semantic_class": 5,
            "face_indices": [0, 1],
        }
    ]
    assert result["raw_pairwise"]["f1"] == 1.0
    assert result["clustering"]["adjusted_rand_index"] == 1.0


def test_instance_summary_micro_averages_pairs_and_means_graph_metrics():
    records = [
        {
            "raw_pairwise": {
                "true_positive": 2,
                "false_positive": 0,
                "false_negative": 0,
            },
            "clustering": {
                "pairwise": {
                    "true_positive": 1,
                    "false_positive": 0,
                    "false_negative": 1,
                },
                "adjusted_rand_index": 0.5,
                "predicted_instance_count": 3,
                "ground_truth_instance_count": 2,
                "instance_count_absolute_error": 1,
            },
        },
        {
            "raw_pairwise": {
                "true_positive": 1,
                "false_positive": 1,
                "false_negative": 0,
            },
            "clustering": {
                "pairwise": {
                    "true_positive": 1,
                    "false_positive": 1,
                    "false_negative": 0,
                },
                "adjusted_rand_index": 1.0,
                "predicted_instance_count": 1,
                "ground_truth_instance_count": 1,
                "instance_count_absolute_error": 0,
            },
        },
    ]

    summary = summarize_instance_results(records)

    assert summary["raw_pairwise"]["true_positive"] == 3
    assert summary["raw_pairwise"]["f1"] == 6.0 / 7.0
    assert summary["clustering_pairwise"]["f1"] == 2.0 / 3.0
    assert summary["mean_adjusted_rand_index"] == 0.75
    assert summary["instance_count_mae"] == 0.5
    assert summary["predicted_instance_count"] == 4
    assert summary["ground_truth_instance_count"] == 3
