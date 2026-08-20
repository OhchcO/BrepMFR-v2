import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.brepseg_model import (
    BrepSeg,
    balanced_affinity_loss,
    instance_bias_progress,
)
from models.modules.instance_bias import InstanceRelationBias
from training_callbacks import TextMetricsLogger


def test_structured_affinity_loss_balances_positive_hard_and_easy_groups():
    logits = torch.tensor(
        [
            [
                [0.0, 0.0, 1.0, -1.0],
                [0.0, 0.0, 3.0, -2.0],
                [1.0, 3.0, 0.0, -3.0],
                [-1.0, -2.0, -3.0, 0.0],
            ]
        ]
    )
    target = torch.zeros(1, 4, 4)
    target[0, 0, 1] = 1.0
    target[0, 1, 0] = 1.0
    padding_mask = torch.zeros(1, 4, dtype=torch.bool)
    semantic_labels = torch.tensor([5, 5, 5, 9])

    loss = balanced_affinity_loss(
        logits,
        target,
        padding_mask,
        available=torch.tensor([True]),
        semantic_labels=semantic_labels,
    )

    positive = F.softplus(torch.tensor(0.0))
    hard_negative = F.softplus(torch.tensor([1.0, 3.0])).mean()
    easy_negative = F.softplus(torch.tensor([-1.0, -2.0, -3.0])).mean()
    expected = torch.stack(
        [positive, hard_negative, easy_negative]
    ).mean()
    assert loss.item() == pytest.approx(expected.item())


def test_structured_affinity_loss_reconstructs_flat_labels_across_graphs():
    logits = torch.zeros(2, 3, 3)
    target = torch.zeros(2, 3, 3)
    target[0, 0, 1] = target[0, 1, 0] = 1.0
    target[1, 0, 1] = target[1, 1, 0] = 1.0
    padding_mask = torch.tensor(
        [
            [False, False, True],
            [False, False, False],
        ]
    )
    semantic_labels = torch.tensor([1, 1, 2, 2, 3])

    loss = balanced_affinity_loss(
        logits,
        target,
        padding_mask,
        semantic_labels=semantic_labels,
    )

    assert loss.item() == pytest.approx(float(torch.log(torch.tensor(2.0))))


@pytest.mark.parametrize(
    ("epoch", "expected"),
    [
        (0, 0.0),
        (1, 0.0),
        (2, 1.0 / 3.0),
        (3, 2.0 / 3.0),
        (4, 1.0),
        (20, 1.0),
    ],
)
def test_instance_bias_progress_has_delay_then_linear_ramp(epoch, expected):
    assert instance_bias_progress(epoch, delay_epochs=2, ramp_epochs=3) == (
        pytest.approx(expected)
    )


def test_instance_bias_progress_supports_immediate_enable_and_validates_inputs():
    assert instance_bias_progress(0, delay_epochs=0, ramp_epochs=0) == 1.0
    with pytest.raises(ValueError, match="epoch"):
        instance_bias_progress(-1, delay_epochs=0, ramp_epochs=1)
    with pytest.raises(ValueError, match="delay"):
        instance_bias_progress(0, delay_epochs=-1, ramp_epochs=1)
    with pytest.raises(ValueError, match="ramp"):
        instance_bias_progress(0, delay_epochs=0, ramp_epochs=-1)


def test_text_training_log_records_instance_bias_progress(tmp_path):
    logger = TextMetricsLogger(tmp_path / "log.txt", configuration={})

    class Trainer:
        current_epoch = 3
        global_step = 7
        callback_metrics = {
            "train_instance_bias_progress": torch.tensor(0.5),
            "train_instance_bias_gate": torch.tensor(0.125),
        }
        logged_metrics = {}

    logger.on_train_epoch_end(Trainer(), pl_module=None)

    assert "train_instance_bias_progress=0.50000000" in (
        tmp_path / "log.txt"
    ).read_text(encoding="utf-8")
    assert "train_instance_bias_gate=0.12500000" in (
        tmp_path / "log.txt"
    ).read_text(encoding="utf-8")


def test_text_validation_log_records_mean_class_accuracy(tmp_path):
    logger = TextMetricsLogger(tmp_path / "log.txt", configuration={})

    class Trainer:
        current_epoch = 2
        global_step = 5
        sanity_checking = False
        callback_metrics = {
            "mean_class_accuracy": torch.tensor(0.75),
        }
        logged_metrics = {}

    logger.on_validation_epoch_end(Trainer(), pl_module=None)

    assert "mean_class_accuracy=0.75000000" in (
        tmp_path / "log.txt"
    ).read_text(encoding="utf-8")


def test_training_step_logs_gate_applied_at_current_progress():
    relation_bias = InstanceRelationBias(
        embedding_dim=2,
        num_heads=1,
        mode="oracle",
        strategy="late_gated",
        gate_init=0.5,
        max_gate_scale=1.0,
    )

    class Encoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.instance_relation_bias = relation_bias

        def forward(self, batch, **kwargs):
            node_state = torch.tensor(
                [
                    [[0.0, 0.0]],
                    [[2.0, 0.0]],
                    [[0.0, 2.0]],
                ]
            )
            return [node_state], torch.zeros(1, 2), None

    class SelectNodeFeatures(nn.Module):
        def forward(self, inputs):
            return inputs[0]

    class TwoClassProbabilities(nn.Module):
        def forward(self, inputs):
            return torch.softmax(inputs, dim=-1)

    logged = {}

    class TrainingHarness:
        num_classes = 2
        experiment_mode = "baseline"
        brep_encoder = Encoder()
        attention = SelectNodeFeatures()
        classifier = TwoClassProbabilities()

        def _instance_bias_progress(self):
            return 0.25

        def log(self, name, value, **kwargs):
            logged[name] = value

    BrepSeg.training_step(
        TrainingHarness(),
        {
            "padding_mask": torch.zeros(1, 2, dtype=torch.bool),
            "label_feature": torch.tensor([0, 1]),
        },
        batch_idx=0,
    )

    assert logged["train_instance_bias_gate"].item() == pytest.approx(0.125)


def test_validation_logs_mean_class_accuracy_without_absent_classes():
    logged = {}

    class ValidationHarness:
        num_classes = 3
        pred = [0, 1, 1, 1]
        label = [0, 0, 1, 1]

        def log(self, name, value, **kwargs):
            logged[name] = value

    BrepSeg.validation_epoch_end(ValidationHarness(), val_step_outputs=[])

    assert logged["per_face_accuracy"] == pytest.approx(0.75)
    assert logged["mean_class_accuracy"] == pytest.approx(0.75)
