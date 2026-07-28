import pytest
import torch

from models.modules.instance_bias import (
    InstanceRelationBias,
    confidence_weighted_relation,
)


def relation_inputs():
    return {
        "node_features": torch.tensor(
            [
                [
                    [0.0, 0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                ]
            ]
        ),
        "padding_mask": torch.tensor([[False, False]]),
        "spatial_pos": torch.zeros(1, 2, 2, dtype=torch.long),
        "d2_distance": torch.zeros(1, 2, 2, 64),
        "angle_distance": torch.zeros(1, 2, 2, 64),
    }


def test_confidence_weighted_relation_is_signed_and_suppresses_uncertainty():
    logits = torch.tensor(
        [[-4.0, 0.0, 4.0]],
        dtype=torch.float32,
    )

    relation = confidence_weighted_relation(logits, power=1.0)

    assert relation[0, 0] < -0.9
    assert relation[0, 1].item() == pytest.approx(0.0, abs=1e-7)
    assert relation[0, 2] > 0.9


def test_confidence_power_reduces_moderate_relation_magnitude():
    logits = torch.tensor([1.0])

    linear = confidence_weighted_relation(logits, power=0.0)
    confidence_gated = confidence_weighted_relation(logits, power=2.0)

    assert 0.0 < confidence_gated.item() < linear.item() < 1.0


def test_positive_only_dynamic_mapping_clamps_negatives_and_preserves_positives():
    logits = torch.tensor([-4.0, 0.0, 4.0])

    default = confidence_weighted_relation(logits, power=1.0)
    signed = confidence_weighted_relation(
        logits,
        power=1.0,
        relation_mapping="signed",
    )
    positive = confidence_weighted_relation(
        logits,
        power=1.0,
        relation_mapping="positive_only",
    )

    assert torch.equal(default, signed)
    assert positive[0].item() == 0.0
    assert positive[1].item() == 0.0
    assert positive[2].item() == pytest.approx(signed[2].item())


def test_confidence_weighted_relation_rejects_unknown_mapping():
    with pytest.raises(ValueError, match="relation mapping"):
        confidence_weighted_relation(
            torch.tensor([1.0]),
            relation_mapping="unknown",
        )


def test_positive_only_oracle_keeps_different_instances_neutral():
    signed_module = InstanceRelationBias(
        embedding_dim=4,
        num_heads=1,
        mode="oracle",
        strategy="late_gated",
        relation_mapping="signed",
        gate_init=0.5,
        max_gate_scale=1.0,
    )
    positive_module = InstanceRelationBias(
        embedding_dim=4,
        num_heads=1,
        mode="oracle",
        strategy="late_gated",
        relation_mapping="positive_only",
        gate_init=0.5,
        max_gate_scale=1.0,
    )
    inputs = relation_inputs()
    inputs["instance"] = torch.zeros(1, 2, 2)
    inputs["instance_available"] = torch.tensor([True])

    signed_bias, _ = signed_module(**inputs, progress=1.0)
    positive_bias, _ = positive_module(**inputs, progress=1.0)

    assert signed_bias[0, 0, 1, 2].item() == pytest.approx(-0.5)
    assert positive_bias[0, 0, 1, 2].item() == 0.0


def test_instance_relation_bias_rejects_unknown_mapping():
    with pytest.raises(ValueError, match="relation mapping"):
        InstanceRelationBias(
            embedding_dim=4,
            num_heads=1,
            mode="oracle",
            strategy="late_gated",
            relation_mapping="unknown",
        )


def test_late_gated_scale_zero_keeps_logits_but_disables_bias():
    module = InstanceRelationBias(
        embedding_dim=4,
        num_heads=2,
        mode="dynamic_affinity",
        affinity_dim=4,
        strategy="late_gated",
        bias_scale=0.0,
    )

    bias, logits = module(**relation_inputs(), progress=1.0)

    assert logits.shape == (1, 2, 2)
    assert torch.count_nonzero(bias).item() == 0


def test_late_gated_progress_scales_bias_and_rejects_invalid_values():
    module = InstanceRelationBias(
        embedding_dim=4,
        num_heads=2,
        mode="oracle",
        strategy="late_gated",
        bias_scale=1.0,
        gate_init=0.5,
        max_gate_scale=1.0,
    )
    inputs = relation_inputs()
    inputs["instance"] = torch.tensor(
        [[[0.0, 1.0], [1.0, 0.0]]]
    )
    inputs["instance_available"] = torch.tensor([True])

    zero_bias, _ = module(**inputs, progress=0.0)
    full_bias, _ = module(**inputs, progress=1.0)

    assert torch.count_nonzero(zero_bias).item() == 0
    assert full_bias[0, 0, 1, 2].item() == pytest.approx(0.5)
    with pytest.raises(ValueError, match="progress"):
        module(**inputs, progress=1.1)


def test_dynamic_affinity_semantic_bias_does_not_update_affinity_head():
    torch.manual_seed(2026)
    module = InstanceRelationBias(
        embedding_dim=4,
        num_heads=2,
        mode="dynamic_affinity",
        affinity_dim=4,
        strategy="late_gated",
        bias_scale=1.0,
        gate_init=0.5,
        max_gate_scale=1.0,
    )

    bias, _ = module(**relation_inputs(), progress=1.0)
    bias.sum().backward()

    affinity_parameters = [
        parameter
        for name, parameter in module.named_parameters()
        if name != "gate_logit"
    ]
    assert module.gate_logit.grad is not None
    assert all(parameter.grad is None for parameter in affinity_parameters)


def test_dynamic_without_affinity_loss_can_learn_predictor_from_semantics():
    torch.manual_seed(2026)
    module = InstanceRelationBias(
        embedding_dim=4,
        num_heads=2,
        mode="dynamic",
        affinity_dim=4,
        strategy="late_gated",
        bias_scale=1.0,
        gate_init=0.5,
        max_gate_scale=1.0,
    )

    bias, _ = module(**relation_inputs(), progress=1.0)
    bias.sum().backward()

    assert any(
        parameter.grad is not None
        for name, parameter in module.named_parameters()
        if name != "gate_logit"
    )


def test_legacy_strategy_retains_non_negative_probability_bias():
    module = InstanceRelationBias(
        embedding_dim=4,
        num_heads=1,
        mode="dynamic",
        affinity_dim=4,
        strategy="legacy",
        bias_scale=1.0,
    )

    bias, logits = module(**relation_inputs())

    expected = torch.sigmoid(logits[0, 0, 1])
    assert bias[0, 0, 1, 2].item() == pytest.approx(expected.item())
    assert torch.all(bias >= 0)
