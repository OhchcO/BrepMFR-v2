import copy

import pytest
import torch
import torch.nn as nn
from fairseq.modules import LayerDropModuleList

from models.modules.brep_encoder import BrepEncoder
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

    assert signed_bias[0, 0, 1, 2].item() == pytest.approx(-1.0)
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


def test_late_gated_relation_bias_is_independent_of_progress():
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

    zero_progress_bias, _ = module(**inputs, progress=0.0)
    full_bias, _ = module(**inputs, progress=1.0)

    assert torch.equal(zero_progress_bias, full_bias)
    assert zero_progress_bias[0, 0, 1, 2].item() == pytest.approx(1.0)
    assert full_bias[0, 0, 1, 2].item() == pytest.approx(1.0)
    with pytest.raises(ValueError, match="progress"):
        module(**inputs, progress=1.1)


def test_late_gated_small_signal_residual_has_one_progress_factor():
    module = InstanceRelationBias(
        embedding_dim=4,
        num_heads=1,
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

    ramp_bias, _ = module(**inputs, progress=0.4)
    full_bias, _ = module(**inputs, progress=1.0)
    ramp_effect = (
        ramp_bias[0, 0, 1, 2] * module.residual_gate(0, progress=0.4)
    )
    full_effect = full_bias[0, 0, 1, 2] * module.residual_gate(0)

    assert ramp_effect.item() == pytest.approx(0.4 * full_effect.item())


def test_dynamic_affinity_logits_update_shared_features_and_affinity_head():
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
    inputs = relation_inputs()
    inputs["node_features"] = (
        inputs["node_features"].clone().requires_grad_(True)
    )

    _, logits = module(**inputs, progress=1.0)
    logits.sum().backward()

    assert inputs["node_features"].grad is not None
    assert torch.count_nonzero(inputs["node_features"].grad).item() > 0
    affinity_parameters = [
        parameter
        for name, parameter in module.named_parameters()
        if name != "gate_logits"
    ]
    assert all(parameter.grad is not None for parameter in affinity_parameters)
    assert any(
        torch.count_nonzero(parameter.grad).item() > 0
        for parameter in affinity_parameters
    )


def test_dynamic_affinity_semantic_bias_does_not_update_encoder_or_affinity_head():
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
    inputs = relation_inputs()
    inputs["node_features"] = (
        inputs["node_features"].clone().requires_grad_(True)
    )

    bias, _ = module(**inputs, progress=1.0)

    assert not bias.requires_grad
    assert inputs["node_features"].grad is None
    affinity_parameters = [
        parameter
        for name, parameter in module.named_parameters()
        if name != "gate_logits"
    ]
    assert module.gate_logits.grad is None
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
    inputs = relation_inputs()
    inputs["node_features"] = (
        inputs["node_features"].clone().requires_grad_(True)
    )

    bias, _ = module(**inputs, progress=1.0)
    bias.sum().backward()

    assert inputs["node_features"].grad is not None
    assert torch.count_nonzero(inputs["node_features"].grad).item() > 0
    assert any(
        parameter.grad is not None
        for name, parameter in module.named_parameters()
        if name != "gate_logits"
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


def test_late_gated_residual_gates_initialize_to_exact_zero():
    module = InstanceRelationBias(
        embedding_dim=4,
        num_heads=1,
        mode="oracle",
        strategy="late_gated",
        gate_init=0.0,
        max_gate_scale=0.75,
        num_residual_gates=3,
    )

    assert torch.equal(module.residual_gates(), torch.zeros(3))
    assert module.gate_scale.item() == pytest.approx(0.0)


def test_late_gated_residual_gates_are_bounded_at_saturation():
    module = InstanceRelationBias(
        embedding_dim=4,
        num_heads=1,
        mode="oracle",
        strategy="late_gated",
        gate_init=0.0,
        max_gate_scale=0.75,
        num_residual_gates=2,
    )
    with torch.no_grad():
        module.gate_logits.copy_(torch.tensor([30.0, -30.0]))

    gates = module.residual_gates()

    assert gates[0].item() == pytest.approx(0.75)
    assert gates[1].item() == pytest.approx(-0.75)
    assert torch.all(gates.abs() <= 0.75)


def test_late_gated_residual_gates_scale_with_progress():
    module = InstanceRelationBias(
        embedding_dim=4,
        num_heads=1,
        mode="oracle",
        strategy="late_gated",
        gate_init=0.5,
        max_gate_scale=1.0,
        num_residual_gates=2,
    )

    assert torch.allclose(module.residual_gates(progress=0.4), torch.full((2,), 0.2))
    assert module.residual_gate(1, progress=0.4).item() == pytest.approx(0.2)
    with pytest.raises(ValueError, match="progress"):
        module.residual_gates(progress=1.1)


def test_zero_initialized_residual_gate_has_nonzero_raw_gradient():
    module = InstanceRelationBias(
        embedding_dim=4,
        num_heads=1,
        mode="oracle",
        strategy="late_gated",
        gate_init=0.0,
        max_gate_scale=0.75,
    )

    module.residual_gate(0).backward()

    assert module.gate_logits.grad.item() == pytest.approx(0.75)


def test_late_gated_residual_gates_are_independent_per_layer():
    module = InstanceRelationBias(
        embedding_dim=4,
        num_heads=1,
        mode="oracle",
        strategy="late_gated",
        gate_init=0.0,
        max_gate_scale=1.0,
        num_residual_gates=3,
    )
    with torch.no_grad():
        module.gate_logits.copy_(torch.tensor([-0.5, 0.0, 0.5]))

    gates = module.residual_gates()

    assert gates[0].item() == pytest.approx(-0.462117, abs=1e-6)
    assert gates[1].item() == pytest.approx(0.0)
    assert gates[2].item() == pytest.approx(0.462117, abs=1e-6)


def test_residual_gate_rejects_invalid_layer_offset():
    module = InstanceRelationBias(
        embedding_dim=4,
        num_heads=1,
        mode="oracle",
        strategy="late_gated",
        gate_init=0.0,
        num_residual_gates=2,
    )

    with pytest.raises(IndexError, match="layer offset"):
        module.residual_gate(2)


class FixedGraphNodeFeature(nn.Module):
    def __init__(self, embedding_dim=8):
        super().__init__()
        self.features = nn.Parameter(
            torch.tensor(
                [
                    [
                        [0.0] * embedding_dim,
                        [1.0] + [0.0] * (embedding_dim - 1),
                        [0.0, 1.0] + [0.0] * (embedding_dim - 2),
                    ]
                ]
            )
        )

    def forward(self, *args, **kwargs):
        return self.features, self.features[:, 1:, :]


class FixedGraphAttnBias(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("bias", torch.full((1, 2, 3, 3), 3.0))

    def forward(self, *args):
        return self.bias


class RecordingGraphLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(self, node_features, **kwargs):
        self.calls.append(kwargs)
        return node_features, None


class AddingGraphLayer(RecordingGraphLayer):
    def __init__(self, increment):
        super().__init__()
        self.increment = float(increment)

    def forward(self, node_features, **kwargs):
        self.calls.append(kwargs)
        return node_features + self.increment, None


class ControlledUniformScores:
    def __init__(self, scores):
        self.scores = scores

    def uniform_(self):
        return torch.tensor(self.scores)


def encoder_batch():
    return {
        "node_data": torch.zeros(1, 2, 1),
        "face_area": torch.zeros(1, 2),
        "face_type": torch.zeros(1, 2, dtype=torch.long),
        "face_loop": torch.zeros(1, 2, dtype=torch.long),
        "in_degree": torch.zeros(1, 2, dtype=torch.long),
        "padding_mask": torch.zeros(1, 2, dtype=torch.bool),
        "attn_bias": None,
        "spatial_pos": torch.zeros(1, 2, 2, dtype=torch.long),
        "d2_distance": torch.zeros(1, 2, 2, 64),
        "angle_distance": torch.zeros(1, 2, 2, 64),
        "edge_data": None,
        "edge_type": None,
        "edge_len": None,
        "edge_ang": None,
        "edge_conv": None,
        "edge_path": None,
        "edge_padding_mask": None,
        "graph": None,
    }


def make_routing_encoder(
    *,
    num_encoder_layers=4,
    instance_bias_last_layers=2,
    mode="dynamic_affinity",
    bias_scale=1.0,
    attention_dropout=0.0,
    record_layers=True,
):
    model = BrepEncoder(
        num_degree=4,
        num_spatial=4,
        num_edge_dis=4,
        edge_type="multi_hop",
        multi_hop_max_dist=1,
        num_encoder_layers=num_encoder_layers,
        embedding_dim=8,
        ffn_embedding_dim=8,
        num_attention_heads=2,
        dropout=0.0,
        attention_dropout=attention_dropout,
        activation_dropout=0.0,
        experiment_mode=mode,
        affinity_dim=4,
        instance_bias_strategy="late_gated",
        instance_bias_last_layers=instance_bias_last_layers,
        instance_bias_gate_init=0.2,
        instance_bias_max_gate_scale=1.0,
        instance_bias_scale=bias_scale,
    )
    model.graph_node_feature = FixedGraphNodeFeature()
    model.graph_attn_bias = FixedGraphAttnBias()
    if record_layers:
        model.layers = nn.ModuleList(
            [RecordingGraphLayer() for _ in range(num_encoder_layers)]
        )
    return model


def test_late_gated_encoder_routes_relation_only_through_final_layers():
    """Removing residual arguments or adding bias before a late layer breaks this."""
    torch.manual_seed(12)
    model = make_routing_encoder()
    model.instance_relation_bias.gate_logits = nn.Parameter(
        torch.atanh(torch.tensor([0.2, 0.6]))
    )
    predicted_biases = []
    model.instance_relation_bias.register_forward_hook(
        lambda module, args, output: predicted_biases.append(output[0])
    )

    model(encoder_batch(), instance_bias_progress=0.5)

    layers = list(model.layers)
    assert len(predicted_biases) == 1
    predicted_bias = predicted_biases[0]
    graph_bias = model.graph_attn_bias.bias
    assert all(
        layer.calls[0]["self_attn_bias"] is graph_bias for layer in layers
    )
    assert layers[0].calls[0].get("instance_attn_bias") is None
    assert layers[1].calls[0].get("instance_attn_bias") is None
    assert layers[2].calls[0]["instance_attn_bias"] is predicted_bias
    assert layers[3].calls[0]["instance_attn_bias"] is predicted_bias
    assert torch.allclose(
        layers[2].calls[0]["instance_attn_gate"],
        torch.tensor(0.1),
    )
    assert torch.allclose(
        layers[3].calls[0]["instance_attn_gate"],
        torch.tensor(0.3),
    )


def test_baseline_encoder_preserves_ordinary_layer_calls_without_residual_gates():
    model = make_routing_encoder(mode="baseline")

    model(encoder_batch())

    assert model.instance_relation_bias.residual_gates() is None
    expected_keys = {
        "self_attn_padding_mask",
        "self_attn_mask",
        "self_attn_bias",
    }
    assert all(
        set(layer.calls[0]) == expected_keys for layer in model.layers
    )


def test_late_gated_encoder_uses_actual_layer_count_when_requested_count_is_larger():
    model = make_routing_encoder(
        num_encoder_layers=2,
        instance_bias_last_layers=3,
    )
    predicted_biases = []
    model.instance_relation_bias.register_forward_hook(
        lambda module, args, output: predicted_biases.append(output[0])
    )

    model(encoder_batch())

    assert len(predicted_biases) == 1
    assert all(
        layer.calls[0]["instance_attn_bias"] is predicted_biases[0]
        for layer in model.layers
    )
    assert torch.equal(
        model.layers[0].calls[0]["instance_attn_gate"],
        model.instance_relation_bias.residual_gate(0),
    )
    assert torch.equal(
        model.layers[1].calls[0]["instance_attn_gate"],
        model.instance_relation_bias.residual_gate(1),
    )


def test_zero_scaled_late_gated_encoder_still_returns_affinity_logits():
    model = make_routing_encoder(bias_scale=0.0)

    _, _, affinity_logits = model(
        encoder_batch(),
        return_instance_logits=True,
    )

    assert affinity_logits is not None
    assert affinity_logits.requires_grad
    assert all(
        layer.calls[0].get("instance_attn_bias") is None
        for layer in list(model.layers)[-2:]
    )


def test_zero_scaled_late_gated_encoder_matches_ordinary_training_attention():
    """Passing a zero bias through a nonzero gate must not add a second dropout path."""
    torch.manual_seed(31)
    affinity_only = make_routing_encoder(
        num_encoder_layers=1,
        instance_bias_last_layers=1,
        bias_scale=0.0,
        attention_dropout=0.5,
        record_layers=False,
    ).train()
    ordinary = copy.deepcopy(affinity_only)
    ordinary.instance_relation_bias.mode = "baseline"

    torch.manual_seed(41)
    _, affinity_only_representation = affinity_only(encoder_batch())
    torch.manual_seed(41)
    _, ordinary_representation = ordinary(encoder_batch())

    assert torch.equal(affinity_only_representation, ordinary_representation)


def test_layerdrop_does_not_promote_earlier_physical_layers_or_shift_gate_offsets(
    monkeypatch,
):
    model = make_routing_encoder()
    physical_layers = [RecordingGraphLayer() for _ in range(4)]
    model.layers = LayerDropModuleList(p=0.5, modules=physical_layers)
    model.train()
    monkeypatch.setattr(
        torch,
        "empty",
        lambda length: ControlledUniformScores([0.9, 0.9, 0.1, 0.9]),
    )

    model(encoder_batch(), instance_bias_progress=0.5)

    assert len(physical_layers[0].calls) == 1
    assert len(physical_layers[1].calls) == 1
    assert len(physical_layers[2].calls) == 0
    assert len(physical_layers[3].calls) == 1
    assert physical_layers[1].calls[0].get("instance_attn_bias") is None
    assert torch.equal(
        physical_layers[3].calls[0]["instance_attn_gate"],
        model.instance_relation_bias.residual_gate(1, progress=0.5),
    )


def test_dropped_prediction_boundary_uses_its_physical_input_for_logits(
    monkeypatch,
):
    model = make_routing_encoder()
    physical_layers = [
        AddingGraphLayer(1.0),
        AddingGraphLayer(10.0),
        AddingGraphLayer(100.0),
        AddingGraphLayer(1000.0),
    ]
    model.layers = LayerDropModuleList(p=0.5, modules=physical_layers)
    model.train()
    monkeypatch.setattr(
        torch,
        "empty",
        lambda length: ControlledUniformScores([0.9, 0.9, 0.1, 0.9]),
    )
    prediction_inputs = []
    relation_forward = model.instance_relation_bias.forward

    def record_prediction_input(*args, **kwargs):
        prediction_inputs.append(
            kwargs["node_features"].detach().clone()
        )
        return relation_forward(*args, **kwargs)

    model.instance_relation_bias.forward = record_prediction_input

    _, _, logits = model(
        encoder_batch(),
        return_instance_logits=True,
    )

    expected_boundary_input = model.graph_node_feature.features.detach() + 11.0
    assert torch.equal(prediction_inputs[0], expected_boundary_input)
    assert logits is not None
    assert len(physical_layers[2].calls) == 0


def test_all_layerdrop_layers_dropped_still_returns_dynamic_affinity_logits():
    model = make_routing_encoder()
    physical_layers = [RecordingGraphLayer() for _ in range(4)]
    model.layers = LayerDropModuleList(p=1.0, modules=physical_layers)
    model.train()

    _, _, logits = model(
        encoder_batch(),
        return_instance_logits=True,
    )

    assert logits is not None
    assert logits.requires_grad
    assert all(not layer.calls for layer in physical_layers)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"num_residual_gates": 0}, "num_residual_gates"),
        ({"gate_init": 1.0, "max_gate_scale": 1.0}, "gate_init"),
        ({"gate_init": -1.0, "max_gate_scale": 1.0}, "gate_init"),
    ],
)
def test_residual_gate_constructor_rejects_invalid_values(kwargs, message):
    with pytest.raises(ValueError, match=message):
        InstanceRelationBias(
            embedding_dim=4,
            num_heads=1,
            mode="oracle",
            strategy="late_gated",
            **kwargs,
        )


@pytest.mark.parametrize(
    "name, value",
    [
        ("gate_init", float("nan")),
        ("gate_init", float("inf")),
        ("gate_init", float("-inf")),
        ("max_gate_scale", float("nan")),
        ("max_gate_scale", float("inf")),
        ("max_gate_scale", float("-inf")),
    ],
)
def test_residual_gate_constructor_rejects_non_finite_values(name, value):
    with pytest.raises(ValueError, match="{} must be finite".format(name)):
        InstanceRelationBias(
            embedding_dim=4,
            num_heads=1,
            mode="oracle",
            strategy="late_gated",
            **{name: value},
        )


@pytest.mark.parametrize(
    "mode, strategy",
    [
        ("baseline", "late_gated"),
        ("oracle", "legacy"),
    ],
)
def test_residual_gate_methods_handle_modes_without_gate_parameters(
    mode, strategy
):
    module = InstanceRelationBias(
        embedding_dim=4,
        num_heads=1,
        mode=mode,
        strategy=strategy,
    )

    assert module.residual_gates() is None
    with pytest.raises(IndexError, match="residual gate layer offset"):
        module.residual_gate(0)
