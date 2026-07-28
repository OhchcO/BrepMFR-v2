import pytest
import torch
import torch.nn as nn

from models.modules.layers.brep_encoder_layer import GraphEncoderLayer


def make_layer(*, pre_layernorm=False):
    torch.manual_seed(20260728)
    layer = GraphEncoderLayer(
        embedding_dim=8,
        ffn_embedding_dim=16,
        num_attention_heads=2,
        dropout=0.0,
        attention_dropout=0.0,
        activation_dropout=0.0,
        pre_layernorm=pre_layernorm,
    )
    return layer.eval()


def attention_inputs():
    torch.manual_seed(7)
    x = torch.randn(3, 1, 8)
    graph_bias = torch.zeros(1, 2, 3, 3)
    instance_bias = torch.zeros_like(graph_bias)
    instance_bias[:, :, :, 0] = 20.0
    return x, graph_bias, instance_bias


class CountingAttention(nn.Module):
    def __init__(self, attention):
        super().__init__()
        self.attention = attention
        self.calls = 0

    def forward(self, *args, **kwargs):
        self.calls += 1
        return self.attention(*args, **kwargs)


def test_zero_gate_equals_the_ordinary_attention_path_exactly():
    layer = make_layer()
    x, graph_bias, instance_bias = attention_inputs()

    ordinary, _ = layer(x, self_attn_bias=graph_bias)
    residual, _ = layer(
        x,
        self_attn_bias=graph_bias,
        instance_attn_bias=instance_bias,
        instance_attn_gate=torch.tensor(0.0),
    )

    assert torch.equal(residual, ordinary)


def test_gate_one_changes_output_when_instance_bias_changes_attention():
    layer = make_layer()
    x, graph_bias, instance_bias = attention_inputs()

    ordinary, _ = layer(x, self_attn_bias=graph_bias)
    residual, _ = layer(
        x,
        self_attn_bias=graph_bias,
        instance_attn_bias=instance_bias,
        instance_attn_gate=torch.tensor(1.0),
    )

    assert not torch.equal(residual, ordinary)


def test_instance_bias_without_graph_bias_changes_the_base_attention_path():
    layer = make_layer()
    x, _, instance_bias = attention_inputs()

    ordinary, _ = layer(x)
    residual, _ = layer(
        x,
        instance_attn_bias=instance_bias,
        instance_attn_gate=torch.tensor(1.0),
    )

    assert residual.shape == ordinary.shape
    assert not torch.equal(residual, ordinary)


def test_zero_gate_receives_gradient_from_the_biased_attention_difference():
    layer = make_layer()
    x, graph_bias, instance_bias = attention_inputs()
    gate = torch.tensor(0.0, requires_grad=True)

    output, _ = layer(
        x,
        self_attn_bias=graph_bias,
        instance_attn_bias=instance_bias,
        instance_attn_gate=gate,
    )
    output[0, 0, 0].backward()

    assert gate.grad is not None
    assert gate.grad.item() != 0.0


def test_existing_path_calls_self_attention_once():
    layer = make_layer()
    original_attention = layer.self_attn
    layer.self_attn = CountingAttention(original_attention)
    x, graph_bias, _ = attention_inputs()

    layer(x, self_attn_bias=graph_bias)

    assert layer.self_attn.calls == 1
    assert layer.self_attn.attention is original_attention


def test_residual_path_reuses_self_attention_module_twice():
    layer = make_layer()
    original_attention = layer.self_attn
    layer.self_attn = CountingAttention(original_attention)
    x, graph_bias, instance_bias = attention_inputs()

    layer(
        x,
        self_attn_bias=graph_bias,
        instance_attn_bias=instance_bias,
        instance_attn_gate=torch.tensor(0.5),
    )

    assert layer.self_attn.calls == 2
    assert layer.self_attn.attention is original_attention


@pytest.mark.parametrize(
    "kwargs",
    [
        {"instance_attn_bias": torch.zeros(1, 2, 3, 3)},
        {"instance_attn_gate": torch.tensor(0.0)},
    ],
)
def test_partial_instance_attention_configuration_raises_value_error(kwargs):
    layer = make_layer()
    x, graph_bias, _ = attention_inputs()

    with pytest.raises(ValueError, match="instance_attn_bias and instance_attn_gate"):
        layer(x, self_attn_bias=graph_bias, **kwargs)


def test_pre_layernorm_supports_the_residual_attention_branch():
    layer = make_layer(pre_layernorm=True)
    x, graph_bias, instance_bias = attention_inputs()

    ordinary, _ = layer(x, self_attn_bias=graph_bias)
    residual, _ = layer(
        x,
        self_attn_bias=graph_bias,
        instance_attn_bias=instance_bias,
        instance_attn_gate=torch.tensor(1.0),
    )

    assert residual.shape == x.shape
    assert not torch.equal(residual, ordinary)
