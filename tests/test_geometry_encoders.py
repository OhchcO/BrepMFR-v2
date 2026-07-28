from types import SimpleNamespace

import pytest
import torch
from torch import nn

from models.brepseg_model import BrepSeg
from models.modules.brep_encoder import BrepEncoder
from models.modules.cylindrical_encoder import CylindricalPositionEncoder
from models.modules.layers.brep_encoder_layer import GraphNodeFeature
from models.modules.pointcloud_encoder import PointCloudFaceEncoder


def valid_face_point_batch(face_count=3):
    points = torch.randn(face_count, 96, 6)
    mask = torch.zeros(face_count, 96, dtype=torch.bool)
    mask[:, :32] = True
    points[:, :32, 3:] = torch.nn.functional.normalize(
        points[:, :32, 3:], dim=-1
    )
    return points, mask


def test_point_encoder_ignores_padded_values():
    encoder = PointCloudFaceEncoder(hidden_dim=128).eval()
    points = torch.randn(2, 96, 6)
    mask = torch.zeros(2, 96, dtype=torch.bool)
    mask[:, :32] = True
    changed = points.clone()
    changed[:, 32:] = 1e6

    with torch.no_grad():
        assert torch.allclose(encoder(points, mask), encoder(changed, mask))


def test_encoders_return_finite_hidden_vectors_and_point_gradients():
    points, mask = valid_face_point_batch(face_count=3)
    points.requires_grad_()
    encoded_points = PointCloudFaceEncoder(128)(points, mask)
    encoded_cylindrical = CylindricalPositionEncoder(128)(torch.randn(3, 4))

    assert encoded_points.shape == (3, 128)
    assert torch.isfinite(encoded_points).all()
    assert encoded_cylindrical.shape == (3, 128)
    assert torch.isfinite(encoded_cylindrical).all()
    encoded_points.sum().backward()
    assert points.grad is not None
    assert torch.isfinite(points.grad).all()


def test_encoders_reject_malformed_inputs():
    points, mask = valid_face_point_batch(face_count=1)
    with pytest.raises(ValueError, match="face_points"):
        PointCloudFaceEncoder(16)(points[..., :5], mask)
    with pytest.raises(ValueError, match="zero valid"):
        PointCloudFaceEncoder(16)(points, torch.zeros_like(mask))
    with pytest.raises(ValueError, match="finite"):
        CylindricalPositionEncoder(16)(torch.tensor([[float("nan"), 0.0, 0.0, 0.0]]))


def make_graph_node_feature(
    use_face_pointcloud=False,
    use_cylindrical_pos=False,
    pointcloud_scale=1.0,
    cylindrical_scale=1.0,
    use_face_uv=True,
):
    return GraphNodeFeature(
        num_heads=8,
        num_degree=128,
        hidden_dim=128,
        n_layers=1,
        use_face_uv=use_face_uv,
        use_face_pointcloud=use_face_pointcloud,
        use_cylindrical_pos=use_cylindrical_pos,
        pointcloud_scale=pointcloud_scale,
        cylindrical_scale=cylindrical_scale,
    ).eval()


def graph_node_inputs():
    points, point_mask = valid_face_point_batch(face_count=2)
    return {
        "x": torch.randn(2, 4, 4, 7),
        "face_area": torch.rand(2),
        "face_type": torch.tensor([1, 2]),
        "face_loop": torch.tensor([1, 1]),
        "face_degree": torch.tensor([1, 2]),
        "padding_mask": torch.tensor([[False, False]]),
        "face_points": points,
        "face_point_mask": point_mask,
        "cylindrical_pos": torch.randn(2, 4),
    }


def make_brep_encoder(**geometry_options):
    return BrepEncoder(
        num_degree=128,
        num_spatial=64,
        num_edge_dis=64,
        edge_type="multi_hop",
        multi_hop_max_dist=16,
        num_encoder_layers=1,
        embedding_dim=128,
        ffn_embedding_dim=128,
        num_attention_heads=8,
        dropout=0.0,
        attention_dropout=0.0,
        activation_dropout=0.0,
        layerdrop=0.0,
        apply_params_init=True,
        experiment_mode="dynamic",
        **geometry_options,
    ).eval()


def test_all_geometry_switches_off_preserve_original_parameter_names():
    baseline = make_graph_node_feature()
    baseline_encoder = make_brep_encoder()

    assert baseline.pointcloud_encoder is None
    assert baseline.cylindrical_encoder is None
    assert not any(
        name.startswith("pointcloud_encoder")
        for name, _ in baseline.named_parameters()
    )
    assert not any(
        name.startswith("cylindrical_encoder")
        for name, _ in baseline.named_parameters()
    )
    assert not any(
        "pointcloud_encoder" in name
        or "cylindrical_encoder" in name
        for name, _ in baseline_encoder.named_parameters()
    )


def test_disabling_uv_does_not_construct_or_consume_surface_encoder():
    feature = make_graph_node_feature(use_face_uv=False)
    inputs = graph_node_inputs()
    changed_inputs = dict(inputs)
    changed_inputs["x"] = torch.full_like(inputs["x"], float("nan"))

    assert feature.surf_encoder is None
    with torch.no_grad():
        original = feature(
            inputs["x"],
            inputs["face_area"],
            inputs["face_type"],
            inputs["face_loop"],
            inputs["face_degree"],
            inputs["padding_mask"],
        )
        changed = feature(
            changed_inputs["x"],
            inputs["face_area"],
            inputs["face_type"],
            inputs["face_loop"],
            inputs["face_degree"],
            inputs["padding_mask"],
        )

    for original_output, changed_output in zip(original, changed):
        assert original_output.shape[-1] == 128
        assert torch.equal(original_output, changed_output)


def test_optional_construction_preserves_downstream_rng_and_shared_weights():
    torch.manual_seed(2026)
    make_graph_node_feature()
    baseline_next_random = torch.rand(8)
    torch.manual_seed(2026)
    make_graph_node_feature(True, True)
    enhanced_next_random = torch.rand(8)
    assert torch.equal(baseline_next_random, enhanced_next_random)

    torch.manual_seed(2026)
    baseline = make_brep_encoder()
    torch.manual_seed(2026)
    enhanced = make_brep_encoder(
        use_face_pointcloud=True,
        use_cylindrical_pos=True,
    )
    baseline_state = baseline.state_dict()
    enhanced_state = enhanced.state_dict()
    for name, value in baseline_state.items():
        assert torch.equal(value, enhanced_state[name]), name


def test_zero_geometry_scales_match_base_node_feature():
    torch.manual_seed(2026)
    baseline = make_graph_node_feature()
    torch.manual_seed(2026)
    enhanced = make_graph_node_feature(
        True,
        True,
        pointcloud_scale=0.0,
        cylindrical_scale=0.0,
    )
    enhanced.load_state_dict(baseline.state_dict(), strict=False)
    inputs = graph_node_inputs()

    with torch.no_grad():
        baseline_outputs = baseline(
            inputs["x"],
            inputs["face_area"],
            inputs["face_type"],
            inputs["face_loop"],
            inputs["face_degree"],
            inputs["padding_mask"],
        )
        enhanced_outputs = enhanced(**inputs)

    for baseline_output, enhanced_output in zip(
        baseline_outputs, enhanced_outputs
    ):
        assert torch.allclose(
            baseline_output,
            enhanced_output,
            atol=0.0,
            rtol=0.0,
        )


class RecordingGraphAttnBias(nn.Module):
    def __init__(self, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.node_features = None

    def forward(self, *args):
        self.node_features = args[-1].detach().clone()
        padding_mask = args[1]
        batch_size, max_nodes, _ = padding_mask.shape
        return self.node_features.new_zeros(
            (batch_size, self.num_heads, max_nodes + 1, max_nodes + 1)
        )


class RecordingInstanceRelationBias(nn.Module):
    def __init__(
        self,
        num_heads,
        mode="dynamic",
        strategy="legacy",
        fill_value=0.0,
        bias_scale=1.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.mode = mode
        self.strategy = strategy
        self.fill_value = fill_value
        self.bias_scale = bias_scale
        self.node_features = None
        self.instance = "not called"
        self.instance_available = "not called"
        self.progress = None
        self.call_count = 0
        self.residual_gate_calls = []

    def forward(
        self,
        node_features,
        padding_mask,
        spatial_pos,
        d2_distance,
        angle_distance,
        instance=None,
        instance_available=None,
        progress=1.0,
    ):
        self.node_features = node_features.detach().clone()
        self.instance = instance
        self.instance_available = instance_available
        self.progress = progress
        self.call_count += 1
        batch_size, max_nodes = padding_mask.shape
        bias = node_features.new_full(
            (batch_size, self.num_heads, max_nodes + 1, max_nodes + 1),
            self.fill_value,
        )
        return bias, None

    def residual_gate(self, layer_offset, progress=1.0):
        self.residual_gate_calls.append((layer_offset, progress))
        return torch.tensor(0.25)


class RecordingGraphLayer(nn.Module):
    def __init__(self, increment=0.0):
        super().__init__()
        self.node_features = None
        self.attention_bias = None
        self.instance_attention_bias = None
        self.instance_attention_gate = None
        self.increment = increment

    def forward(self, node_features, **kwargs):
        self.node_features = node_features.detach().clone()
        self.attention_bias = kwargs["self_attn_bias"].detach().clone()
        instance_bias = kwargs.get("instance_attn_bias")
        self.instance_attention_bias = (
            None if instance_bias is None else instance_bias.detach().clone()
        )
        self.instance_attention_gate = kwargs.get("instance_attn_gate")
        return node_features + self.increment, None


def geometry_encoder_batch():
    inputs = graph_node_inputs()
    batch = {
        "node_data": inputs["x"],
        "face_area": inputs["face_area"],
        "face_type": inputs["face_type"],
        "face_loop": inputs["face_loop"],
        "in_degree": inputs["face_degree"],
        "padding_mask": inputs["padding_mask"],
        "face_points": inputs["face_points"],
        "face_point_mask": inputs["face_point_mask"],
        "cylindrical_pos": inputs["cylindrical_pos"],
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
    return batch


def test_enhanced_feature_reaches_graph_dynamic_and_transformer_paths():
    model = make_brep_encoder(
        use_face_pointcloud=True,
        use_cylindrical_pos=True,
    )
    graph_bias = RecordingGraphAttnBias(num_heads=8)
    relation_bias = RecordingInstanceRelationBias(num_heads=8)
    graph_layer = RecordingGraphLayer()
    model.graph_attn_bias = graph_bias
    model.instance_relation_bias = relation_bias
    model.layers = nn.ModuleList([graph_layer])

    with torch.no_grad():
        model(
            geometry_encoder_batch(),
            last_state_only=True,
            return_instance_logits=True,
        )

    relation_faces = relation_bias.node_features[:, 1:, :]
    assert torch.equal(graph_bias.node_features, relation_faces[0])
    assert torch.equal(
        relation_bias.node_features,
        graph_layer.node_features.permute(1, 0, 2),
    )
    assert relation_bias.instance is None
    assert relation_bias.instance_available is None


def test_oracle_relation_path_still_receives_instance_labels():
    model = make_brep_encoder(
        use_face_pointcloud=True,
        use_cylindrical_pos=True,
    )
    graph_bias = RecordingGraphAttnBias(num_heads=8)
    relation_bias = RecordingInstanceRelationBias(
        num_heads=8,
        mode="oracle",
    )
    model.graph_attn_bias = graph_bias
    model.instance_relation_bias = relation_bias
    model.layers = nn.ModuleList([RecordingGraphLayer()])
    batch = geometry_encoder_batch()
    batch["instance"] = torch.ones(1, 2, 2)
    batch["instance_available"] = torch.tensor([True])

    with torch.no_grad():
        model(batch, last_state_only=True)

    assert relation_bias.instance is batch["instance"]
    assert relation_bias.instance_available is batch["instance_available"]


def test_late_gated_relation_is_predicted_midway_and_only_reaches_last_layers():
    model = BrepEncoder(
        num_degree=128,
        num_spatial=64,
        num_edge_dis=64,
        edge_type="multi_hop",
        multi_hop_max_dist=16,
        num_encoder_layers=4,
        embedding_dim=128,
        ffn_embedding_dim=128,
        num_attention_heads=8,
        dropout=0.0,
        attention_dropout=0.0,
        activation_dropout=0.0,
        layerdrop=0.0,
        apply_params_init=True,
        experiment_mode="dynamic_affinity",
        instance_bias_strategy="late_gated",
        instance_relation_mapping="positive_only",
        instance_bias_last_layers=2,
    ).eval()
    model.graph_attn_bias = RecordingGraphAttnBias(num_heads=8)
    relation_bias = RecordingInstanceRelationBias(
        num_heads=8,
        mode="dynamic_affinity",
        strategy="late_gated",
        fill_value=2.0,
    )
    layers = [RecordingGraphLayer(increment=1.0) for _ in range(4)]
    model.instance_relation_bias = relation_bias
    model.layers = nn.ModuleList(layers)

    with torch.no_grad():
        model(
            geometry_encoder_batch(),
            last_state_only=True,
            return_instance_logits=True,
            instance_bias_progress=0.75,
        )

    assert relation_bias.call_count == 1
    assert relation_bias.progress == pytest.approx(0.75)
    assert torch.equal(
        relation_bias.node_features,
        layers[2].node_features.permute(1, 0, 2),
    )
    assert torch.count_nonzero(layers[0].attention_bias).item() == 0
    assert torch.count_nonzero(layers[1].attention_bias).item() == 0
    assert torch.count_nonzero(layers[2].attention_bias).item() == 0
    assert torch.count_nonzero(layers[3].attention_bias).item() == 0
    assert layers[0].instance_attention_bias is None
    assert layers[1].instance_attention_bias is None
    assert torch.all(layers[2].instance_attention_bias == 2.0)
    assert torch.all(layers[3].instance_attention_bias == 2.0)
    assert relation_bias.residual_gate_calls == [(0, 0.75), (1, 0.75)]


def test_brepseg_saves_and_threads_geometry_hyperparameters():
    args = SimpleNamespace(
        num_classes=24,
        n_layers_encode=1,
        dim_node=128,
        d_model=128,
        n_heads=8,
        dropout=0.0,
        attention_dropout=0.0,
        act_dropout=0.0,
        use_face_uv=False,
        use_face_pointcloud=True,
        use_cylindrical_pos=True,
        pointcloud_scale=0.25,
        cylindrical_scale=0.5,
        experiment_mode="dynamic_affinity",
        instance_bias_strategy="late_gated",
        instance_relation_mapping="positive_only",
        instance_bias_last_layers=2,
        instance_bias_confidence_power=2.0,
        instance_bias_gate_init=0.1,
        instance_bias_max_gate_scale=0.5,
        instance_bias_delay_epochs=5,
        instance_bias_ramp_epochs=7,
    )
    model = BrepSeg(args)

    assert model.hparams["use_face_uv"] is False
    assert model.hparams["use_face_pointcloud"] is True
    assert model.hparams["use_cylindrical_pos"] is True
    assert model.hparams["pointcloud_scale"] == 0.25
    assert model.hparams["cylindrical_scale"] == 0.5
    assert model.brep_encoder.graph_node_feature.pointcloud_scale == 0.25
    assert model.brep_encoder.graph_node_feature.cylindrical_scale == 0.5
    assert model.brep_encoder.graph_node_feature.surf_encoder is None
    assert model.hparams["instance_bias_strategy"] == "late_gated"
    assert model.hparams["instance_relation_mapping"] == "positive_only"
    assert model.hparams["instance_bias_last_layers"] == 2
    assert model.hparams["instance_bias_confidence_power"] == 2.0
    assert model.hparams["instance_bias_gate_init"] == 0.1
    assert model.hparams["instance_bias_max_gate_scale"] == 0.5
    assert model.hparams["instance_bias_delay_epochs"] == 5
    assert model.hparams["instance_bias_ramp_epochs"] == 7
    assert model.brep_encoder.instance_bias_last_layers == 2
    assert model.brep_encoder.instance_relation_bias.strategy == "late_gated"
    assert (
        model.brep_encoder.instance_relation_bias.relation_mapping
        == "positive_only"
    )
    assert (
        model.brep_encoder.instance_relation_bias.confidence_power
        == 2.0
    )


def test_predict_with_affinity_returns_semantics_and_relation_logits():
    args = SimpleNamespace(
        num_classes=3,
        n_layers_encode=1,
        dim_node=128,
        d_model=128,
        n_heads=8,
        dropout=0.0,
        attention_dropout=0.0,
        act_dropout=0.0,
        experiment_mode="dynamic",
    )
    model = BrepSeg(args).eval()
    expected_logits = torch.tensor(
        [[[0.0, 2.0], [2.0, 0.0]]], dtype=torch.float32
    )

    class FakeEncoder(nn.Module):
        def forward(
            self,
            batch,
            last_state_only=True,
            return_instance_logits=False,
        ):
            state = torch.randn(3, 1, 128)
            graph = torch.randn(1, 128)
            if return_instance_logits:
                return [state], graph, expected_logits
            return [state], graph

    model.brep_encoder = FakeEncoder()
    batch = {"padding_mask": torch.tensor([[False, False]])}

    with torch.no_grad():
        probabilities, logits = model.predict_with_affinity(batch)

    assert probabilities.shape == (2, 3)
    assert torch.allclose(probabilities.sum(dim=1), torch.ones(2))
    assert logits is expected_logits
