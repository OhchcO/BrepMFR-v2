"""Instance-relation attention bias used by the four ablation modes."""

import torch
from torch import nn
import torch.nn.functional as F


EXPERIMENT_MODES = ("baseline", "oracle", "dynamic", "dynamic_affinity")
INSTANCE_BIAS_STRATEGIES = ("legacy", "late_gated")
INSTANCE_RELATION_MAPPINGS = ("signed", "positive_only")


def confidence_weighted_relation(
    logits,
    power=1.0,
    relation_mapping="signed",
):
    """Convert affinity logits to configurable confidence-weighted relations."""
    if power < 0:
        raise ValueError("confidence power must be non-negative")
    if relation_mapping not in INSTANCE_RELATION_MAPPINGS:
        raise ValueError(
            "Unsupported instance relation mapping: {}".format(
                relation_mapping
            )
        )
    centered = 2.0 * torch.sigmoid(logits) - 1.0
    confidence = centered.abs().pow(float(power))
    relation = centered * confidence
    if relation_mapping == "positive_only":
        relation = relation.clamp_min(0.0)
    return relation


class InstanceRelationBias(nn.Module):
    """Build an instance-relation attention prior.

    Oracle mode uses the ground-truth affinity matrix and is an upper bound.
    Dynamic modes predict a symmetric affinity from node content and the
    existing pair descriptors, so inference needs no instance label. The
    late-gated strategy uses configurable confidence-weighted relations.
    """

    def __init__(
        self,
        embedding_dim,
        num_heads,
        mode="baseline",
        affinity_dim=64,
        temperature=0.2,
        bias_scale=1.0,
        num_spatial=64,
        strategy="legacy",
        relation_mapping="signed",
        confidence_power=1.0,
        gate_init=0.05,
        max_gate_scale=1.0,
    ):
        super().__init__()
        if mode not in EXPERIMENT_MODES:
            raise ValueError("Unsupported experiment mode: {}".format(mode))
        if strategy not in INSTANCE_BIAS_STRATEGIES:
            raise ValueError(
                "Unsupported instance bias strategy: {}".format(strategy)
            )
        if relation_mapping not in INSTANCE_RELATION_MAPPINGS:
            raise ValueError(
                "Unsupported instance relation mapping: {}".format(
                    relation_mapping
                )
            )
        if affinity_dim < 1:
            raise ValueError("affinity_dim must be positive")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if bias_scale < 0:
            raise ValueError("bias_scale must be non-negative")
        if confidence_power < 0:
            raise ValueError("confidence_power must be non-negative")
        if max_gate_scale <= 0:
            raise ValueError("max_gate_scale must be positive")
        if not 0 < gate_init < max_gate_scale:
            raise ValueError(
                "gate_init must be between zero and max_gate_scale"
            )

        self.mode = mode
        self.num_heads = num_heads
        self.temperature = float(temperature)
        self.bias_scale = float(bias_scale)
        self.strategy = strategy
        self.relation_mapping = relation_mapping
        self.confidence_power = float(confidence_power)
        self.max_gate_scale = float(max_gate_scale)

        if self.strategy == "late_gated" and self.mode != "baseline":
            gate_fraction = float(gate_init) / self.max_gate_scale
            gate_logit = torch.log(
                torch.tensor(gate_fraction / (1.0 - gate_fraction))
            )
            self.gate_logit = nn.Parameter(gate_logit)

        if self.is_dynamic:
            self.node_projection = nn.Sequential(
                nn.LayerNorm(embedding_dim),
                nn.Linear(embedding_dim, affinity_dim),
                nn.GELU(),
                nn.Linear(affinity_dim, affinity_dim, bias=False),
            )
            self.spatial_projection = nn.Embedding(
                num_spatial, 1, padding_idx=0
            )
            self.d2_projection = nn.Linear(64, 1, bias=False)
            self.angle_projection = nn.Linear(64, 1, bias=False)

    @property
    def is_dynamic(self):
        return self.mode in ("dynamic", "dynamic_affinity")

    @property
    def gate_scale(self):
        if self.strategy != "late_gated" or self.mode == "baseline":
            return None
        return self.max_gate_scale * torch.sigmoid(self.gate_logit)

    def _valid_pair_mask(self, padding_mask):
        valid_node = ~padding_mask
        valid_pair = valid_node.unsqueeze(2) & valid_node.unsqueeze(1)
        diagonal = torch.eye(
            padding_mask.size(1),
            device=padding_mask.device,
            dtype=torch.bool,
        ).unsqueeze(0)
        return valid_pair & ~diagonal

    def _dynamic_logits(
        self,
        node_features,
        padding_mask,
        spatial_pos,
        d2_distance,
        angle_distance,
    ):
        projected = self.node_projection(node_features[:, 1:, :])
        valid_node = (~padding_mask).unsqueeze(-1).to(projected.dtype)
        node_mean = (projected * valid_node).sum(dim=1, keepdim=True)
        node_mean = node_mean / valid_node.sum(dim=1, keepdim=True).clamp_min(1.0)
        projected = (projected - node_mean) * valid_node
        projected = F.normalize(projected, p=2, dim=-1, eps=1e-6)
        content_logits = torch.matmul(
            projected, projected.transpose(1, 2)
        ) / self.temperature

        spatial_logits = self.spatial_projection(spatial_pos).squeeze(-1)
        d2_logits = self.d2_projection(d2_distance).squeeze(-1)
        angle_logits = self.angle_projection(angle_distance).squeeze(-1)
        logits = content_logits + spatial_logits + d2_logits + angle_logits
        return 0.5 * (logits + logits.transpose(1, 2))

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
        """Return ``(bias, logits)``.

        ``bias`` has shape [B, H, N+1, N+1]. ``logits`` is [B, N, N]
        only for dynamic modes and is used by the optional affinity loss.
        """
        batch_size, max_nodes = padding_mask.shape
        if not 0.0 <= float(progress) <= 1.0:
            raise ValueError("instance bias progress must be between 0 and 1")
        pair_mask = self._valid_pair_mask(padding_mask)
        logits = None

        if self.mode == "baseline":
            relation = node_features.new_zeros(
                (batch_size, max_nodes, max_nodes)
            )
        elif self.mode == "oracle":
            if instance is None:
                raise ValueError("Oracle mode requires an instance matrix")
            if tuple(instance.shape) != (batch_size, max_nodes, max_nodes):
                raise ValueError(
                    "Oracle instance shape {} does not match batch ({}, {}, {})".format(
                        tuple(instance.shape),
                        batch_size,
                        max_nodes,
                        max_nodes,
                    )
                )
            if instance_available is not None and not bool(
                instance_available.all().item()
            ):
                raise ValueError(
                    "Oracle mode requires instance annotations for every graph"
                )
            relation = instance.to(node_features.dtype)
        else:
            logits = self._dynamic_logits(
                node_features,
                padding_mask,
                spatial_pos,
                d2_distance,
                angle_distance,
            )
            if self.strategy == "late_gated":
                relation = confidence_weighted_relation(
                    logits,
                    power=self.confidence_power,
                    relation_mapping=self.relation_mapping,
                )
                if self.mode == "dynamic_affinity":
                    relation = relation.detach()
            else:
                relation = torch.sigmoid(logits)

        if self.strategy == "late_gated" and self.mode == "oracle":
            relation = 2.0 * relation - 1.0
            if self.relation_mapping == "positive_only":
                relation = relation.clamp_min(0.0)

        relation = relation * pair_mask.to(relation.dtype)
        if self.strategy == "late_gated" and self.mode != "baseline":
            effective_scale = (
                self.bias_scale * float(progress) * self.gate_scale
            )
        else:
            effective_scale = self.bias_scale
        face_bias = effective_scale * relation
        bias = node_features.new_zeros(
            (batch_size, self.num_heads, max_nodes + 1, max_nodes + 1)
        )
        bias[:, :, 1:, 1:] = face_bias.unsqueeze(1)
        return bias, logits
