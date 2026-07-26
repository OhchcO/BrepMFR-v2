"""Instance-relation attention bias used by the four ablation modes."""

import torch
from torch import nn
import torch.nn.functional as F


EXPERIMENT_MODES = ("baseline", "oracle", "dynamic", "dynamic_affinity")


class InstanceRelationBias(nn.Module):
    """Build a positive same-instance attention prior.

    Oracle mode uses the ground-truth affinity matrix and is an upper bound.
    Dynamic modes predict a symmetric affinity from node content and the
    existing SPE-PCAAG pair descriptors, so inference needs no instance label.
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
    ):
        super().__init__()
        if mode not in EXPERIMENT_MODES:
            raise ValueError("Unsupported experiment mode: {}".format(mode))
        if affinity_dim < 1:
            raise ValueError("affinity_dim must be positive")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if bias_scale < 0:
            raise ValueError("bias_scale must be non-negative")

        self.mode = mode
        self.num_heads = num_heads
        self.temperature = float(temperature)
        self.bias_scale = float(bias_scale)

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
    ):
        """Return ``(bias, logits)``.

        ``bias`` has shape [B, H, N+1, N+1]. ``logits`` is [B, N, N]
        only for dynamic modes and is used by the optional affinity loss.
        """
        batch_size, max_nodes = padding_mask.shape
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
            relation = torch.sigmoid(logits)

        relation = relation * pair_mask.to(relation.dtype)
        face_bias = self.bias_scale * relation
        bias = node_features.new_zeros(
            (batch_size, self.num_heads, max_nodes + 1, max_nodes + 1)
        )
        bias[:, :, 1:, 1:] = face_bias.unsqueeze(1)
        return bias, logits
