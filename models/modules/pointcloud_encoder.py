"""Masked point-cloud encoder for per-face geometry features."""

import torch
from torch import Tensor, nn


class PointCloudFaceEncoder(nn.Module):
    """Encode valid 10D surface samples into one vector per face."""

    def __init__(self, hidden_dim: int, eps: float = 1e-8) -> None:
        super().__init__()
        if not isinstance(hidden_dim, int) or isinstance(hidden_dim, bool) or hidden_dim <= 0:
            raise ValueError("hidden_dim must be a positive integer")
        if eps <= 0:
            raise ValueError("eps must be positive")

        self.eps = eps
        self.point_mlp = nn.Sequential(
            nn.Linear(10, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, 256),
            nn.LayerNorm(256),
            nn.GELU(),
        )
        self.output_mlp = nn.Sequential(
            nn.Linear(256, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

    def forward(self, face_points: Tensor, face_point_mask: Tensor) -> Tensor:
        """Return an embedding for every face with at least one valid point."""
        if face_points.ndim != 3 or face_points.shape[-1] != 10:
            raise ValueError("face_points must have shape [F, P, 10]")
        if face_point_mask.ndim != 2 or face_point_mask.shape != face_points.shape[:2]:
            raise ValueError("face_point_mask must have shape [F, P]")
        if face_point_mask.dtype != torch.bool:
            raise ValueError("face_point_mask must have dtype torch.bool")
        if not face_points.is_floating_point():
            raise ValueError("face_points must use a floating-point dtype")
        if not torch.isfinite(face_points).all():
            raise ValueError("face_points must contain only finite values")

        valid_counts = face_point_mask.sum(dim=1)
        if torch.any(valid_counts == 0):
            raise ValueError("face_points contains a face with zero valid points")

        valid = face_point_mask.unsqueeze(-1)
        xyz = face_points[..., :3]
        surface_attributes = face_points[..., 3:]
        valid_counts_float = valid_counts.to(dtype=face_points.dtype).unsqueeze(-1)
        center = (xyz * valid).sum(dim=1) / valid_counts_float
        centered_xyz = xyz - center.unsqueeze(1)
        radii = torch.linalg.vector_norm(centered_xyz, dim=-1)
        max_radius = radii.masked_fill(~face_point_mask, 0).amax(dim=1)
        scale = max_radius.clamp_min(self.eps).unsqueeze(-1).unsqueeze(-1)
        point_features = torch.cat(
            (centered_xyz / scale, surface_attributes), dim=-1
        )

        point_features = self.point_mlp(point_features)
        negative_infinity = torch.finfo(point_features.dtype).min
        point_features = point_features.masked_fill(~valid, negative_infinity)
        pooled = point_features.amax(dim=1)
        return self.output_mlp(pooled)
