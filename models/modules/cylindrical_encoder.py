"""Encoder for normalized cylindrical face positions."""

import torch
from torch import Tensor, nn


class CylindricalPositionEncoder(nn.Module):
    """Project four cylindrical position values into the model hidden space."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        if not isinstance(hidden_dim, int) or isinstance(hidden_dim, bool) or hidden_dim <= 0:
            raise ValueError("hidden_dim must be a positive integer")
        self.mlp = nn.Sequential(
            nn.Linear(4, 64),
            nn.GELU(),
            nn.Linear(64, hidden_dim),
        )

    def forward(self, cylindrical_pos: Tensor) -> Tensor:
        """Return one embedding per face."""
        if cylindrical_pos.ndim != 2 or cylindrical_pos.shape[-1] != 4:
            raise ValueError("cylindrical_pos must have shape [F, 4]")
        if not cylindrical_pos.is_floating_point():
            raise ValueError("cylindrical_pos must use a floating-point dtype")
        if not torch.isfinite(cylindrical_pos).all():
            raise ValueError("cylindrical_pos must contain only finite values")
        return self.mlp(cylindrical_pos)
