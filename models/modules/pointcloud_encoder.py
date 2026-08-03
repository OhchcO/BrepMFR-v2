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


class ParametricSurfaceFaceEncoder(nn.Module):
    """Trim-aware one-level hierarchical encoder for sampled B-Rep faces."""

    def __init__(
        self,
        hidden_dim: int,
        neighbor_metric: str = "intrinsic",
        trim_aware: bool = True,
        neighbor_count: int = 16,
        coarse_count: int = 24,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if neighbor_metric not in ("xyz", "uv", "intrinsic"):
            raise ValueError("neighbor_metric must be xyz, uv, or intrinsic")
        if trim_aware and neighbor_metric != "intrinsic":
            raise ValueError("trim_aware requires the intrinsic neighbor metric")
        if neighbor_count < 1 or coarse_count < 1:
            raise ValueError("neighbor_count and coarse_count must be positive")
        self.neighbor_metric = neighbor_metric
        self.trim_aware = bool(trim_aware)
        self.neighbor_count = int(neighbor_count)
        self.coarse_count = int(coarse_count)
        self.eps = float(eps)
        self.point_mlp = nn.Sequential(
            nn.Linear(13, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, 64),
            nn.LayerNorm(64),
            nn.GELU(),
        )
        self.edge_mlp = nn.Sequential(
            nn.Linear(70, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.GELU(),
        )
        self.local_mlp = nn.Sequential(
            nn.Linear(192, 128),
            nn.LayerNorm(128),
            nn.GELU(),
        )
        self.output_mlp = nn.Sequential(
            nn.Linear(128, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

    def _validate(
        self,
        points: Tensor,
        mask: Tensor,
        metrics: Tensor,
        trim_neighbors: Tensor,
    ) -> None:
        if points.ndim != 3 or points.shape[-1] != 10:
            raise ValueError("face_points must have shape [F, P, 10]")
        if mask.shape != points.shape[:2] or mask.dtype != torch.bool:
            raise ValueError("face_point_mask must be bool with shape [F, P]")
        if metrics.shape != points.shape[:2] + (3,):
            raise ValueError("face_surface_metric must have shape [F, P, 3]")
        if not points.is_floating_point() or not metrics.is_floating_point():
            raise ValueError("surface inputs must use floating-point dtypes")
        if not torch.isfinite(points).all() or not torch.isfinite(metrics).all():
            raise ValueError("surface inputs must contain only finite values")
        if torch.any(mask.sum(dim=1) == 0):
            raise ValueError("every face must have at least one valid point")
        if self.trim_aware:
            if trim_neighbors is None:
                raise ValueError(
                    "face_trim_neighbors is required in trim-aware mode"
                )
            if trim_neighbors.shape != points.shape[:2] + (
                self.neighbor_count,
            ):
                raise ValueError(
                    "face_trim_neighbors has an invalid shape"
                )
            if trim_neighbors.dtype not in (
                torch.int16,
                torch.int32,
                torch.int64,
            ):
                raise ValueError("face_trim_neighbors must be integer")

    def _pair_distance(
        self, points: Tensor, metrics: Tensor
    ) -> Tensor:
        if self.neighbor_metric == "xyz":
            return torch.cdist(points[..., :3], points[..., :3]).square()
        uv = points[..., 6:8]
        delta = uv.unsqueeze(2) - uv.unsqueeze(1)
        if self.neighbor_metric == "uv":
            return delta.square().sum(dim=-1)
        average = (metrics.unsqueeze(2) + metrics.unsqueeze(1)) * 0.5
        distance = (
            average[..., 0] * delta[..., 0].square()
            + 2.0
            * average[..., 1]
            * delta[..., 0]
            * delta[..., 1]
            + average[..., 2] * delta[..., 1].square()
        )
        return distance.clamp_min(0.0)

    def _knn(self, points: Tensor, mask: Tensor, metrics: Tensor) -> Tensor:
        distance = self._pair_distance(points, metrics)
        valid_pairs = mask.unsqueeze(1) & mask.unsqueeze(2)
        point_count = points.shape[1]
        diagonal = torch.eye(
            point_count, device=points.device, dtype=torch.bool
        ).unsqueeze(0)
        distance = distance.masked_fill(~valid_pairs | diagonal, float("inf"))
        count = min(self.neighbor_count, max(point_count - 1, 1))
        neighbors = distance.topk(count, dim=-1, largest=False).indices
        if count < self.neighbor_count:
            padding = neighbors.new_full(
                neighbors.shape[:-1] + (self.neighbor_count - count,), -1
            )
            neighbors = torch.cat((neighbors, padding), dim=-1)
        selected_distance = torch.gather(
            distance, -1, neighbors.clamp_min(0)
        )
        return neighbors.masked_fill(~torch.isfinite(selected_distance), -1)

    @staticmethod
    def _gather(values: Tensor, indices: Tensor) -> Tensor:
        face_count, point_count, feature_dim = values.shape
        safe = indices.clamp(min=0, max=max(point_count - 1, 0)).long()
        expanded = values.unsqueeze(1).expand(
            face_count, point_count, point_count, feature_dim
        )
        return torch.gather(
            expanded,
            2,
            safe.unsqueeze(-1).expand(*safe.shape, feature_dim),
        )

    def forward(
        self,
        face_points: Tensor,
        face_point_mask: Tensor,
        face_surface_metric: Tensor,
        face_trim_neighbors: Tensor = None,
    ) -> Tensor:
        self._validate(
            face_points,
            face_point_mask,
            face_surface_metric,
            face_trim_neighbors,
        )
        valid = face_point_mask.unsqueeze(-1)
        counts = face_point_mask.sum(dim=1).to(face_points.dtype).unsqueeze(-1)
        xyz = face_points[..., :3]
        center = (xyz * valid).sum(dim=1) / counts
        centered = xyz - center.unsqueeze(1)
        radius = torch.linalg.vector_norm(centered, dim=-1)
        scale = (
            radius.masked_fill(~face_point_mask, 0.0)
            .amax(dim=1)
            .clamp_min(self.eps)
            .view(-1, 1, 1)
        )
        normalized_points = torch.cat(
            (
                centered / scale,
                face_points[..., 3:],
                face_surface_metric,
            ),
            dim=-1,
        )
        latent = self.point_mlp(normalized_points)
        neighbors = (
            face_trim_neighbors.long()
            if self.trim_aware
            else self._knn(face_points, face_point_mask, face_surface_metric)
        )
        neighbor_valid = neighbors >= 0
        neighbor_valid = neighbor_valid & self._gather(
            face_point_mask.unsqueeze(-1), neighbors
        ).squeeze(-1)
        neighbor_latent = self._gather(latent, neighbors)
        neighbor_xyz = self._gather(centered / scale, neighbors)
        neighbor_uv = self._gather(face_points[..., 6:8], neighbors)
        delta_xyz = neighbor_xyz - (centered / scale).unsqueeze(2)
        delta_uv = neighbor_uv - face_points[..., 6:8].unsqueeze(2)
        neighbor_metrics = self._gather(face_surface_metric, neighbors)
        average_metric = (
            neighbor_metrics + face_surface_metric.unsqueeze(2)
        ) * 0.5
        intrinsic = (
            average_metric[..., 0] * delta_uv[..., 0].square()
            + 2.0
            * average_metric[..., 1]
            * delta_uv[..., 0]
            * delta_uv[..., 1]
            + average_metric[..., 2] * delta_uv[..., 1].square()
        ).clamp_min(self.eps).sqrt().unsqueeze(-1)
        edge = torch.cat(
            (neighbor_latent, delta_xyz, delta_uv, intrinsic), dim=-1
        )
        edge = self.edge_mlp(edge)
        negative_infinity = torch.finfo(edge.dtype).min
        edge = edge.masked_fill(
            ~neighbor_valid.unsqueeze(-1), negative_infinity
        )
        local = edge.amax(dim=2)
        no_neighbor = ~neighbor_valid.any(dim=2)
        local = local.masked_fill(no_neighbor.unsqueeze(-1), 0.0)
        local = self.local_mlp(torch.cat((latent, local), dim=-1))

        coarse_limit = min(self.coarse_count, face_points.shape[1])
        coarse = local[:, :coarse_limit]
        coarse_mask = face_point_mask[:, :coarse_limit]
        coarse = coarse.masked_fill(
            ~coarse_mask.unsqueeze(-1), negative_infinity
        )
        pooled = coarse.amax(dim=1)
        return self.output_mlp(pooled)


class GeometryFeatureFusion(nn.Module):
    """Fuse optional geometry branches with exact base initialization."""

    def __init__(
        self, hidden_dim: int, mode: str = "add", branch_count: int = 2
    ) -> None:
        super().__init__()
        if mode not in ("add", "gated"):
            raise ValueError("mode must be add or gated")
        if hidden_dim < 1 or branch_count < 1:
            raise ValueError("hidden_dim and branch_count must be positive")
        self.mode = mode
        self.branch_count = int(branch_count)
        self.gates = nn.ModuleList()
        if mode == "gated":
            for _ in range(branch_count):
                gate = nn.Sequential(
                    nn.LayerNorm(2 * hidden_dim),
                    nn.Linear(2 * hidden_dim, hidden_dim),
                    nn.GELU(),
                    nn.Linear(hidden_dim, hidden_dim),
                )
                nn.init.zeros_(gate[-1].weight)
                nn.init.zeros_(gate[-1].bias)
                self.gates.append(gate)

    def reset_output_gates(self) -> None:
        """Restore exact-base initialization after project-wide init passes."""
        for gate in self.gates:
            nn.init.zeros_(gate[-1].weight)
            nn.init.zeros_(gate[-1].bias)

    def forward(self, base: Tensor, branches, scales) -> Tensor:
        if len(branches) != self.branch_count or len(scales) != self.branch_count:
            raise ValueError("branch inputs must match branch_count")
        output = base
        for index, (branch, scale) in enumerate(zip(branches, scales)):
            if branch is None:
                continue
            if branch.shape != base.shape:
                raise ValueError("branch shape must match base shape")
            contribution = float(scale) * branch
            if self.mode == "gated":
                gate = torch.tanh(
                    self.gates[index](torch.cat((base, branch), dim=-1))
                )
                contribution = gate * contribution
            output = output + contribution
        return output
