"""Core GeoPeak3D network.

The implementation keeps only the released method: layer-wise NMS/Top-K,
learned global candidate weighting, Gaussian geometry projection, anisotropic
frustum fusion, residual stGCF correction, and offset-aware expectation.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def _group_count(channels: int) -> int:
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class CandidateGeometryField(nn.Module):
    """Extract, score, and project competing peaks onto the frustum grid."""

    def __init__(
        self,
        depth_bins: int = 9,
        candidates_per_depth: int = 4,
        nms_size: int = 7,
        embedding_dim: int = 64,
        score_prior_weight: float = 2.0,
        projection_sigma: float = 2.0,
    ) -> None:
        super().__init__()
        if nms_size % 2 != 1:
            raise ValueError("nms_size must be odd")
        self.depth_bins = depth_bins
        self.candidates_per_depth = candidates_per_depth
        self.nms_size = nms_size
        self.score_prior_weight = score_prior_weight
        self.projection_sigma = projection_sigma
        self.token_embedding = nn.Sequential(
            nn.Linear(11, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU(),
        )
        self.depth_embedding = nn.Embedding(depth_bins, embedding_dim)
        self.selector = nn.Linear(embedding_dim, 1)

    def _candidates(self, volume: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        planes = volume[:, 0]
        batch, depth, height, width = planes.shape
        if depth != self.depth_bins:
            raise ValueError(f"Expected {self.depth_bins} depth layers, got {depth}")

        flat_planes = planes.reshape(batch * depth, 1, height, width)
        local_max = F.max_pool2d(
            flat_planes, self.nms_size, stride=1, padding=self.nms_size // 2
        ).reshape(batch, depth, height, width)
        nms_scores = torch.where(
            planes >= local_max - 1e-6, planes, planes.new_full((), -1e4)
        )
        scores, indices = nms_scores.flatten(2).topk(self.candidates_per_depth, dim=2)
        candidate_v = torch.div(indices, width, rounding_mode="floor").to(volume.dtype)
        candidate_u = (indices % width).to(volume.dtype)
        candidate_xy = torch.stack((candidate_u, candidate_v), dim=-1)
        local_mean = F.avg_pool2d(
            flat_planes, self.nms_size, stride=1, padding=self.nms_size // 2
        ).reshape(batch, depth, -1).gather(2, indices)
        return candidate_xy, scores, local_mean

    def forward(
        self, volume: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        candidate_xy, scores, local_mean = self._candidates(volume)
        planes = volume[:, 0]
        batch, depth, height, width = planes.shape
        count = self.candidates_per_depth
        flat_planes = planes.flatten(2)
        layer_mean = flat_planes.mean(dim=2)
        layer_std = flat_planes.std(dim=2).clamp_min(1e-5)

        raw_index = planes.flatten(1).argmax(dim=1) % (height * width)
        raw_xy = torch.stack(
            ((raw_index % width).to(volume.dtype),
             torch.div(raw_index, width, rounding_mode="floor").to(volume.dtype)),
            dim=1,
        )
        depth_id = torch.arange(depth, device=volume.device, dtype=volume.dtype)
        depth_feature = depth_id.view(1, depth, 1).expand(batch, depth, count) / max(depth - 1, 1)
        tokens = torch.stack(
            (
                scores,
                scores - local_mean,
                (scores - layer_mean[:, :, None]) / layer_std[:, :, None],
                local_mean,
                layer_mean[:, :, None].expand_as(scores),
                layer_std[:, :, None].expand_as(scores),
                candidate_xy[..., 0] / max(width - 1, 1),
                candidate_xy[..., 1] / max(height - 1, 1),
                depth_feature,
                (candidate_xy[..., 0] - raw_xy[:, None, None, 0]) / max(width - 1, 1),
                (candidate_xy[..., 1] - raw_xy[:, None, None, 1]) / max(height - 1, 1),
            ),
            dim=-1,
        )

        encoded = self.token_embedding(tokens)
        encoded = encoded + self.depth_embedding(
            depth_id.long().view(1, depth, 1).expand(batch, depth, count)
        )
        learned_logits = self.selector(encoded).squeeze(-1)
        flat_scores = scores.flatten(1)
        standardized = (flat_scores - flat_scores.mean(dim=1, keepdim=True)) / (
            flat_scores.std(dim=1, keepdim=True) + 1e-5
        )
        candidate_logits = learned_logits.flatten(1) + self.score_prior_weight * standardized
        weights = torch.softmax(candidate_logits, dim=1).reshape(batch, depth, count)

        grid_v = torch.arange(height, device=volume.device, dtype=volume.dtype).view(1, 1, 1, height, 1)
        grid_u = torch.arange(width, device=volume.device, dtype=volume.dtype).view(1, 1, 1, 1, width)
        distance2 = (
            (grid_u - candidate_xy[..., 0, None, None]).square()
            + (grid_v - candidate_xy[..., 1, None, None]).square()
        )
        kernels = torch.exp(-0.5 * distance2 / self.projection_sigma**2)
        geometry = (weights[..., None, None] * kernels).sum(dim=2).unsqueeze(1)
        return geometry, candidate_xy.reshape(batch, depth * count, 2), candidate_logits


class AnisotropicResidualBlock(nn.Module):
    """Separate image-plane and ray-wise aggregation on a calibrated frustum."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        groups = _group_count(channels)
        self.in_plane = nn.Sequential(
            nn.Conv3d(channels, channels, (1, 3, 3), padding=(0, 1, 1), bias=False),
            nn.GroupNorm(groups, channels),
            nn.ReLU(inplace=True),
        )
        self.ray_wise = nn.Sequential(
            nn.Conv3d(channels, channels, (3, 1, 1), padding=(1, 0, 0), bias=False),
            nn.GroupNorm(groups, channels),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return F.relu(inputs + self.ray_wise(self.in_plane(inputs)), inplace=True)


class GeoPeak3D(nn.Module):
    """Geometry-guided multi-peak disambiguation for 3D acoustic localization."""

    def __init__(
        self,
        depth_bins: int = 9,
        candidates_per_depth: int = 4,
        channels: int = 8,
        residual_blocks: int = 3,
        nms_size: int = 7,
        candidate_dim: int = 64,
        score_prior_weight: float = 2.0,
        projection_sigma: float = 2.0,
        raw_scale_max: float = 100.0,
        offset_bounds: tuple[float, float, float] = (0.5, 2.0, 2.0),
    ) -> None:
        super().__init__()
        self.depth_bins = depth_bins
        self.raw_scale_max = raw_scale_max
        self.register_buffer("offset_bounds", torch.tensor(offset_bounds).view(1, 3, 1, 1, 1))
        self.geometry = CandidateGeometryField(
            depth_bins=depth_bins,
            candidates_per_depth=candidates_per_depth,
            nms_size=nms_size,
            embedding_dim=candidate_dim,
            score_prior_weight=score_prior_weight,
            projection_sigma=projection_sigma,
        )
        groups = _group_count(channels)
        self.stem = nn.Sequential(
            nn.Conv3d(4, channels, (1, 3, 3), padding=(0, 1, 1), bias=False),
            nn.GroupNorm(groups, channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(channels, channels, (3, 1, 1), padding=(1, 0, 0), bias=False),
            nn.GroupNorm(groups, channels),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.Sequential(
            *(AnisotropicResidualBlock(channels) for _ in range(residual_blocks))
        )
        self.residual_head = nn.Conv3d(channels, 1, 1)
        self.offset_head = nn.Conv3d(channels, 3, 1)
        self.log_raw_scale = nn.Parameter(torch.tensor(3.0))
        nn.init.zeros_(self.residual_head.weight)
        nn.init.zeros_(self.residual_head.bias)
        nn.init.zeros_(self.offset_head.weight)
        nn.init.zeros_(self.offset_head.bias)

    @staticmethod
    def dense_response_features(volume: torch.Tensor) -> torch.Tensor:
        layer_mean = volume.mean(dim=(-2, -1), keepdim=True)
        layer_std = volume.std(dim=(-2, -1), keepdim=True).clamp_min(1e-5)
        volume_mean = volume.mean(dim=(-3, -2, -1), keepdim=True)
        volume_std = volume.std(dim=(-3, -2, -1), keepdim=True).clamp_min(1e-5)
        return torch.cat(
            (volume, (volume - layer_mean) / layer_std, (volume - volume_mean) / volume_std),
            dim=1,
        )

    def forward(self, volume: torch.Tensor, return_aux: bool = False):
        if volume.ndim != 5 or volume.shape[1] != 1 or volume.shape[2] != self.depth_bins:
            raise ValueError(
                f"Expected [B, 1, {self.depth_bins}, H, W], got {tuple(volume.shape)}"
            )
        geometry, candidates, candidate_logits = self.geometry(volume)
        features = self.blocks(self.stem(torch.cat((self.dense_response_features(volume), geometry), dim=1)))
        alpha = self.log_raw_scale.exp().clamp(max=self.raw_scale_max)
        logits = alpha * volume + self.residual_head(features)
        offsets = torch.tanh(self.offset_head(features)) * self.offset_bounds.to(volume)
        if return_aux:
            return logits, offsets, candidates, candidate_logits
        return logits, offsets

    @staticmethod
    def probability(logits: torch.Tensor) -> torch.Tensor:
        return torch.softmax(logits.flatten(1), dim=1).reshape_as(logits)

    @staticmethod
    def continuous_coordinates(logits: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
        """Return the offset-aware expected coordinate in ``(depth, v, u)`` order."""
        batch, _, depth, height, width = logits.shape
        grid = torch.stack(
            torch.meshgrid(
                torch.arange(depth, device=logits.device, dtype=logits.dtype),
                torch.arange(height, device=logits.device, dtype=logits.dtype),
                torch.arange(width, device=logits.device, dtype=logits.dtype),
                indexing="ij",
            ),
            dim=0,
        ).unsqueeze(0)
        return (GeoPeak3D.probability(logits) * (grid + offsets)).sum(dim=(-3, -2, -1))

    @staticmethod
    def candidate_loss(
        candidates_xy: torch.Tensor,
        candidate_logits: torch.Tensor,
        target_xy: torch.Tensor,
        sigma: float = 3.0,
    ) -> torch.Tensor:
        distance2 = (candidates_xy - target_xy[:, None]).square().sum(dim=-1)
        target = torch.softmax(-distance2 / (2.0 * sigma**2), dim=1)
        return -(target * torch.log_softmax(candidate_logits, dim=1)).sum(dim=1).mean()
