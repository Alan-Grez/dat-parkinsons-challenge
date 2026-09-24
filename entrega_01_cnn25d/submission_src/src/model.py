from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn


@dataclass(frozen=True)
class ModelConfig:
    architecture: str = "2.5d"
    feature_variant: str = "image_only"
    in_channels: int = 1
    slices_per_view: int = 5
    base_channels: int = 12
    image_embedding_dim: int = 128
    tabular_embedding_dim: int = 32
    radiomics_dim: int = 0
    sbr_dim: int = 0
    sbr_hidden_dim: int = 16
    dropout: float = 0.25
    pooling: str = "gap_max"
    gem_initial_p: float = 3.0
    gem_max_p: float = 6.0

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ModelConfig":
        fields = cls.__dataclass_fields__
        return cls(**{key: value for key, value in payload.items() if key in fields})


def _group_count(channels: int) -> int:
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class GeneralizedMeanPool(nn.Module):
    def __init__(self, dimensions: int, initial_p: float, max_p: float) -> None:
        super().__init__()
        self.dimensions = dimensions
        self.max_p = max_p
        fraction = (initial_p - 1.0) / max(max_p - 1.0, 1e-6)
        fraction = min(max(fraction, 1e-4), 1.0 - 1e-4)
        self.raw_p = nn.Parameter(torch.tensor(math.log(fraction / (1.0 - fraction))))

    @property
    def p(self) -> torch.Tensor:
        return 1.0 + (self.max_p - 1.0) * torch.sigmoid(self.raw_p)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        spatial = tuple(range(2, 2 + self.dimensions))
        return (
            features.relu()
            .clamp_min(1e-6)
            .pow(self.p)
            .mean(dim=spatial)
            .pow(1.0 / self.p)
        )


class MultiGlobalPool(nn.Module):
    def __init__(self, dimensions: int, mode: str, initial_p: float, max_p: float) -> None:
        super().__init__()
        self.dimensions = dimensions
        self.mode = mode
        self.gem = GeneralizedMeanPool(dimensions, initial_p, max_p)

    @property
    def multiplier(self) -> int:
        return 1 if self.mode == "gap" else 2

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        spatial = tuple(range(2, 2 + self.dimensions))
        average = features.mean(dim=spatial)
        if self.mode == "gap":
            return average
        if self.mode == "gap_max":
            return torch.cat([average, features.amax(dim=spatial)], dim=1)
        if self.mode == "gap_gem":
            return torch.cat([average, self.gem(features)], dim=1)
        raise ValueError(f"Unknown pooling mode: {self.mode}")


class ResidualBlock2D(nn.Module):
    def __init__(self, input_channels: int, output_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            input_channels, output_channels, 3, stride=stride, padding=1, bias=False
        )
        self.norm1 = nn.GroupNorm(_group_count(output_channels), output_channels)
        self.conv2 = nn.Conv2d(output_channels, output_channels, 3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(_group_count(output_channels), output_channels)
        self.skip = (
            nn.Identity()
            if stride == 1 and input_channels == output_channels
            else nn.Sequential(
                nn.Conv2d(input_channels, output_channels, 1, stride=stride, bias=False),
                nn.GroupNorm(_group_count(output_channels), output_channels),
            )
        )
        self.activation = nn.SiLU(inplace=True)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        residual = self.skip(inputs)
        output = self.activation(self.norm1(self.conv1(inputs)))
        output = self.norm2(self.conv2(output))
        return self.activation(output + residual)


class CompactPlaneEncoder(nn.Module):
    def __init__(self, slices: int, base_channels: int, config: ModelConfig) -> None:
        super().__init__()
        channels = (base_channels, base_channels * 2, base_channels * 4)
        self.network = nn.Sequential(
            nn.Conv2d(slices, channels[0], 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(_group_count(channels[0]), channels[0]),
            nn.SiLU(inplace=True),
            ResidualBlock2D(channels[0], channels[0]),
            ResidualBlock2D(channels[0], channels[1], stride=2),
            ResidualBlock2D(channels[1], channels[1]),
            ResidualBlock2D(channels[1], channels[2], stride=2),
            ResidualBlock2D(channels[2], channels[2]),
        )
        self.pool = MultiGlobalPool(
            2, config.pooling, config.gem_initial_p, config.gem_max_p
        )
        self.output_dim = channels[-1] * self.pool.multiplier

    def forward(self, plane: torch.Tensor) -> torch.Tensor:
        return self.pool(self.network(plane))


class Triplanar25DEncoder(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.shared = CompactPlaneEncoder(
            config.slices_per_view, config.base_channels, config
        )
        plane_dim = self.shared.output_dim
        self.plane_attention = nn.Sequential(
            nn.LayerNorm(plane_dim), nn.Linear(plane_dim, 1, bias=False)
        )
        self.projection = nn.Sequential(
            nn.LayerNorm(plane_dim * 2),
            nn.Linear(plane_dim * 2, config.image_embedding_dim),
            nn.SiLU(inplace=True),
            nn.Dropout(config.dropout),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        if image.ndim != 5 or image.shape[1] != 3:
            raise ValueError("2.5D input must have shape Bx3xKxHxW.")
        embeddings = torch.stack(
            [self.shared(image[:, view]) for view in range(3)], dim=1
        )
        weights = torch.softmax(self.plane_attention(embeddings), dim=1)
        attentive = (weights * embeddings).sum(dim=1)
        maximum = embeddings.max(dim=1).values
        return self.projection(torch.cat([attentive, maximum], dim=1))


class HybridDaTClassifier(nn.Module):
    """Exact image-only 2.5D architecture used by the selected five-fold ensemble."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        if config.architecture != "2.5d" or config.feature_variant != "image_only":
            raise ValueError("This submission only supports the selected 2.5D image-only model.")
        self.image_encoder = Triplanar25DEncoder(config)
        self.fusion_head = nn.Sequential(
            nn.LayerNorm(config.image_embedding_dim),
            nn.Linear(config.image_embedding_dim, 64),
            nn.SiLU(inplace=True),
            nn.Dropout(config.dropout),
            nn.Linear(64, 1),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.fusion_head(self.image_encoder(image)).squeeze(1)
