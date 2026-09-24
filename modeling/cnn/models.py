from __future__ import annotations

import math
from dataclasses import replace

import torch
from torch import nn

from .config import ModelConfig


def _group_count(channels: int) -> int:
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class GeneralizedMeanPool(nn.Module):
    def __init__(self, dimensions: int, initial_p: float = 3.0, max_p: float = 6.0) -> None:
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
        return features.relu().clamp_min(1e-6).pow(self.p).mean(dim=spatial).pow(1.0 / self.p)


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
            maximum = features.amax(dim=spatial)
            return torch.cat([average, maximum], dim=1)
        if self.mode == "gap_gem":
            return torch.cat([average, self.gem(features)], dim=1)
        raise ValueError(f"Pooling desconocido: {self.mode}")


class ResidualBlock3D(nn.Module):
    def __init__(self, input_channels: int, output_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv3d(
            input_channels, output_channels, 3, stride=stride, padding=1, bias=False
        )
        self.norm1 = nn.GroupNorm(_group_count(output_channels), output_channels)
        self.conv2 = nn.Conv3d(output_channels, output_channels, 3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(_group_count(output_channels), output_channels)
        self.activation = nn.SiLU(inplace=True)
        self.skip = (
            nn.Identity()
            if stride == 1 and input_channels == output_channels
            else nn.Sequential(
                nn.Conv3d(input_channels, output_channels, 1, stride=stride, bias=False),
                nn.GroupNorm(_group_count(output_channels), output_channels),
            )
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        residual = self.skip(inputs)
        output = self.activation(self.norm1(self.conv1(inputs)))
        output = self.norm2(self.conv2(output))
        return self.activation(output + residual)


class Compact3DEncoder(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        base = config.base_channels
        channels = (base, base * 2, base * 4, base * 6)
        self.stem = nn.Sequential(
            nn.Conv3d(config.in_channels, channels[0], 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(_group_count(channels[0]), channels[0]),
            nn.SiLU(inplace=True),
        )
        self.stages = nn.ModuleList(
            [
                ResidualBlock3D(channels[0], channels[0]),
                nn.Sequential(
                    ResidualBlock3D(channels[0], channels[1], stride=2),
                    ResidualBlock3D(channels[1], channels[1]),
                ),
                nn.Sequential(
                    ResidualBlock3D(channels[1], channels[2], stride=2),
                    ResidualBlock3D(channels[2], channels[2]),
                ),
                nn.Sequential(
                    ResidualBlock3D(channels[2], channels[3], stride=2),
                    ResidualBlock3D(channels[3], channels[3]),
                ),
            ]
        )
        self.pool = MultiGlobalPool(
            3, config.pooling, config.gem_initial_p, config.gem_max_p
        )
        pooled_dim = channels[-1] * self.pool.multiplier
        self.projection = nn.Sequential(
            nn.LayerNorm(pooled_dim),
            nn.Linear(pooled_dim, config.image_embedding_dim),
            nn.SiLU(inplace=True),
            nn.Dropout(config.dropout),
        )
        self.output_dim = config.image_embedding_dim
        self.last_spatial: torch.Tensor | None = None

    @property
    def gradcam_target_layer(self) -> nn.Module:
        return self.stages[-1]

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        output = self.stem(image)
        for stage in self.stages:
            output = stage(output)
        self.last_spatial = output
        return self.projection(self.pool(output))


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
        self.output_dim = config.image_embedding_dim

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        if image.ndim != 5 or image.shape[1] != 3:
            raise ValueError("2.5D espera Bx3xKxHxW.")
        embeddings = torch.stack(
            [self.shared(image[:, view]) for view in range(3)], dim=1
        )
        weights = torch.softmax(self.plane_attention(embeddings), dim=1)
        attentive = (weights * embeddings).sum(dim=1)
        maximum = embeddings.max(dim=1).values
        return self.projection(torch.cat([attentive, maximum], dim=1))


class TabularEncoder(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, dropout: float) -> None:
        super().__init__()
        hidden = max(32, min(96, input_dim * 4))
        self.network = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, output_dim),
            nn.SiLU(inplace=True),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values)


class HybridDaTClassifier(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.image_encoder: nn.Module = (
            Compact3DEncoder(config)
            if config.architecture == "3d"
            else Triplanar25DEncoder(config)
        )
        fusion_dim = config.image_embedding_dim
        self.radiomics_encoder: nn.Module | None = None
        if config.feature_variant != "image_only":
            if config.radiomics_dim <= 0:
                raise ValueError("La variante radiomics requiere radiomics_dim > 0.")
            self.radiomics_encoder = TabularEncoder(
                config.radiomics_dim, config.tabular_embedding_dim, config.dropout
            )
            fusion_dim += config.tabular_embedding_dim
        self.fusion_head = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, 64),
            nn.SiLU(inplace=True),
            nn.Dropout(config.dropout),
            nn.Linear(64, 1),
        )
        self.sbr_adapter: nn.Module | None = None
        if config.feature_variant == "image_radiomics_sbr":
            if config.sbr_dim <= 0:
                raise ValueError("La variante SBR requiere sbr_dim > 0.")
            self.sbr_adapter = nn.Sequential(
                nn.LayerNorm(config.sbr_dim),
                nn.Linear(config.sbr_dim, config.sbr_hidden_dim),
                nn.Tanh(),
                nn.Dropout(config.dropout),
                nn.Linear(config.sbr_hidden_dim, 1),
            )

    def forward(
        self,
        image: torch.Tensor,
        radiomics: torch.Tensor | None = None,
        sbr: torch.Tensor | None = None,
        sbr_valid: torch.Tensor | None = None,
        *,
        tabular: torch.Tensor | None = None,
        return_embeddings: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if radiomics is None and tabular is not None:
            radiomics = tabular
        image_embedding = self.image_encoder(image)
        pieces = [image_embedding]
        tabular_embedding: torch.Tensor | None = None
        if self.radiomics_encoder is not None:
            if radiomics is None:
                raise ValueError("Faltan radiomics para esta variante.")
            tabular_embedding = self.radiomics_encoder(radiomics)
            pieces.append(tabular_embedding)
        fused = torch.cat(pieces, dim=1)
        base_logit = self.fusion_head(fused).squeeze(1)
        logit = base_logit
        sbr_delta = torch.zeros_like(base_logit)
        if self.sbr_adapter is not None:
            if sbr is None or sbr_valid is None:
                raise ValueError("Faltan SBR o su mascara de validez.")
            valid = sbr_valid.reshape(-1).to(dtype=base_logit.dtype)
            gated_values = sbr * valid[:, None]
            # Center the residual adapter at the neutral (all-zero) SBR vector.
            # This prevents the mere fact that background QC is valid from
            # becoming an intercept-like technical predictor.
            neutral = self.sbr_adapter(torch.zeros_like(gated_values))
            sbr_delta = (
                self.sbr_adapter(gated_values) - neutral
            ).squeeze(1) * valid
            logit = base_logit + sbr_delta
        if return_embeddings:
            return logit, {
                "image": image_embedding,
                "tabular": tabular_embedding
                if tabular_embedding is not None
                else image_embedding.new_zeros((len(image_embedding), 0)),
                "fused": fused,
                "sbr_delta": sbr_delta[:, None],
            }
        return logit


def with_input_dimensions(
    config: ModelConfig, radiomics_dim: int, sbr_dim: int
) -> ModelConfig:
    return replace(config, radiomics_dim=radiomics_dim, sbr_dim=sbr_dim)
