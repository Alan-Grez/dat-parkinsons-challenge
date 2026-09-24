from __future__ import annotations

from dataclasses import replace

import torch
from torch import nn

from .config import ModelConfig


def _groups(channels: int) -> int:
    for value in (8, 4, 2, 1):
        if channels % value == 0:
            return value
    return 1


class Residual2D(nn.Module):
    def __init__(self, input_channels: int, output_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 3, stride=stride, padding=1, bias=False),
            nn.GroupNorm(_groups(output_channels), output_channels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(output_channels, output_channels, 3, padding=1, bias=False),
            nn.GroupNorm(_groups(output_channels), output_channels),
        )
        self.skip = (
            nn.Identity()
            if input_channels == output_channels and stride == 1
            else nn.Sequential(
                nn.Conv2d(input_channels, output_channels, 1, stride=stride, bias=False),
                nn.GroupNorm(_groups(output_channels), output_channels),
            )
        )
        self.activation = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.activation(self.main(inputs) + self.skip(inputs))


class Residual3D(nn.Module):
    def __init__(self, input_channels: int, output_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv3d(input_channels, output_channels, 3, stride=stride, padding=1, bias=False),
            nn.GroupNorm(_groups(output_channels), output_channels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv3d(output_channels, output_channels, 3, padding=1, bias=False),
            nn.GroupNorm(_groups(output_channels), output_channels),
        )
        self.skip = (
            nn.Identity()
            if input_channels == output_channels and stride == 1
            else nn.Sequential(
                nn.Conv3d(input_channels, output_channels, 1, stride=stride, bias=False),
                nn.GroupNorm(_groups(output_channels), output_channels),
            )
        )
        self.activation = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.activation(self.main(inputs) + self.skip(inputs))


class GlobalPool(nn.Module):
    def __init__(self, channels: int, dimensions: int, mode: str) -> None:
        super().__init__()
        self.dimensions = dimensions
        self.mode = mode
        self.attention = (
            nn.Conv2d(channels, 1, 1)
            if mode == "attention" and dimensions == 2
            else nn.Conv3d(channels, 1, 1)
            if mode == "attention"
            else None
        )
        self.raw_gem_p = nn.Parameter(torch.tensor(0.0))

    @property
    def multiplier(self) -> int:
        return 2 if self.mode in {"gap_max", "gap_gem"} else 1

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        spatial = tuple(range(2, features.ndim))
        if self.mode == "gmp":
            return features.amax(dim=spatial)
        average = features.mean(dim=spatial)
        if self.mode == "gap_max":
            return torch.cat([average, features.amax(dim=spatial)], dim=1)
        if self.mode == "gap_gem":
            p = 1.0 + 5.0 * torch.sigmoid(self.raw_gem_p)
            gem = features.relu().clamp_min(1e-6).pow(p).mean(dim=spatial).pow(1.0 / p)
            return torch.cat([average, gem], dim=1)
        if self.mode == "attention":
            if self.attention is None:
                raise RuntimeError("Pooling attention no inicializado.")
            logits = self.attention(features).flatten(2)
            weights = torch.softmax(logits, dim=-1)
            return (features.flatten(2) * weights).sum(dim=-1)
        raise ValueError(f"Pooling desconocido: {self.mode}")


class HamburgSlabEncoder(nn.Module):
    """Small residual 16-32-64 encoder for a 72x72, 12-mm axial slab."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        base = config.base_channels
        self.stem = nn.Sequential(
            nn.Conv2d(config.in_channels, base, 5, stride=3, padding=2, bias=False),
            nn.GroupNorm(_groups(base), base),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.stages = nn.Sequential(
            Residual2D(base, base),
            Residual2D(base, base * 2, stride=2),
            Residual2D(base * 2, base * 4, stride=2),
            Residual2D(base * 4, base * 4),
        )
        self.pool = GlobalPool(base * 4, 2, config.pooling)
        pooled = base * 4 * self.pool.multiplier
        self.projection = nn.Sequential(
            nn.LayerNorm(pooled),
            nn.Linear(pooled, config.image_embedding_dim),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(config.dropout),
        )
        self.output_dim = config.image_embedding_dim

    @property
    def gradcam_target_layer(self) -> nn.Module:
        return self.stages[-1]

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.projection(self.pool(self.stages(self.stem(image))))


class CompactVolumeEncoder(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        base = config.base_channels
        self.stem = nn.Sequential(
            nn.Conv3d(config.in_channels, base, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(_groups(base), base),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.stages = nn.Sequential(
            Residual3D(base, base),
            Residual3D(base, base * 2, stride=2),
            Residual3D(base * 2, base * 4, stride=2),
            Residual3D(base * 4, base * 4),
        )
        self.pool = GlobalPool(base * 4, 3, config.pooling)
        pooled = base * 4 * self.pool.multiplier
        self.projection = nn.Sequential(
            nn.LayerNorm(pooled),
            nn.Linear(pooled, config.image_embedding_dim),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(config.dropout),
        )
        self.output_dim = config.image_embedding_dim

    @property
    def gradcam_target_layer(self) -> nn.Module:
        return self.stages[-1]

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.projection(self.pool(self.stages(self.stem(image))))


class PlaneEncoder(nn.Module):
    def __init__(self, slices: int, base: int, pooling: str) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(slices, base, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(_groups(base), base),
            nn.LeakyReLU(0.1, inplace=True),
            Residual2D(base, base),
            Residual2D(base, base * 2, stride=2),
            Residual2D(base * 2, base * 4, stride=2),
        )
        self.pool = GlobalPool(base * 4, 2, pooling)
        self.output_dim = base * 4 * self.pool.multiplier

    def forward(self, plane: torch.Tensor) -> torch.Tensor:
        return self.pool(self.network(plane))


class TriplanarEncoder(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.shared = PlaneEncoder(config.slices_per_view, config.base_channels, config.pooling)
        dimension = self.shared.output_dim
        self.attention = nn.Sequential(nn.LayerNorm(dimension), nn.Linear(dimension, 1, bias=False))
        self.projection = nn.Sequential(
            nn.LayerNorm(dimension * 2),
            nn.Linear(dimension * 2, config.image_embedding_dim),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(config.dropout),
        )
        self.output_dim = config.image_embedding_dim

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        if image.ndim != 5 or image.shape[1] != 3:
            raise ValueError("2.5D espera Bx3xKxHxW.")
        planes = torch.stack([self.shared(image[:, index]) for index in range(3)], dim=1)
        weights = torch.softmax(self.attention(planes), dim=1)
        attentive = (weights * planes).sum(dim=1)
        maximum = planes.amax(dim=1)
        return self.projection(torch.cat([attentive, maximum], dim=1))


class TabularEncoder(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, dropout: float) -> None:
        super().__init__()
        hidden = max(64, min(256, input_dim * 2))
        self.network = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, output_dim),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values)


class MultiTaskDaTClassifier(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        if config.architecture == "slab2d":
            self.image_encoder: nn.Module = HamburgSlabEncoder(config)
        elif config.architecture == "3d":
            self.image_encoder = CompactVolumeEncoder(config)
        else:
            self.image_encoder = TriplanarEncoder(config)
        fusion_dim = config.image_embedding_dim
        self.radiomics_encoder: nn.Module | None = None
        if config.feature_variant != "image_only":
            if config.radiomics_dim <= 0:
                raise ValueError("La fusion regional requiere radiomics_dim > 0.")
            self.radiomics_encoder = TabularEncoder(
                config.radiomics_dim, config.tabular_embedding_dim, config.dropout
            )
            fusion_dim += config.tabular_embedding_dim
        self.classifier = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, 64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(config.dropout),
            nn.Linear(64, 1),
        )
        self.auxiliary_head = nn.Sequential(
            nn.LayerNorm(config.image_embedding_dim),
            nn.Linear(config.image_embedding_dim, config.auxiliary_hidden_dim),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(config.auxiliary_hidden_dim, config.auxiliary_dim),
        )
        self.sbr_adapter: nn.Module | None = None
        if config.feature_variant == "image_radiomics_sbr":
            if config.sbr_dim <= 0:
                raise ValueError("La rama SBR requiere sbr_dim > 0.")
            self.sbr_adapter = nn.Sequential(
                nn.LayerNorm(config.sbr_dim),
                nn.Linear(config.sbr_dim, 16),
                nn.Tanh(),
                nn.Dropout(config.dropout),
                nn.Linear(16, 1),
            )

    def forward(
        self,
        image: torch.Tensor,
        radiomics: torch.Tensor | None = None,
        sbr: torch.Tensor | None = None,
        sbr_valid: torch.Tensor | None = None,
        *,
        tabular: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if radiomics is None and tabular is not None:
            radiomics = tabular
        image_embedding = self.image_encoder(image)
        pieces = [image_embedding]
        tabular_embedding = image_embedding.new_zeros((len(image_embedding), 0))
        if self.radiomics_encoder is not None:
            if radiomics is None:
                raise ValueError("Faltan las variables regionales del fold.")
            tabular_embedding = self.radiomics_encoder(radiomics)
            pieces.append(tabular_embedding)
        fused = torch.cat(pieces, dim=1)
        base_logit = self.classifier(fused).squeeze(1)
        sbr_delta = torch.zeros_like(base_logit)
        if self.sbr_adapter is not None:
            if sbr is None or sbr_valid is None:
                raise ValueError("Faltan SBR o mascara de validez.")
            valid = sbr_valid.reshape(-1).to(dtype=base_logit.dtype)
            gated = sbr * valid[:, None]
            neutral = self.sbr_adapter(torch.zeros_like(gated))
            sbr_delta = (self.sbr_adapter(gated) - neutral).squeeze(1) * valid
        return {
            "logit": base_logit + sbr_delta,
            "auxiliary": self.auxiliary_head(image_embedding),
            "image_embedding": image_embedding,
            "tabular_embedding": tabular_embedding,
            "sbr_delta": sbr_delta,
        }


def with_input_dimensions(
    config: ModelConfig,
    *,
    radiomics_dim: int,
    sbr_dim: int,
) -> ModelConfig:
    return replace(config, radiomics_dim=radiomics_dim, sbr_dim=sbr_dim)
