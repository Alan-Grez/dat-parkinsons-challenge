from __future__ import annotations

import torch
from torch import nn


def _groups(channels: int) -> int:
    for value in (8, 4, 2, 1):
        if channels % value == 0:
            return value
    return 1


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

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.activation(self.main(values) + self.skip(values))


class VolumeEncoder(nn.Module):
    def __init__(self, base_channels: int, embedding_dim: int, dropout: float) -> None:
        super().__init__()
        base = base_channels
        self.features = nn.Sequential(
            nn.Conv3d(1, base, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(_groups(base), base),
            nn.LeakyReLU(0.1, inplace=True),
            Residual3D(base, base),
            Residual3D(base, base * 2, stride=2),
            Residual3D(base * 2, base * 4, stride=2),
            Residual3D(base * 4, base * 4),
        )
        pooled = base * 8
        self.projection = nn.Sequential(
            nn.LayerNorm(pooled),
            nn.Linear(pooled, embedding_dim),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, volume: torch.Tensor) -> torch.Tensor:
        features = self.features(volume)
        spatial = tuple(range(2, features.ndim))
        pooled = torch.cat([features.mean(dim=spatial), features.amax(dim=spatial)], dim=1)
        return self.projection(pooled)


class DualStreamMultiTaskCNN(nn.Module):
    """Two independent 3D stems with gated, controlled concatenation.

    Intensity enters as the actual voxel values of ``volume_intensity`` and as
    four explicit magnitude scalars. The second branch receives an L2
    self-normalized view and therefore cannot recover those magnitudes.
    """

    def __init__(
        self,
        *,
        base_channels: int = 12,
        embedding_dim: int = 96,
        dropout: float = 0.25,
        auxiliary_dim: int = 9,
        magnitude_dim: int = 4,
    ) -> None:
        super().__init__()
        self.intensity_encoder = VolumeEncoder(base_channels, embedding_dim, dropout)
        self.pattern_encoder = VolumeEncoder(base_channels, embedding_dim, dropout)
        self.magnitude_encoder = nn.Sequential(
            nn.LayerNorm(magnitude_dim),
            nn.Linear(magnitude_dim, 24),
            nn.LeakyReLU(0.1, inplace=True),
        )
        # Separate sigmoid gates keep one branch from numerically swallowing the other.
        self.gates = nn.Sequential(
            nn.Linear(embedding_dim * 2 + 24, 3),
            nn.Sigmoid(),
        )
        fused_dim = embedding_dim * 2 + 24
        self.fusion = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Linear(fused_dim, 96),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(96, 1)
        self.uptake_head = nn.Linear(96, 6)
        self.side_head = nn.Linear(96, 1)
        self.asymmetry_head = nn.Linear(96, 1)
        self.fragmentation_head = nn.Linear(96, 1)
        self.auxiliary_dim = auxiliary_dim

    def forward(
        self,
        intensity: torch.Tensor,
        selfnorm: torch.Tensor,
        magnitude: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        intensity_embedding = self.intensity_encoder(intensity)
        pattern_embedding = self.pattern_encoder(selfnorm)
        magnitude_embedding = self.magnitude_encoder(magnitude)
        ungated = torch.cat([intensity_embedding, pattern_embedding, magnitude_embedding], dim=1)
        gates = self.gates(ungated)
        controlled = torch.cat(
            [
                intensity_embedding * gates[:, 0:1],
                pattern_embedding * gates[:, 1:2],
                magnitude_embedding * gates[:, 2:3],
            ],
            dim=1,
        )
        fused = self.fusion(controlled)
        uptake = self.uptake_head(fused)
        side = self.side_head(fused).squeeze(1)
        asymmetry = self.asymmetry_head(fused).squeeze(1)
        fragmentation = self.fragmentation_head(fused).squeeze(1)
        auxiliary = torch.cat(
            [uptake, side[:, None], asymmetry[:, None], fragmentation[:, None]], dim=1
        )
        return {
            "logit": self.classifier(fused).squeeze(1),
            "uptake": uptake,
            "side_logit": side,
            "asymmetry": asymmetry,
            "fragmentation": fragmentation,
            "auxiliary": auxiliary,
            "embedding": fused,
            "gates": gates,
        }


class DenseGraphConvolution(nn.Module):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim, bias=False)
        self.bias = nn.Parameter(torch.zeros(output_dim))

    def forward(self, nodes: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        return torch.bmm(adjacency, self.linear(nodes)) + self.bias


class RegionalGCN(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, dropout: float = 0.4) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                DenseGraphConvolution(input_dim, hidden_dim),
                DenseGraphConvolution(hidden_dim, hidden_dim),
                DenseGraphConvolution(hidden_dim, hidden_dim),
            ]
        )
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in self.layers])
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, 32),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, nodes: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        values = nodes
        for layer, norm in zip(self.layers, self.norms, strict=True):
            values = self.dropout(torch.relu(norm(layer(values, adjacency))))
        pooled = torch.cat([values.mean(dim=1), values.amax(dim=1)], dim=1)
        return self.classifier(pooled).squeeze(1)


class TabularMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 96, dropout: float = 0.25) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, max(24, hidden_dim // 2)),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(dropout),
            nn.Linear(max(24, hidden_dim // 2), 1),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values).squeeze(1)


class SubtypeMixture(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 96,
        n_subtypes: int = 4,
        dropout: float = 0.25,
    ) -> None:
        super().__init__()
        self.shared = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(dropout),
        )
        self.experts = nn.Linear(hidden_dim, n_subtypes)
        self.gate = nn.Linear(hidden_dim, n_subtypes)

    def forward(self, values: torch.Tensor) -> dict[str, torch.Tensor]:
        hidden = self.shared(values)
        expert_logits = self.experts(hidden)
        weights = torch.softmax(self.gate(hidden), dim=1)
        logit = (weights * expert_logits).sum(dim=1)
        return {"logit": logit, "expert_logits": expert_logits, "weights": weights}
