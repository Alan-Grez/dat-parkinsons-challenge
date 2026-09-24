from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

Architecture = Literal["2.5d", "3d"]
FeatureVariant = Literal[
    "image_only",
    "image_radiomics",
    "image_radiomics_sbr",
]
PoolingMode = Literal["gap", "gap_max", "gap_gem"]


def stable_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class DataConfig:
    node4_profile: str = "v3"
    image_shape_3d: tuple[int, int, int] = (64, 80, 64)
    view_size_2d: int = 96
    slices_per_view: int = 5
    clip_percentiles: tuple[float, float] = (0.5, 99.5)
    scale_percentiles: tuple[float, float] = (5.0, 95.0)
    normalized_clip: tuple[float, float] = (-3.0, 3.0)
    high_uptake_percentile: float = 85.0
    texture_levels: int = 24
    cache_dtype: str = "float16"
    cache_checkpoint_every: int = 50
    feature_schema_version: str = "cnn_core_radiomics_v3_lps_shape"


@dataclass(frozen=True)
class ModelConfig:
    architecture: Architecture = "3d"
    feature_variant: FeatureVariant = "image_only"
    in_channels: int = 1
    slices_per_view: int = 5
    base_channels: int = 16
    image_embedding_dim: int = 128
    tabular_embedding_dim: int = 32
    radiomics_dim: int = 0
    sbr_dim: int = 0
    sbr_hidden_dim: int = 16
    dropout: float = 0.25
    pooling: PoolingMode = "gap_gem"
    gem_initial_p: float = 3.0
    gem_max_p: float = 6.0


@dataclass(frozen=True)
class TrainConfig:
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    batch_size_3d: int = 3
    batch_size_25d: int = 12
    max_epochs_search: int = 12
    patience: int = 4
    consistency_weight: float = 0.08
    consistency_warmup_epochs: int = 3
    gradient_clip_norm: float = 2.0
    num_workers: int = field(
        default_factory=lambda: max(1, min(6, (os.cpu_count() or 2) - 1))
    )
    amp: bool = True
    seed: int = 20260821
    rotation_degrees: float = 6.0
    translation_fraction: float = 0.045
    scale_range: tuple[float, float] = (0.96, 1.04)
    intensity_gain_range: tuple[float, float] = (0.85, 1.15)
    gamma_range: tuple[float, float] = (0.85, 1.15)
    noise_std: float = 0.025
    blur_probability: float = 0.15
    resolution_probability: float = 0.15
    left_right_flip_probability: float = 0.5


@dataclass(frozen=True)
class SearchConfig:
    n_splits_search: int = 3
    n_splits_final: int = 5
    trials_image: int = 20
    trials_fusion: int = 10
    trials_sbr: int = 8
    finalists: int = 3
    fold_candidates: int = 256
    optuna_timeout_seconds: int | None = None
    pruning_startup_trials: int = 4
    pruning_warmup_folds: int = 1
    stale_trial_grace_seconds: int = 300


@dataclass(frozen=True)
class ExperimentConfig:
    run_id: str = "cnn_compact_v1"
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    pca_options: tuple[float | None, ...] = (None, 0.90, 0.95)
    upstream_contract: str = "node4_registered_crops_global_template_exploratory"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def config_hash(self) -> str:
        payload = self.to_dict()
        # Trial targets and wall-clock limits are extensible execution budgets:
        # increasing them must continue the same Optuna studies, not force a
        # new scientific run. Search spaces, folds and pruning remain hashed.
        for key in (
            "trials_image",
            "trials_fusion",
            "trials_sbr",
            "optuna_timeout_seconds",
            "stale_trial_grace_seconds",
        ):
            payload["search"].pop(key, None)
        return stable_hash(payload)
