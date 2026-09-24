from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

Architecture = Literal["slab2d", "2.5d", "3d"]
FeatureVariant = Literal[
    "image_only",
    "image_radiomics",
    "image_radiomics_sbr",
]
LateralStrategy = Literal["random_flip", "canonical"]
RegistrationMode = Literal["single_template", "fold_multitemplate_affine"]
PoolingMode = Literal["gmp", "gap_max", "gap_gem", "attention"]


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
    preprocessing_version: str = "physical_striatal_roi_fixed_midline_v2"
    node4_profile: str = "v3"
    output_spacing_mm: float = 2.0
    output_shape_zyx: tuple[int, int, int] = (40, 72, 72)
    slab_thickness_mm: float = 12.0
    striatal_roi_radii_zyx_mm: tuple[float, float, float] = (18.0, 34.0, 44.0)
    striatal_roi_hot_percentile: float = 85.0
    striatal_roi_recenter_max_mm: float = 10.0
    view_size_2d: int = 72
    slices_per_view: int = 5
    cache_dtype: str = "float16"
    cache_checkpoint_every: int = 25
    regional_feature_budget: int = 930
    regional_feature_schema: str = "regional930_selfnorm_v2"
    template_sample_per_prototype: int = 64
    template_softmax_temperature: float = 0.12
    affine_steps: int = 12
    affine_learning_rate: float = 0.035
    affine_max_linear_delta: float = 0.08
    affine_max_translation_fraction: float = 0.10

    @property
    def slab_slices(self) -> int:
        count = round(self.slab_thickness_mm / self.output_spacing_mm)
        return max(1, count)


@dataclass(frozen=True)
class AugmentationConfig:
    # Buddenkotte/Buchert winning cross-domain neighbourhood.
    magnitude: float = 2.5
    probability: float = 0.90
    blur_base_fwhm_mm: tuple[float, float] = (1.0, 6.0)
    correlated_noise_fwhm_mm: tuple[float, float] = (5.0, 15.0)
    correlated_noise_sd_fraction: tuple[float, float] = (0.10, 0.25)
    rotation_degrees: float = 2.0
    translation_fraction: float = 0.02
    intensity_gain_range: tuple[float, float] = (0.95, 1.05)
    left_right_flip_probability: float = 0.5


@dataclass(frozen=True)
class ModelConfig:
    architecture: Architecture = "slab2d"
    feature_variant: FeatureVariant = "image_only"
    lateral_strategy: LateralStrategy = "canonical"
    registration_mode: RegistrationMode = "fold_multitemplate_affine"
    in_channels: int = 1
    slices_per_view: int = 5
    base_channels: int = 16
    image_embedding_dim: int = 128
    tabular_embedding_dim: int = 32
    radiomics_dim: int = 0
    sbr_dim: int = 0
    dropout: float = 0.25
    pooling: PoolingMode = "gmp"
    auxiliary_dim: int = 4
    auxiliary_hidden_dim: int = 64


@dataclass(frozen=True)
class TrainConfig:
    learning_rate: float = 3e-4
    weight_decay: float = 3e-5
    batch_size_slab: int = 12
    batch_size_25d: int = 10
    batch_size_3d: int = 3
    max_epochs_search: int = 18
    patience: int = 5
    auxiliary_weight: float = 0.10
    consistency_weight: float = 0.04
    consistency_warmup_epochs: int = 3
    gradient_clip_norm: float = 2.0
    num_workers: int = field(default_factory=lambda: max(1, min(6, (os.cpu_count() or 2) - 1)))
    amp: bool = True
    seed: int = 20260828
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)


@dataclass(frozen=True)
class SearchConfig:
    n_splits_search: int = 3
    n_splits_final: int = 5
    # Slab receives the largest budget because it is the highest-priority new branch.
    trials_slab_image: int = 10
    trials_slab_fusion: int = 6
    trials_slab_sbr: int = 4
    trials_25d_image: int = 4
    trials_25d_fusion: int = 3
    trials_25d_sbr: int = 2
    trials_3d_image: int = 4
    trials_3d_fusion: int = 3
    trials_3d_sbr: int = 2
    finalists: int = 3
    fold_candidates: int = 256
    feature_top_k_options: tuple[int, ...] = (64, 128, 256)
    pca_options: tuple[float | None, ...] = (None, 0.95)
    optuna_timeout_seconds: int | None = None
    pruning_startup_trials: int = 3
    pruning_warmup_folds: int = 1
    stale_trial_grace_seconds: int = 600

    def trials_for(self, architecture: Architecture, variant: FeatureVariant) -> int:
        arch = "25d" if architecture == "2.5d" else architecture.replace("2d", "")
        suffix = {
            "image_only": "image",
            "image_radiomics": "fusion",
            "image_radiomics_sbr": "sbr",
        }[variant]
        return int(getattr(self, f"trials_{arch}_{suffix}"))


@dataclass(frozen=True)
class ExperimentConfig:
    run_id: str = "dat_spect_slab_v4"
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    architectures: tuple[Architecture, ...] = ("slab2d", "2.5d", "3d")
    feature_variants: tuple[FeatureVariant, ...] = (
        "image_only",
        "image_radiomics",
        "image_radiomics_sbr",
    )
    lateral_strategies: tuple[LateralStrategy, ...] = ("canonical", "random_flip")
    primary_registration_mode: RegistrationMode = "fold_multitemplate_affine"
    upstream_contract: str = "node4_registered_crops_then_fold_safe_multitemplate_v4"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def config_hash(self) -> str:
        payload = self.to_dict()
        # Increasing trial counts or timeouts continues the same scientific run.
        for key in list(payload["search"]):
            if key.startswith("trials_") or key in {
                "optuna_timeout_seconds",
                "stale_trial_grace_seconds",
            }:
                payload["search"].pop(key, None)
        return stable_hash(payload)
