from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from typing import Any

from modeling.dat_spect_v2.config import DataConfig as Node07DataConfig
from modeling.dat_spect_v2.config import stable_hash

MODEL_FAMILIES = (
    "dual_stream_multitask",
    "regional_hgb",
    "topology_hgb",
    "graph_elasticnet",
    "graph_random_forest",
    "graph_mlp",
    "graph_gcn",
    "diffusion_map",
    "subtype_mixture",
)


@dataclass(frozen=True)
class DataConfig:
    node4_profile: str = "v3"
    preprocessing_version: str = "node10_dual_registered_counts_selfnorm_v2"
    output_spacing_mm: float = 2.0
    output_shape_zyx: tuple[int, int, int] = (40, 72, 72)
    striatal_roi_radii_zyx_mm: tuple[float, float, float] = (18.0, 34.0, 44.0)
    striatal_roi_hot_percentile: float = 85.0
    striatal_roi_recenter_max_mm: float = 10.0
    cache_dtype: str = "float16"
    cache_checkpoint_every: int = 25
    topology_percentiles: tuple[float, ...] = (50.0, 60.0, 70.0, 80.0, 85.0, 90.0, 95.0)
    graph_z_bands: int = 2

    def __post_init__(self) -> None:
        if self.output_spacing_mm <= 0 or any(value <= 0 for value in self.output_shape_zyx):
            raise ValueError("Spacing y forma de salida deben ser positivos.")
        if self.graph_z_bands < 2 or self.graph_z_bands > self.output_shape_zyx[0]:
            raise ValueError("graph_z_bands debe estar entre 2 y la profundidad de salida.")
        if self.cache_dtype not in {"float16", "float32"}:
            raise ValueError("cache_dtype debe ser float16 o float32.")

    def as_node07(self) -> Node07DataConfig:
        return Node07DataConfig(
            node4_profile=self.node4_profile,
            output_spacing_mm=self.output_spacing_mm,
            output_shape_zyx=self.output_shape_zyx,
            striatal_roi_radii_zyx_mm=self.striatal_roi_radii_zyx_mm,
            striatal_roi_hot_percentile=self.striatal_roi_hot_percentile,
            striatal_roi_recenter_max_mm=self.striatal_roi_recenter_max_mm,
            cache_dtype=self.cache_dtype,
            cache_checkpoint_every=self.cache_checkpoint_every,
        )


@dataclass(frozen=True)
class TrainConfig:
    max_epochs_search: int = 16
    max_epochs_final: int = 24
    patience: int = 5
    batch_size_3d: int = 4
    batch_size_graph: int = 32
    learning_rate: float = 2e-4
    weight_decay: float = 3e-5
    dropout: float = 0.25
    base_channels: int = 12
    embedding_dim: int = 96
    auxiliary_weight: float = 0.15
    consistency_weight: float = 0.03
    augmentation_probability: float = 0.90
    augmentation_magnitude: float = 2.5
    blur_base_fwhm_mm: tuple[float, float] = (1.0, 6.0)
    correlated_noise_fwhm_mm: tuple[float, float] = (4.0, 12.0)
    correlated_noise_sd_fraction: tuple[float, float] = (0.02, 0.08)
    rotation_degrees: float = 2.0
    translation_fraction: float = 0.025
    intensity_gain_range: tuple[float, float] = (0.90, 1.10)
    gradient_clip_norm: float = 2.0
    num_workers: int = field(default_factory=lambda: max(1, min(4, (os.cpu_count() or 2) - 1)))
    amp: bool = True
    seed: int = 20260831

    def __post_init__(self) -> None:
        if min(self.max_epochs_search, self.max_epochs_final, self.patience) < 1:
            raise ValueError("Epocas y paciencia deben ser positivas.")
        if min(self.batch_size_3d, self.batch_size_graph) < 1:
            raise ValueError("Los batch sizes deben ser positivos.")
        if not 0 <= self.augmentation_probability <= 1:
            raise ValueError("augmentation_probability debe pertenecer a [0, 1].")
        if self.augmentation_magnitude < 0:
            raise ValueError("augmentation_magnitude no puede ser negativa.")


@dataclass(frozen=True)
class SearchConfig:
    n_splits_search: int = 3
    n_splits_final: int = 5
    completed_trials_per_model: int = 10
    pruning_startup_trials: int = 5
    pruning_warmup_folds: int = 1
    optuna_timeout_seconds: int | None = None
    stale_trial_grace_seconds: int = 900
    fold_candidates: int = 256
    stack_completed_trials: int = 10

    def __post_init__(self) -> None:
        if min(self.n_splits_search, self.n_splits_final) < 2:
            raise ValueError("Cross-validation requiere al menos dos folds.")
        if min(self.completed_trials_per_model, self.stack_completed_trials) < 1:
            raise ValueError("Los presupuestos de Optuna deben ser positivos.")

    @property
    def effective_completed_trials(self) -> int:
        return max(10, int(self.completed_trials_per_model))

    @property
    def effective_stack_trials(self) -> int:
        return max(10, int(self.stack_completed_trials))


@dataclass(frozen=True)
class ExperimentConfig:
    run_id: str = "node10_hybrid_v1"
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    model_families: tuple[str, ...] = MODEL_FAMILIES
    upstream_contract: str = "node04_registered_intensity01_dual_view_fold_safe_hybrid_v1"

    def __post_init__(self) -> None:
        unknown = sorted(set(self.model_families) - set(MODEL_FAMILIES))
        if unknown:
            raise ValueError(f"Familias desconocidas en nodo 10: {unknown}")
        if len(set(self.model_families)) != len(self.model_families):
            raise ValueError("model_families contiene duplicados.")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def config_hash(self) -> str:
        payload = self.to_dict()
        # Los presupuestos son extensibles: subir el objetivo reanuda el mismo estudio.
        for key in (
            "completed_trials_per_model",
            "stack_completed_trials",
            "optuna_timeout_seconds",
            "stale_trial_grace_seconds",
        ):
            payload["search"].pop(key, None)
        return stable_hash(payload)
