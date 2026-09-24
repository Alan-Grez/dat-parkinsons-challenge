from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from typing import Any

from modeling.cnn.config import DataConfig as CNNDataConfig
from modeling.dat_spect_v2.config import stable_hash
from modeling.node10_hybrid.config import DataConfig as RegionalDataConfig


@dataclass(frozen=True)
class TrainConfig:
    """Long-horizon training contract for the two node-11 experts."""

    max_epochs: int = 500
    cnn_patience: int = 80
    cnn_learning_rate_anchor: float = 1.2e-4
    cnn_batch_size: int = 12
    hgb_max_iter: int = 500
    hgb_patience: int = 60
    hgb_validation_fraction: float = 0.15
    hgb_tolerance: float = 1e-5
    num_workers: int = field(
        default_factory=lambda: max(1, min(6, (os.cpu_count() or 2) - 1))
    )
    amp: bool = True
    seed: int = 20260902

    def __post_init__(self) -> None:
        if min(self.max_epochs, self.cnn_patience, self.hgb_max_iter, self.hgb_patience) < 1:
            raise ValueError("Los horizontes y paciencias del nodo 11 deben ser positivos.")
        if self.cnn_patience >= self.max_epochs:
            raise ValueError("cnn_patience debe ser menor que max_epochs.")
        if self.hgb_patience >= self.hgb_max_iter:
            raise ValueError("hgb_patience debe ser menor que hgb_max_iter.")
        if not 0.05 <= self.hgb_validation_fraction <= 0.30:
            raise ValueError("hgb_validation_fraction debe pertenecer a [0.05, 0.30].")


@dataclass(frozen=True)
class SearchConfig:
    n_splits_search: int = 3
    n_splits_final: int = 5
    inner_epoch_splits: int = 4
    inner_epoch_repeats: int = 2
    completed_trials_per_expert: int = 10
    pruning_startup_trials: int = 6
    pruning_warmup_folds: int = 2
    fold_candidates: int = 256
    optuna_timeout_seconds: int | None = None
    stale_trial_grace_seconds: int = 1800

    def __post_init__(self) -> None:
        if min(self.n_splits_search, self.n_splits_final, self.inner_epoch_splits) < 2:
            raise ValueError("Todos los esquemas de validacion requieren al menos dos folds.")
        if not 1 <= self.inner_epoch_repeats < self.inner_epoch_splits:
            raise ValueError("inner_epoch_repeats debe ser menor que inner_epoch_splits.")
        if self.completed_trials_per_expert < 1:
            raise ValueError("El presupuesto Optuna debe ser positivo.")

    @property
    def effective_completed_trials(self) -> int:
        # The prior node-10 contract established ten successful trials as the
        # minimum useful evidence. Pruned trials never count toward this target.
        return max(10, int(self.completed_trials_per_expert))


@dataclass(frozen=True)
class ExperimentConfig:
    run_id: str = "node11_best_shot_v1"
    source_node06_run_id: str = "cnn_compact_v1"
    source_node10_run_id: str = "node10_hybrid_v1"
    cnn_data: CNNDataConfig = field(default_factory=CNNDataConfig)
    regional_data: RegionalDataConfig = field(default_factory=RegionalDataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    regional_blend_weight: float = 0.50
    upstream_contract: str = (
        "node11_common_patient_folds_nested_epoch_selection_raw_blend_calibrate_after_v1"
    )

    def __post_init__(self) -> None:
        if not 0.0 <= self.regional_blend_weight <= 1.0:
            raise ValueError("regional_blend_weight debe pertenecer a [0, 1].")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def config_hash(self) -> str:
        payload = self.to_dict()
        # Budgets are operational and may be increased while resuming the same
        # scientific run. Search spaces, seeds, folds and training horizons stay hashed.
        for key in (
            "completed_trials_per_expert",
            "optuna_timeout_seconds",
            "stale_trial_grace_seconds",
        ):
            payload["search"].pop(key, None)
        return stable_hash(payload)
