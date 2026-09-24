from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from modeling.dat_spect_v2.config import DataConfig, stable_hash


@dataclass(frozen=True)
class RefinementSearchConfig:
    """Budget for three local searches around the node-07 finalists."""

    n_splits_search: int = 3
    n_splits_final: int = 5
    trials_per_model: int = 18
    max_epochs_search: int = 32
    patience: int = 9
    minimum_fixed_epochs: int = 8
    maximum_fixed_epochs: int = 32
    optuna_timeout_seconds: int | None = None
    pruning_startup_trials: int = 8
    pruning_warmup_folds: int = 1
    stale_trial_grace_seconds: int = 900
    maximum_attempt_multiplier: int = 5


@dataclass(frozen=True)
class RefinementExperiment:
    run_id: str = "node09_fine_v1"
    source_node07_run_id: str = "dat_spect_slab_v4"
    data: DataConfig = field(default_factory=DataConfig)
    search: RefinementSearchConfig = field(default_factory=RefinementSearchConfig)
    seed: int = 20260830
    upstream_contract: str = "node07_cv5_top2_plus_best3d_refinement_v1"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def config_hash(self) -> str:
        payload = self.to_dict()
        # Increasing the requested number of completed trials resumes the same
        # scientific study rather than invalidating existing checkpoints.
        payload["search"].pop("trials_per_model", None)
        payload["search"].pop("optuna_timeout_seconds", None)
        payload["search"].pop("stale_trial_grace_seconds", None)
        payload["search"].pop("maximum_attempt_multiplier", None)
        return stable_hash(payload)

