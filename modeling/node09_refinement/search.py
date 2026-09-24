from __future__ import annotations

import json
import time
import warnings
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import pandas as pd
from optuna.exceptions import ExperimentalWarning
from optuna.trial import TrialState

from modeling.cnn.io import atomic_csv, atomic_json
from modeling.dat_spect_v2.config import (
    AugmentationConfig,
    ModelConfig,
    TrainConfig,
    stable_hash,
)
from modeling.dat_spect_v2.training import train_one_fold

from .config import RefinementExperiment, RefinementSearchConfig


@dataclass
class RefinementSearchResult:
    summary: pd.DataFrame
    finalists: list[dict[str, Any]]
    database_path: Path


def _storage(path: Path, config: RefinementSearchConfig) -> optuna.storages.RDBStorage:
    path.parent.mkdir(parents=True, exist_ok=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ExperimentalWarning)
        retry = optuna.storages.RetryHeartbeatStaleTrialCallback(max_retry=2)
        return optuna.storages.RDBStorage(
            f"sqlite:///{path.resolve().as_posix()}",
            engine_kwargs={"connect_args": {"timeout": 60}},
            heartbeat_interval=60,
            grace_period=config.stale_trial_grace_seconds,
            heartbeat_stale_trial_callback=retry,
        )


def _completed(study: optuna.Study) -> list[optuna.trial.FrozenTrial]:
    return [trial for trial in study.trials if trial.state == TrialState.COMPLETE]


def _numeric_center(reference: dict[str, Any], name: str, fallback: float) -> float:
    context = reference.get("trajectory_context", {})
    value = context.get("parameter_centers", {}).get(name, fallback)
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return float(fallback)
    return numeric if np.isfinite(numeric) else float(fallback)


def refinement_space(reference: dict[str, Any]) -> dict[str, Any]:
    """Return deterministic local bounds derived from the source trajectory."""

    source_model = dict(reference["model_config"])
    source_train = dict(reference["train_config"])
    source_augmentation = dict(source_train["augmentation"])
    source_lr = float(source_train["learning_rate"])
    trajectory_lr = _numeric_center(
        reference, "fusion_learning_rate", source_lr
    )
    lr_center = min(source_lr, trajectory_lr)
    lr_low = max(1.5e-5, min(source_lr * 0.22, lr_center * 0.35))
    lr_high = min(2.5e-4, source_lr * 0.78, max(lr_center * 0.95, lr_low * 1.8))
    if lr_high <= lr_low:
        lr_low = max(1e-5, lr_high / 3.0)

    source_decay = float(source_train["weight_decay"])
    trajectory_decay = _numeric_center(
        reference, "fusion_weight_decay", source_decay
    )
    decay_center = max(1e-7, float(np.sqrt(source_decay * trajectory_decay)))
    decay_low = max(1e-7, decay_center / 5.0)
    decay_high = min(2e-3, max(decay_center * 12.0, decay_low * 10.0))

    source_dropout = float(source_model["dropout"])
    dropout_center = _numeric_center(
        reference, "fusion_dropout", source_dropout
    )
    dropout_center = float(np.clip((source_dropout + dropout_center) / 2.0, 0.10, 0.45))

    tabular_dim = int(source_model["tabular_embedding_dim"])
    tabular_choices = sorted(
        {max(16, tabular_dim // 2), tabular_dim, min(64, tabular_dim * 2)}
    )
    top_k = int(reference["feature_top_k"])
    top_k_choices = sorted({max(32, top_k // 2), top_k, min(256, top_k * 2)})
    pca_source = reference.get("pca_variance")
    pca_choices = ["none", "0.95", "0.98"]
    if pca_source is not None:
        pca_choices.append(str(float(pca_source)))
    return {
        "learning_rate": (float(lr_low), float(lr_high)),
        "weight_decay": (float(decay_low), float(decay_high)),
        "dropout": (
            float(max(0.10, dropout_center - 0.075)),
            float(min(0.45, dropout_center + 0.075)),
        ),
        "auxiliary_weight": (
            max(0.01, float(source_train["auxiliary_weight"]) - 0.05),
            min(0.25, float(source_train["auxiliary_weight"]) + 0.05),
        ),
        "consistency_weight": (
            max(0.0, float(source_train["consistency_weight"]) - 0.035),
            min(0.12, float(source_train["consistency_weight"]) + 0.035),
        ),
        "augmentation_magnitude": (
            max(1.6, float(source_augmentation["magnitude"]) - 0.30),
            min(3.2, float(source_augmentation["magnitude"]) + 0.30),
        ),
        "augmentation_probability": (
            max(0.70, float(source_augmentation["probability"]) - 0.07),
            min(0.99, float(source_augmentation["probability"]) + 0.07),
        ),
        "rotation_degrees": (
            max(0.0, float(source_augmentation["rotation_degrees"]) - 0.75),
            min(3.5, float(source_augmentation["rotation_degrees"]) + 0.50),
        ),
        "tabular_embedding_dim": tabular_choices,
        "feature_top_k": top_k_choices,
        "pca_variance": sorted(set(pca_choices)),
    }


def _suggest(
    trial: optuna.Trial,
    reference: dict[str, Any],
    experiment: RefinementExperiment,
) -> tuple[ModelConfig, TrainConfig, int, float | None]:
    space = refinement_space(reference)
    model = ModelConfig(**reference["model_config"])
    train_payload = dict(reference["train_config"])
    train_payload["augmentation"] = AugmentationConfig(**train_payload["augmentation"])
    train = TrainConfig(**train_payload)
    dropout = trial.suggest_float("dropout", *space["dropout"])
    tabular_dim = trial.suggest_categorical(
        "tabular_embedding_dim", space["tabular_embedding_dim"]
    )
    model = replace(model, dropout=dropout, tabular_embedding_dim=int(tabular_dim))
    augmentation = replace(
        train.augmentation,
        magnitude=trial.suggest_float(
            "augmentation_magnitude", *space["augmentation_magnitude"]
        ),
        probability=trial.suggest_float(
            "augmentation_probability", *space["augmentation_probability"]
        ),
        rotation_degrees=trial.suggest_float(
            "rotation_degrees", *space["rotation_degrees"]
        ),
    )
    train = replace(
        train,
        learning_rate=trial.suggest_float(
            "learning_rate", *space["learning_rate"], log=True
        ),
        weight_decay=trial.suggest_float(
            "weight_decay", *space["weight_decay"], log=True
        ),
        auxiliary_weight=trial.suggest_float(
            "auxiliary_weight", *space["auxiliary_weight"]
        ),
        consistency_weight=trial.suggest_float(
            "consistency_weight", *space["consistency_weight"]
        ),
        max_epochs_search=experiment.search.max_epochs_search,
        patience=experiment.search.patience,
        seed=experiment.seed,
        augmentation=augmentation,
    )
    top_k = int(trial.suggest_categorical("feature_top_k", space["feature_top_k"]))
    pca_choice = trial.suggest_categorical("pca_variance", space["pca_variance"])
    pca_variance = None if pca_choice == "none" else float(pca_choice)
    return model, train, top_k, pca_variance


def _anchor_parameters(reference: dict[str, Any]) -> dict[str, Any]:
    space = refinement_space(reference)
    source_train = reference["train_config"]
    source_aug = source_train["augmentation"]

    def midpoint(name: str, *, geometric: bool = False) -> float:
        low, high = space[name]
        return float(np.sqrt(low * high) if geometric else (low + high) / 2.0)

    return {
        "dropout": float(np.clip(reference["model_config"]["dropout"], *space["dropout"])),
        "tabular_embedding_dim": int(reference["model_config"]["tabular_embedding_dim"]),
        "augmentation_magnitude": float(
            np.clip(source_aug["magnitude"], *space["augmentation_magnitude"])
        ),
        "augmentation_probability": float(
            np.clip(source_aug["probability"], *space["augmentation_probability"])
        ),
        "rotation_degrees": float(
            np.clip(source_aug["rotation_degrees"], *space["rotation_degrees"])
        ),
        "learning_rate": midpoint("learning_rate", geometric=True),
        "weight_decay": midpoint("weight_decay", geometric=True),
        "auxiliary_weight": float(
            np.clip(source_train["auxiliary_weight"], *space["auxiliary_weight"])
        ),
        "consistency_weight": float(
            np.clip(source_train["consistency_weight"], *space["consistency_weight"])
        ),
        "feature_top_k": int(reference["feature_top_k"]),
        "pca_variance": (
            "none" if reference.get("pca_variance") is None else str(reference["pca_variance"])
        ),
    }


def _optimize_to_target(
    study: optuna.Study,
    objective: Any,
    config: RefinementSearchConfig,
) -> None:
    started = time.monotonic()
    attempts = 0
    limit = max(
        config.trials_per_model * config.maximum_attempt_multiplier,
        config.trials_per_model + 6,
    )
    while len(_completed(study)) < config.trials_per_model and attempts < limit:
        if (
            config.optuna_timeout_seconds is not None
            and time.monotonic() - started >= config.optuna_timeout_seconds
        ):
            break
        study.optimize(objective, n_trials=1, gc_after_trial=True, show_progress_bar=False)
        attempts += 1


def _trial_table(study: optuna.Study) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "trial": trial.number,
                "state": trial.state.name,
                "value": trial.value,
                "duration_seconds": trial.duration.total_seconds() if trial.duration else np.nan,
                "parameters": json.dumps(trial.params, sort_keys=True),
                "candidate": json.dumps(trial.user_attrs.get("candidate", {}), sort_keys=True),
            }
            for trial in study.trials
        ]
    )


def _candidate(trial: optuna.trial.FrozenTrial, study_name: str) -> dict[str, Any]:
    candidate = dict(trial.user_attrs["candidate"])
    candidate.update(
        {
            "study_name": study_name,
            "trial_number": int(trial.number),
            "search_log_loss": float(trial.value),
        }
    )
    candidate["candidate_id"] = stable_hash(candidate)[:16]
    return candidate


def run_refinement_search(
    cohort: pd.DataFrame,
    folds: pd.DataFrame,
    references: list[dict[str, Any]],
    *,
    experiment: RefinementExperiment,
    run_dir: Path,
    cache_root: Path,
    upstream_hash: str,
    device: str | None = None,
) -> RefinementSearchResult:
    if len(references) != 3:
        raise ValueError("El nodo 09 exige exactamente tres referencias unicas.")
    source_ids = [str(reference["source_candidate_id"]) for reference in references]
    if len(set(source_ids)) != 3:
        raise ValueError("Las tres referencias del nodo 09 deben ser distintas.")
    run_dir.mkdir(parents=True, exist_ok=True)
    database = run_dir / "optuna_node09.sqlite3"
    storage = _storage(database, experiment.search)
    trials_dir = run_dir / "trials"
    studies_dir = run_dir / "studies"
    studies_dir.mkdir(parents=True, exist_ok=True)
    winners: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for branch, reference in enumerate(references):
        source_id = str(reference["source_candidate_id"])
        architecture = str(reference["architecture"])
        study_name = f"node09_{architecture.replace('.', '')}_{source_id[:10]}"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ExperimentalWarning)
            sampler = optuna.samplers.TPESampler(
                seed=experiment.seed + branch,
                n_startup_trials=min(8, experiment.search.trials_per_model),
                multivariate=True,
                constant_liar=True,
            )
        pruner = optuna.pruners.MedianPruner(
            n_startup_trials=experiment.search.pruning_startup_trials,
            n_warmup_steps=experiment.search.pruning_warmup_folds,
        )
        study = optuna.create_study(
            study_name=study_name,
            storage=storage,
            load_if_exists=True,
            direction="minimize",
            sampler=sampler,
            pruner=pruner,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ExperimentalWarning)
            optuna.storages.fail_stale_trials(study)
        if not study.trials:
            study.enqueue_trial(_anchor_parameters(reference))

        def objective(
            trial: optuna.Trial,
            reference: dict[str, Any] = reference,
            study_name: str = study_name,
        ) -> float:
            model, train, top_k, pca_variance = _suggest(trial, reference, experiment)
            payload = {
                "source_candidate_id": str(reference["source_candidate_id"]),
                "source_node07_rank": int(reference["node07_rank"]),
                "source_node07_calibrated_log_loss": float(
                    reference["node07_calibrated_log_loss"]
                ),
                "selection_roles": list(reference["selection_roles"]),
                "architecture": model.architecture,
                "feature_variant": model.feature_variant,
                "lateral_strategy": model.lateral_strategy,
                "model_config": asdict(model),
                "train_config": asdict(train),
                "feature_top_k": top_k,
                "pca_variance": pca_variance,
            }
            parameter_hash = stable_hash(payload)[:20]
            scores: list[float] = []
            selected_epochs: list[int] = []
            for fold in range(experiment.search.n_splits_search):
                result = train_one_fold(
                    cohort,
                    folds,
                    fold=fold,
                    model_config=model,
                    data_config=experiment.data,
                    train_config=train,
                    cache_root=cache_root / "search",
                    output_dir=trials_dir / study_name / parameter_hash / f"fold_{fold}",
                    upstream_hash=upstream_hash,
                    feature_top_k=top_k,
                    pca_variance=pca_variance,
                    device=device,
                )
                scores.append(float(result.validation_log_loss))
                selected_epochs.append(int(result.selected_epoch) + 1)
                trial.report(float(np.mean(scores)), step=fold)
                if trial.should_prune():
                    raise optuna.TrialPruned()
            fixed_epochs = round(float(np.median(selected_epochs)))
            payload["fold_log_loss"] = scores
            payload["fixed_epochs"] = int(
                np.clip(
                    fixed_epochs,
                    experiment.search.minimum_fixed_epochs,
                    experiment.search.maximum_fixed_epochs,
                )
            )
            trial.set_user_attr("candidate", payload)
            return float(np.mean(scores))

        _optimize_to_target(study, objective, experiment.search)
        atomic_csv(_trial_table(study), studies_dir / f"{study_name}_trials.csv")
        complete = _completed(study)
        if not complete:
            raise RuntimeError(f"El estudio {study_name} no completo ningun trial.")
        winner = _candidate(min(complete, key=lambda item: float(item.value)), study_name)
        atomic_json(winner, studies_dir / f"{study_name}_best.json")
        winners.append(winner)
        rows.append(
            {
                "study_name": study_name,
                "source_candidate_id": source_id,
                "source_node07_rank": reference["node07_rank"],
                "selection_roles": ",".join(reference["selection_roles"]),
                "architecture": winner["architecture"],
                "feature_variant": winner["feature_variant"],
                "lateral_strategy": winner["lateral_strategy"],
                "completed_trials": len(complete),
                "total_trials": len(study.trials),
                "search_log_loss": winner["search_log_loss"],
                "fixed_epochs": winner["fixed_epochs"],
                "candidate_id": winner["candidate_id"],
            }
        )
    summary = pd.DataFrame(rows).sort_values("search_log_loss").reset_index(drop=True)
    atomic_csv(summary, run_dir / "search_summary.csv")
    atomic_json({"finalists": winners}, run_dir / "finalists.json")
    return RefinementSearchResult(summary, winners, database)
