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

from .config import (
    AugmentationConfig,
    ExperimentConfig,
    FeatureVariant,
    LateralStrategy,
    ModelConfig,
    SearchConfig,
    TrainConfig,
    stable_hash,
)
from .training import train_one_fold


@dataclass
class SearchResult:
    summary: pd.DataFrame
    finalists: list[dict[str, Any]]
    database_path: Path


def _storage(path: Path, config: SearchConfig) -> optuna.storages.RDBStorage:
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


def _suggest(
    trial: optuna.Trial,
    *,
    architecture: str,
    variant: FeatureVariant,
    lateral: LateralStrategy,
    experiment: ExperimentConfig,
    base_candidate: dict[str, Any] | None,
) -> tuple[ModelConfig, TrainConfig, int, float | None]:
    if base_candidate is None:
        pooling_choices = (
            ["gmp", "gap_max"]
            if architecture == "slab2d"
            else [
                "gap_max",
                "gap_gem",
                "attention",
            ]
        )
        base_channels = (
            16 if architecture == "slab2d" else trial.suggest_categorical("base_channels", [12, 16])
        )
        model = ModelConfig(
            architecture=architecture,  # type: ignore[arg-type]
            feature_variant=variant,
            lateral_strategy=lateral,
            registration_mode=experiment.primary_registration_mode,
            base_channels=base_channels,
            pooling=trial.suggest_categorical("pooling", pooling_choices),
            dropout=trial.suggest_float("dropout", 0.10, 0.35),
        )
        augmentation = replace(
            experiment.train.augmentation,
            magnitude=trial.suggest_float("augmentation_magnitude", 2.0, 3.0),
            probability=trial.suggest_float("augmentation_probability", 0.80, 0.97),
            rotation_degrees=trial.suggest_float("rotation_degrees", 0.0, 3.0),
            left_right_flip_probability=0.5 if lateral == "random_flip" else 0.0,
        )
        train = replace(
            experiment.train,
            learning_rate=trial.suggest_float("learning_rate", 8e-5, 6e-4, log=True),
            weight_decay=trial.suggest_float("weight_decay", 1e-6, 8e-4, log=True),
            auxiliary_weight=trial.suggest_float("auxiliary_weight", 0.03, 0.20),
            consistency_weight=trial.suggest_float("consistency_weight", 0.0, 0.08),
            augmentation=augmentation,
        )
        return model, train, 128, None
    model = ModelConfig(**base_candidate["model_config"])
    train_payload = dict(base_candidate["train_config"])
    train_payload["augmentation"] = AugmentationConfig(**train_payload["augmentation"])
    train = TrainConfig(**train_payload)
    model = replace(
        model,
        feature_variant=variant,
        tabular_embedding_dim=trial.suggest_categorical("tabular_embedding_dim", [24, 32, 48]),
        dropout=trial.suggest_float("fusion_dropout", 0.10, 0.35),
    )
    train = replace(
        train,
        learning_rate=trial.suggest_float("fusion_learning_rate", 5e-5, 4e-4, log=True),
        weight_decay=trial.suggest_float("fusion_weight_decay", 1e-6, 5e-4, log=True),
    )
    top_k = trial.suggest_categorical(
        "feature_top_k", list(experiment.search.feature_top_k_options)
    )
    pca_choice = trial.suggest_categorical(
        "pca_variance", ["none"] + [str(value) for value in experiment.search.pca_options if value]
    )
    pca_variance = None if pca_choice == "none" else float(pca_choice)
    return model, train, int(top_k), pca_variance


def _completed(study: optuna.Study) -> list[optuna.trial.FrozenTrial]:
    return [trial for trial in study.trials if trial.state == TrialState.COMPLETE]


def _optimize_to_target(
    study: optuna.Study,
    objective: Any,
    target: int,
    timeout_seconds: int | None,
) -> None:
    started = time.monotonic()
    attempts = 0
    while len(_completed(study)) < target and attempts < max(target * 4, target + 4):
        if timeout_seconds is not None and time.monotonic() - started >= timeout_seconds:
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
    result = dict(trial.user_attrs["candidate"])
    result.update(
        {
            "study_name": study_name,
            "trial_number": int(trial.number),
            "search_log_loss": float(trial.value),
        }
    )
    result["candidate_id"] = stable_hash(result)[:16]
    return result


def run_staged_search(
    cohort: pd.DataFrame,
    folds: pd.DataFrame,
    *,
    experiment: ExperimentConfig,
    run_dir: Path,
    cache_root: Path,
    upstream_hash: str,
    device: str | None = None,
) -> SearchResult:
    run_dir.mkdir(parents=True, exist_ok=True)
    database = run_dir / "optuna_node07.sqlite3"
    storage = _storage(database, experiment.search)
    candidates: list[dict[str, Any]] = []
    studies_dir = run_dir / "studies"
    trials_dir = run_dir / "trials"
    studies_dir.mkdir(parents=True, exist_ok=True)
    best_image: dict[tuple[str, str], dict[str, Any]] = {}
    for architecture in experiment.architectures:
        for lateral in experiment.lateral_strategies:
            base_candidate: dict[str, Any] | None = None
            for variant in experiment.feature_variants:
                if variant != "image_only" and base_candidate is None:
                    raise RuntimeError("La busqueda de fusion requiere primero image_only.")
                study_name = f"node07_{architecture.replace('.', '')}_{lateral}_{variant}"
                sampler = optuna.samplers.TPESampler(
                    seed=experiment.train.seed,
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

                def objective(
                    trial: optuna.Trial,
                    architecture: str = architecture,
                    lateral: LateralStrategy = lateral,
                    variant: FeatureVariant = variant,
                    base_candidate: dict[str, Any] | None = base_candidate,
                    study_name: str = study_name,
                ) -> float:
                    model, train, top_k, pca_variance = _suggest(
                        trial,
                        architecture=architecture,
                        variant=variant,
                        lateral=lateral,
                        experiment=experiment,
                        base_candidate=base_candidate,
                    )
                    payload = {
                        "architecture": architecture,
                        "feature_variant": variant,
                        "lateral_strategy": lateral,
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
                        scores.append(result.validation_log_loss)
                        selected_epochs.append(result.selected_epoch + 1)
                        trial.report(float(np.mean(scores)), step=fold)
                        if trial.should_prune():
                            raise optuna.TrialPruned()
                    payload["fold_log_loss"] = scores
                    payload["fixed_epochs"] = int(max(3, round(float(np.median(selected_epochs)))))
                    trial.set_user_attr("candidate", payload)
                    return float(np.mean(scores))

                target = experiment.search.trials_for(architecture, variant)
                _optimize_to_target(
                    study,
                    objective,
                    target,
                    experiment.search.optuna_timeout_seconds,
                )
                atomic_csv(_trial_table(study), studies_dir / f"{study_name}_trials.csv")
                complete = _completed(study)
                if not complete:
                    raise RuntimeError(f"El estudio {study_name} no completo trials.")
                best = _candidate(min(complete, key=lambda value: float(value.value)), study_name)
                atomic_json(best, studies_dir / f"{study_name}_best.json")
                candidates.append(best)
                base_candidate = best
                if variant == "image_only":
                    best_image[(architecture, lateral)] = best
    ordered = sorted(candidates, key=lambda value: value["search_log_loss"])
    finalists: list[dict[str, Any]] = []
    # Prefer the strongest slab, then architectural diversity, then best remaining score.
    for selector in (
        lambda item: item["architecture"] == "slab2d",
        lambda item: item["architecture"] == "3d",
        lambda item: item["architecture"] == "2.5d",
    ):
        match = next((item for item in ordered if selector(item)), None)
        if match is not None and match not in finalists:
            finalists.append(match)
        if len(finalists) >= experiment.search.finalists:
            break
    for item in ordered:
        if len(finalists) >= experiment.search.finalists:
            break
        if item not in finalists:
            finalists.append(item)
    summary = pd.DataFrame(
        [
            {
                "study_name": item["study_name"],
                "architecture": item["architecture"],
                "feature_variant": item["feature_variant"],
                "lateral_strategy": item["lateral_strategy"],
                "search_log_loss": item["search_log_loss"],
                "fixed_epochs": item["fixed_epochs"],
                "candidate_id": item["candidate_id"],
                "is_finalist": item in finalists,
            }
            for item in ordered
        ]
    )
    atomic_csv(summary, run_dir / "search_summary.csv")
    atomic_json({"finalists": finalists}, run_dir / "finalists.json")
    return SearchResult(summary, finalists, database)
