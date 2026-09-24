from __future__ import annotations

import json
import time
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import pandas as pd
from optuna.exceptions import ExperimentalWarning
from optuna.trial import TrialState

from modeling.cnn.config import ModelConfig as CNNModelConfig
from modeling.cnn.config import TrainConfig as CNNTrainConfig
from modeling.cnn.io import atomic_csv, atomic_json
from modeling.cnn.training import train_one_fold
from modeling.dat_spect_v2.config import stable_hash

from .config import ExperimentConfig, SearchConfig
from .hgb import train_hgb_fold


@dataclass
class SearchResult:
    summary: pd.DataFrame
    finalists: dict[str, dict[str, Any]]
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


def completed_trials(study: optuna.Study) -> list[optuna.trial.FrozenTrial]:
    return [trial for trial in study.trials if trial.state == TrialState.COMPLETE]


def _enqueue_failed_parameters_for_resume(study: optuna.Study) -> None:
    """Retry interrupted parameter sets once without duplicating queued work.

    A notebook ``KeyboardInterrupt`` marks the active Optuna trial as FAIL even
    though its fold checkpoint remains valid.  Node 11 keys fold artifacts by
    parameter hash, so re-enqueuing the same parameters resumes that checkpoint
    instead of discarding hours of CNN training.
    """

    def parameter_payload(trial: optuna.trial.FrozenTrial) -> dict[str, Any]:
        if trial.params:
            return dict(trial.params)
        fixed = trial.system_attrs.get("fixed_params", {})
        return dict(fixed) if isinstance(fixed, dict) else {}

    trials = list(study.trials)
    active_hashes = {
        stable_hash(parameter_payload(trial))
        for trial in trials
        if parameter_payload(trial)
        and trial.state
        in {TrialState.COMPLETE, TrialState.PRUNED, TrialState.RUNNING, TrialState.WAITING}
    }
    queued_hashes: set[str] = set()
    for trial in trials:
        parameters = parameter_payload(trial)
        if trial.state != TrialState.FAIL or not parameters:
            continue
        parameter_hash = stable_hash(parameters)
        if parameter_hash in active_hashes or parameter_hash in queued_hashes:
            continue
        study.enqueue_trial(parameters, user_attrs={"resumed_failed_trial": trial.number})
        queued_hashes.add(parameter_hash)


def optimize_until_complete(
    study: optuna.Study,
    objective: Callable[[optuna.Trial], float],
    *,
    target: int,
    timeout_seconds: int | None,
) -> None:
    target = max(10, int(target))
    started = time.monotonic()
    while len(completed_trials(study)) < target:
        if timeout_seconds is not None and time.monotonic() - started >= timeout_seconds:
            raise RuntimeError(
                f"{study.study_name}: timeout antes de alcanzar {target} trials COMPLETE. "
                "Reanuda con el mismo run_id; los trials y checkpoints quedaron persistidos."
            )
        study.optimize(objective, n_trials=1, gc_after_trial=True, show_progress_bar=False)


def cnn_search_space(trial: optuna.Trial) -> dict[str, Any]:
    """Local, conservative search around the complementary node-06 expert."""

    return {
        "base_channels": trial.suggest_categorical("base_channels", [12, 16]),
        "image_embedding_dim": trial.suggest_categorical(
            "image_embedding_dim", [96, 128]
        ),
        "pooling": trial.suggest_categorical("pooling", ["gap_max", "gap_gem"]),
        "dropout": trial.suggest_float("dropout", 0.28, 0.44),
        "learning_rate": trial.suggest_float("learning_rate", 4e-5, 1.8e-4, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 2e-5, 3e-3, log=True),
        "consistency_weight": trial.suggest_float("consistency_weight", 0.0, 0.04),
        "noise_std": trial.suggest_float("noise_std", 0.012, 0.032),
        "rotation_degrees": trial.suggest_float("rotation_degrees", 3.5, 6.5),
    }


def hgb_search_space(trial: optuna.Trial) -> dict[str, Any]:
    """Local search around node10 regional_hgb trial 7."""

    return {
        "learning_rate": trial.suggest_float("learning_rate", 0.018, 0.055, log=True),
        "max_leaf_nodes": trial.suggest_categorical("max_leaf_nodes", [8, 12, 16, 20, 24]),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 30, 64),
        "l2_regularization": trial.suggest_float(
            "l2_regularization", 1e-5, 0.15, log=True
        ),
    }


def cnn_configs(
    parameters: dict[str, Any], experiment: ExperimentConfig
) -> tuple[CNNModelConfig, CNNTrainConfig]:
    model = CNNModelConfig(
        architecture="2.5d",
        feature_variant="image_only",
        slices_per_view=experiment.cnn_data.slices_per_view,
        base_channels=int(parameters["base_channels"]),
        image_embedding_dim=int(parameters["image_embedding_dim"]),
        dropout=float(parameters["dropout"]),
        pooling=str(parameters["pooling"]),  # type: ignore[arg-type]
    )
    train = CNNTrainConfig(
        learning_rate=float(parameters["learning_rate"]),
        weight_decay=float(parameters["weight_decay"]),
        batch_size_25d=experiment.train.cnn_batch_size,
        max_epochs_search=experiment.train.max_epochs,
        patience=experiment.train.cnn_patience,
        consistency_weight=float(parameters["consistency_weight"]),
        num_workers=experiment.train.num_workers,
        amp=experiment.train.amp,
        seed=experiment.train.seed,
        rotation_degrees=float(parameters["rotation_degrees"]),
        noise_std=float(parameters["noise_std"]),
    )
    return model, train


def _trial_table(study: optuna.Study) -> pd.DataFrame:
    parameter_names = sorted({name for trial in study.trials for name in trial.params})
    rows: list[dict[str, Any]] = []
    for trial in study.trials:
        row: dict[str, Any] = {
            "trial": int(trial.number),
            "state": trial.state.name,
            "objective": trial.value,
            "duration_seconds": (
                trial.duration.total_seconds() if trial.duration is not None else np.nan
            ),
        }
        row.update({f"param__{name}": trial.params.get(name) for name in parameter_names})
        row["fold_log_loss"] = json.dumps(trial.user_attrs.get("fold_log_loss", []))
        row["selected_epochs"] = json.dumps(trial.user_attrs.get("selected_epochs", []))
        rows.append(row)
    return pd.DataFrame(rows)


def _candidate(
    trial: optuna.trial.FrozenTrial, *, family: str, study_name: str
) -> dict[str, Any]:
    candidate = {
        "family": family,
        "study_name": study_name,
        "trial_number": int(trial.number),
        "search_objective": float(trial.value),
        "parameters": dict(trial.params),
        "fold_log_loss": list(trial.user_attrs.get("fold_log_loss", [])),
        "selected_epochs": list(trial.user_attrs.get("selected_epochs", [])),
    }
    candidate["candidate_id"] = stable_hash(candidate)[:16]
    return candidate


def run_search(
    cohort: pd.DataFrame,
    folds: pd.DataFrame,
    *,
    experiment: ExperimentConfig,
    run_dir: Path,
    device: str | None = None,
) -> SearchResult:
    run_dir.mkdir(parents=True, exist_ok=True)
    database = run_dir / "optuna_node11.sqlite3"
    storage = _storage(database, experiment.search)
    trial_root = run_dir / "trials"
    studies_root = run_dir / "studies"
    studies_root.mkdir(parents=True, exist_ok=True)
    target = experiment.search.effective_completed_trials
    finalists: dict[str, dict[str, Any]] = {}
    summary_rows: list[dict[str, Any]] = []

    study_specs = (
        (
            "regional_hgb",
            "node11_regional_hgb",
            hgb_search_space,
            {
                "learning_rate": 0.03341260125347943,
                "max_leaf_nodes": 16,
                "min_samples_leaf": 45,
                "l2_regularization": 0.00014495975114378547,
            },
        ),
        (
            "cnn_25d_image_only",
            "node11_cnn_25d_image_only",
            cnn_search_space,
            {
                "base_channels": 12,
                "image_embedding_dim": 128,
                "pooling": "gap_max",
                "dropout": 0.360124684611175,
                "learning_rate": experiment.train.cnn_learning_rate_anchor,
                "weight_decay": 0.0010914487203784716,
                "consistency_weight": 0.004944629447301126,
                "noise_std": 0.021731977854449854,
                "rotation_degrees": 5.743268338153312,
            },
        ),
    )
    for branch, (family, study_name, suggest, anchor) in enumerate(study_specs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ExperimentalWarning)
            sampler = optuna.samplers.TPESampler(
                seed=experiment.train.seed + branch * 10_007,
                n_startup_trials=min(experiment.search.pruning_startup_trials, target),
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
        _enqueue_failed_parameters_for_resume(study)
        if not study.trials:
            study.enqueue_trial(anchor)

        def objective(
            trial: optuna.Trial,
            family: str = family,
            suggest: Callable[[optuna.Trial], dict[str, Any]] = suggest,
            study_name: str = study_name,
        ) -> float:
            parameters = suggest(trial)
            parameter_hash = stable_hash(parameters)[:20]
            scores: list[float] = []
            selected_epochs: list[int] = []
            for fold in range(experiment.search.n_splits_search):
                output_dir = trial_root / study_name / parameter_hash / f"fold_{fold}"
                if family == "regional_hgb":
                    result = train_hgb_fold(
                        cohort,
                        folds,
                        fold=fold,
                        parameters=parameters,
                        train_config=experiment.train,
                        output_dir=output_dir,
                        seed_offset=100_000,
                    )
                    selected_epochs.append(int(result.best_iteration))
                else:
                    model, train = cnn_configs(parameters, experiment)
                    result = train_one_fold(
                        cohort,
                        folds,
                        fold=fold,
                        model_config=model,
                        data_config=experiment.cnn_data,
                        train_config=train,
                        output_dir=output_dir,
                        pca_variance=None,
                        fixed_epochs=None,
                        scheduler_epochs=experiment.train.max_epochs,
                        device=device,
                    )
                    selected_epochs.append(int(result.selected_epoch) + 1)
                scores.append(float(result.validation_log_loss))
                # Steps are one-based: warmup=2 means pruning is first allowed
                # only after two complete folds, never from a single lucky fold.
                trial.report(float(np.mean(scores)), step=fold + 1)
                if trial.should_prune():
                    raise optuna.TrialPruned()
            trial.set_user_attr("fold_log_loss", scores)
            trial.set_user_attr("selected_epochs", selected_epochs)
            trial.set_user_attr("mean_log_loss", float(np.mean(scores)))
            trial.set_user_attr("std_log_loss", float(np.std(scores, ddof=0)))
            return float(np.mean(scores))

        optimize_until_complete(
            study,
            objective,
            target=target,
            timeout_seconds=experiment.search.optuna_timeout_seconds,
        )
        trials = _trial_table(study)
        atomic_csv(trials, studies_root / f"{study_name}_trials.csv")
        complete = completed_trials(study)
        if len(complete) < target:
            raise RuntimeError(f"{study_name} no alcanzo {target} trials COMPLETE.")
        winner_trial = min(complete, key=lambda item: float(item.value))
        winner = _candidate(winner_trial, family=family, study_name=study_name)
        finalists[family] = winner
        atomic_json(winner, studies_root / f"{study_name}_best.json")
        summary_rows.append(
            {
                "study_name": study_name,
                "family": family,
                "completed_trials": len(complete),
                "total_trials": len(study.trials),
                "best_trial": int(winner_trial.number),
                "search_log_loss": float(winner_trial.value),
                "search_fold_std": float(
                    np.std(winner_trial.user_attrs.get("fold_log_loss", []), ddof=0)
                ),
                "median_selected_epoch_or_iteration": float(
                    np.median(winner_trial.user_attrs.get("selected_epochs", [np.nan]))
                ),
                "candidate_id": winner["candidate_id"],
            }
        )
    summary = pd.DataFrame(summary_rows).sort_values("search_log_loss").reset_index(drop=True)
    atomic_csv(summary, run_dir / "search_summary.csv")
    atomic_json({"finalists": finalists}, run_dir / "finalists.json")
    return SearchResult(summary, finalists, database)
