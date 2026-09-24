from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import pandas as pd
from optuna.trial import TrialState

from .config import (
    DataConfig,
    ModelConfig,
    SearchConfig,
    TrainConfig,
    stable_hash,
)
from .io import atomic_csv, atomic_json
from .training import train_one_fold


@dataclass
class StagedSearchResult:
    summary: pd.DataFrame
    finalists: list[dict[str, Any]]
    database_path: Path


def _sqlite_storage(database_path: Path, search_config: SearchConfig) -> optuna.storages.RDBStorage:
    database_path.parent.mkdir(parents=True, exist_ok=True)
    url = f"sqlite:///{database_path.resolve().as_posix()}"
    retry = optuna.storages.RetryHeartbeatStaleTrialCallback(max_retry=2)
    return optuna.storages.RDBStorage(
        url,
        engine_kwargs={"connect_args": {"timeout": 60}},
        heartbeat_interval=60,
        grace_period=search_config.stale_trial_grace_seconds,
        heartbeat_stale_trial_callback=retry,
    )


def _suggest_candidate(
    trial: optuna.Trial,
    *,
    architecture: str,
    variant: str,
    base_candidate: dict[str, Any] | None,
    base_train: TrainConfig,
) -> tuple[ModelConfig, TrainConfig, float | None]:
    if base_candidate is None:
        model = ModelConfig(
            architecture=architecture,  # type: ignore[arg-type]
            feature_variant=variant,  # type: ignore[arg-type]
            base_channels=trial.suggest_int("base_channels", 12, 20, step=4),
            pooling=trial.suggest_categorical("pooling", ["gap", "gap_max", "gap_gem"]),
            dropout=trial.suggest_float("dropout", 0.15, 0.40),
        )
        train = replace(
            base_train,
            learning_rate=trial.suggest_float("learning_rate", 8e-5, 8e-4, log=True),
            weight_decay=trial.suggest_float("weight_decay", 1e-6, 5e-3, log=True),
            consistency_weight=trial.suggest_float("consistency_weight", 0.0, 0.15),
            noise_std=trial.suggest_float("noise_std", 0.01, 0.04),
            rotation_degrees=trial.suggest_float("rotation_degrees", 4.0, 8.0),
        )
        pca_variance = None
    else:
        base_model = ModelConfig(**base_candidate["model_config"])
        base_training = TrainConfig(**base_candidate["train_config"])
        model = replace(
            base_model,
            feature_variant=variant,  # type: ignore[arg-type]
            tabular_embedding_dim=trial.suggest_categorical(
                "tabular_embedding_dim", [16, 24, 32, 48]
            ),
            dropout=trial.suggest_float("fusion_dropout", 0.15, 0.40),
        )
        if variant == "image_radiomics_sbr":
            model = replace(
                model,
                sbr_hidden_dim=trial.suggest_categorical("sbr_hidden_dim", [8, 16, 24]),
            )
        train = replace(
            base_training,
            learning_rate=trial.suggest_float("fusion_learning_rate", 5e-5, 5e-4, log=True),
            weight_decay=trial.suggest_float("fusion_weight_decay", 1e-6, 2e-3, log=True),
        )
        pca_choice = trial.suggest_categorical("pca", ["none", "0.90", "0.95"])
        pca_variance = None if pca_choice == "none" else float(pca_choice)
    return model, train, pca_variance


def _candidate_from_trial(trial: optuna.trial.FrozenTrial) -> dict[str, Any]:
    candidate = dict(trial.user_attrs["candidate"])
    candidate.update(
        {
            "trial_number": int(trial.number),
            "search_log_loss": float(trial.value),
            "candidate_id": stable_hash(
                {
                    "study": trial.study_id if hasattr(trial, "study_id") else "",
                    "candidate": trial.user_attrs["candidate"],
                }
            )[:16],
        }
    )
    return candidate


def _study_trials_frame(study: optuna.Study) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for trial in study.trials:
        records.append(
            {
                "number": trial.number,
                "state": trial.state.name,
                "value": trial.value,
                "duration_seconds": (
                    trial.duration.total_seconds() if trial.duration is not None else np.nan
                ),
                "params_json": json.dumps(trial.params, sort_keys=True),
                "candidate_json": json.dumps(trial.user_attrs.get("candidate", {}), sort_keys=True),
            }
        )
    return pd.DataFrame.from_records(records)


def _optimize_to_completed(
    study: optuna.Study,
    objective: Any,
    target_completed: int,
    timeout_seconds: int | None,
) -> None:
    started = time.monotonic()
    maximum_attempts = max(target_completed * 4, target_completed + 4)
    attempts = 0
    while sum(trial.state == TrialState.COMPLETE for trial in study.trials) < target_completed:
        if attempts >= maximum_attempts:
            break
        if timeout_seconds is not None and time.monotonic() - started >= timeout_seconds:
            break
        study.optimize(objective, n_trials=1, gc_after_trial=True, show_progress_bar=False)
        attempts += 1


def run_staged_search(
    cohort: pd.DataFrame,
    search_folds: pd.DataFrame,
    *,
    run_dir: Path,
    data_config: DataConfig,
    train_config: TrainConfig,
    search_config: SearchConfig,
    device: str | None = None,
) -> StagedSearchResult:
    run_dir.mkdir(parents=True, exist_ok=True)
    database_path = run_dir / "optuna_cnn.sqlite3"
    storage = _sqlite_storage(database_path, search_config)
    studies_dir = run_dir / "studies"
    trials_dir = run_dir / "trials"
    studies_dir.mkdir(parents=True, exist_ok=True)
    trials_dir.mkdir(parents=True, exist_ok=True)
    best_by_stage: dict[tuple[str, str], dict[str, Any]] = {}
    candidates: list[dict[str, Any]] = []
    stages = ["image_only", "image_radiomics", "image_radiomics_sbr"]
    for architecture in ("2.5d", "3d"):
        base_candidate: dict[str, Any] | None = None
        for variant in stages:
            study_name = f"cnn_{architecture.replace('.', '')}_{variant}"
            sampler = optuna.samplers.TPESampler(
                seed=train_config.seed,
                multivariate=True,
                constant_liar=True,
            )
            pruner = optuna.pruners.MedianPruner(
                n_startup_trials=search_config.pruning_startup_trials,
                n_warmup_steps=search_config.pruning_warmup_folds,
            )
            study = optuna.create_study(
                study_name=study_name,
                storage=storage,
                load_if_exists=True,
                direction="minimize",
                sampler=sampler,
                pruner=pruner,
            )
            optuna.storages.fail_stale_trials(study)

            def objective(
                trial: optuna.Trial,
                architecture: str = architecture,
                variant: str = variant,
                base_candidate: dict[str, Any] | None = base_candidate,
                study_name: str = study_name,
            ) -> float:
                model, training, pca_variance = _suggest_candidate(
                    trial,
                    architecture=architecture,
                    variant=variant,
                    base_candidate=base_candidate,
                    base_train=train_config,
                )
                candidate_payload = {
                    "architecture": architecture,
                    "feature_variant": variant,
                    "model_config": asdict(model),
                    "train_config": asdict(training),
                    "pca_variance": pca_variance,
                }
                parameter_hash = stable_hash(candidate_payload)[:20]
                scores: list[float] = []
                epochs: list[int] = []
                for fold in range(search_config.n_splits_search):
                    result = train_one_fold(
                        cohort,
                        search_folds,
                        fold=fold,
                        model_config=model,
                        data_config=data_config,
                        train_config=training,
                        output_dir=trials_dir / study_name / parameter_hash / f"fold_{fold}",
                        pca_variance=pca_variance,
                        fixed_epochs=None,
                        device=device,
                    )
                    scores.append(result.validation_log_loss)
                    epochs.append(result.selected_epoch + 1)
                    trial.report(float(np.mean(scores)), step=fold)
                    if trial.should_prune():
                        raise optuna.TrialPruned()
                candidate_payload["fixed_epochs"] = int(max(2, round(float(np.median(epochs)))))
                candidate_payload["fold_log_loss"] = scores
                trial.set_user_attr("candidate", candidate_payload)
                return float(np.mean(scores))

            trial_target = {
                "image_only": search_config.trials_image,
                "image_radiomics": search_config.trials_fusion,
                "image_radiomics_sbr": search_config.trials_sbr,
            }[variant]
            _optimize_to_completed(
                study,
                objective,
                trial_target,
                search_config.optuna_timeout_seconds,
            )
            atomic_csv(_study_trials_frame(study), studies_dir / f"{study_name}_trials.csv")
            complete_trials = [trial for trial in study.trials if trial.state == TrialState.COMPLETE]
            if not complete_trials:
                raise RuntimeError(f"El estudio {study_name} no completo ningun trial.")
            best_trial = min(complete_trials, key=lambda trial: float(trial.value))
            best_candidate = _candidate_from_trial(best_trial)
            best_candidate["study_name"] = study_name
            best_by_stage[(architecture, variant)] = best_candidate
            candidates.append(best_candidate)
            base_candidate = best_candidate
            atomic_json(best_candidate, studies_dir / f"{study_name}_best.json")

    ordered = sorted(candidates, key=lambda candidate: candidate["search_log_loss"])
    finalists: list[dict[str, Any]] = []
    if ordered:
        finalists.append(ordered[0])
    other_architecture = next(
        (
            candidate
            for candidate in ordered
            if finalists and candidate["architecture"] != finalists[0]["architecture"]
        ),
        None,
    )
    if other_architecture is not None:
        finalists.append(other_architecture)
    for candidate in ordered:
        if len(finalists) >= search_config.finalists:
            break
        if candidate in finalists:
            continue
        if candidate["feature_variant"] != finalists[0]["feature_variant"] or len(finalists) < 2:
            finalists.append(candidate)
    for candidate in ordered:
        if len(finalists) >= search_config.finalists:
            break
        if candidate not in finalists:
            finalists.append(candidate)
    summary = pd.DataFrame(
        [
            {
                "study_name": candidate["study_name"],
                "architecture": candidate["architecture"],
                "feature_variant": candidate["feature_variant"],
                "search_log_loss": candidate["search_log_loss"],
                "fixed_epochs": candidate["fixed_epochs"],
                "candidate_id": candidate["candidate_id"],
                "is_finalist": candidate in finalists,
            }
            for candidate in ordered
        ]
    )
    atomic_csv(summary, run_dir / "search_summary.csv")
    atomic_json({"finalists": finalists}, run_dir / "finalists.json")
    return StagedSearchResult(summary, finalists, database_path)
