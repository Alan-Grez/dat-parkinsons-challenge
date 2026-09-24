"""Reanudable Optuna tuning for fold-safe DaT tabular experiments.

The notebook owns all medical-image preprocessing and passes matrices fitted
strictly within each validation fold.  This module only tunes classifiers,
persists Optuna journals, and checkpoints out-of-fold predictions.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import pandas as pd
from optuna.storages.journal import (
    JournalFileBackend,
    JournalFileOpenLock,
    JournalStorage,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from xgboost import XGBClassifier
from xgboost.core import XGBoostError

MODEL_NAMES = ("logistic", "random_forest", "xgboost")


@dataclass(frozen=True)
class FoldData:
    """One leakage-safe train/validation matrix pair."""

    fold: int
    train_indices: np.ndarray
    valid_indices: np.ndarray
    X_train: np.ndarray
    X_valid: np.ndarray
    y_train: np.ndarray
    y_valid: np.ndarray


@dataclass(frozen=True)
class OptunaExperimentResult:
    """Durable outputs returned to notebook 05."""

    tuning_summary: pd.DataFrame
    oof_predictions: pd.DataFrame
    metrics: pd.DataFrame
    best_parameters: dict[str, dict[str, Any]]
    xgboost_device: str


def _replace_with_retry(temporary: Path, destination: Path, attempts: int = 10) -> None:
    delay_seconds = 0.25
    for attempt in range(attempts):
        try:
            temporary.replace(destination)
            return
        except PermissionError:
            if attempt == attempts - 1:
                raise
            time.sleep(delay_seconds)
            delay_seconds = min(delay_seconds * 1.8, 3.0)


def _atomic_csv(frame: pd.DataFrame, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    _replace_with_retry(temporary, destination)


def _atomic_json(payload: dict[str, Any], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_json_default),
        encoding="utf-8",
    )
    _replace_with_retry(temporary, destination)


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Cannot serialize {type(value)!r}")


def _probe_xgboost_device(prefer_gpu: bool, random_seed: int) -> str:
    """Return the device XGBoost actually selected, not merely the requested one."""

    if not prefer_gpu:
        return "cpu"
    X_probe = np.asarray([[0.0], [1.0], [0.1], [0.9]], dtype=np.float32)
    y_probe = np.asarray([0, 1, 0, 1], dtype=np.int32)
    try:
        probe = XGBClassifier(
            n_estimators=2,
            max_depth=1,
            learning_rate=0.3,
            tree_method="hist",
            device="cuda",
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=random_seed,
            verbosity=0,
        )
        probe.fit(X_probe, y_probe)
        config = json.loads(probe.get_booster().save_config())
        actual_device = str(config["learner"]["generic_param"]["device"])
        return actual_device if actual_device.startswith("cuda") else "cpu"
    except (XGBoostError, KeyError, ValueError, RuntimeError):
        return "cpu"


def _suggest_parameters(trial: optuna.Trial, model_name: str) -> dict[str, Any]:
    if model_name == "logistic":
        return {
            "C": trial.suggest_float("C", 1e-3, 30.0, log=True),
            "l1_ratio": trial.suggest_float("l1_ratio", 0.0, 1.0),
            "class_weight": trial.suggest_categorical(
                "class_weight", [None, "balanced"]
            ),
            "tol": trial.suggest_float("tol", 1e-5, 1e-3, log=True),
        }
    if model_name == "random_forest":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 300, 1000, step=50),
            "criterion": trial.suggest_categorical("criterion", ["gini", "log_loss"]),
            "max_depth": trial.suggest_int("max_depth", 2, 14),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 40),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 25),
            "max_features": trial.suggest_float("max_features", 0.30, 1.0),
            "max_samples": trial.suggest_float("max_samples", 0.60, 1.0),
            "class_weight": trial.suggest_categorical(
                "class_weight", [None, "balanced", "balanced_subsample"]
            ),
        }
    if model_name == "xgboost":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 150, 1000, step=50),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.30, log=True),
            "max_depth": trial.suggest_int("max_depth", 2, 7),
            "min_child_weight": trial.suggest_float(
                "min_child_weight", 0.5, 20.0, log=True
            ),
            "subsample": trial.suggest_float("subsample", 0.60, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.60, 1.0),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-7, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 30.0, log=True),
        }
    raise ValueError(f"Unknown model: {model_name}")


def _build_estimator(
    model_name: str,
    parameters: dict[str, Any],
    *,
    random_seed: int,
    cpu_jobs: int,
    xgboost_device: str,
) -> Any:
    if model_name == "logistic":
        return LogisticRegression(
            **parameters,
            solver="saga",
            max_iter=5000,
            random_state=random_seed,
        )
    if model_name == "random_forest":
        return RandomForestClassifier(
            **parameters,
            bootstrap=True,
            random_state=random_seed,
            n_jobs=cpu_jobs,
        )
    if model_name == "xgboost":
        return XGBClassifier(
            **parameters,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            device=xgboost_device,
            random_state=random_seed,
            n_jobs=1 if xgboost_device.startswith("cuda") else cpu_jobs,
            verbosity=0,
        )
    raise ValueError(f"Unknown model: {model_name}")


def _mean_or_nan(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=float)
    return float(np.nanmean(array)) if np.isfinite(array).any() else float("nan")


def _study_for_model(
    model_name: str,
    folds: Sequence[FoldData],
    output_dir: Path,
    *,
    experiment_hash: str,
    target_complete_trials: int,
    random_seed: int,
    cpu_jobs: int,
    xgboost_device: str,
) -> tuple[optuna.Study, dict[str, Any]]:
    studies_dir = output_dir / "studies"
    studies_dir.mkdir(parents=True, exist_ok=True)
    journal_path = str(studies_dir / f"{model_name}.journal")
    storage = JournalStorage(
        JournalFileBackend(journal_path, lock_obj=JournalFileOpenLock(journal_path))
    )
    study_name = f"{model_name}-{experiment_hash[:16]}"
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        sampler=optuna.samplers.TPESampler(seed=random_seed),
        direction="minimize",
        load_if_exists=True,
    )
    previous_hash = study.user_attrs.get("experiment_hash")
    if previous_hash not in (None, experiment_hash):
        raise RuntimeError(
            f"Study {study_name!r} belongs to another experiment configuration."
        )
    study.set_user_attr("experiment_hash", experiment_hash)
    study.set_user_attr("model_name", model_name)
    study.set_user_attr("primary_metric", "mean_grouped_cv_log_loss")
    if model_name == "xgboost":
        study.set_user_attr("device", xgboost_device)

    progress_path = studies_dir / f"{model_name}_progress.json"

    def persist_progress(current_study: optuna.Study) -> None:
        completed_now = sum(
            trial.state == optuna.trial.TrialState.COMPLETE
            for trial in current_study.trials
        )
        _atomic_json(
            {
                "model": model_name,
                "completed_trials": completed_now,
                "target_complete_trials": target_complete_trials,
                "best_cv_log_loss": (
                    float(current_study.best_value) if completed_now else None
                ),
            },
            progress_path,
        )

    def objective(trial: optuna.Trial) -> float:
        parameters = _suggest_parameters(trial, model_name)
        fold_log_losses: list[float] = []
        fold_aucs: list[float] = []
        fold_briers: list[float] = []
        for fold_data in folds:
            estimator = _build_estimator(
                model_name,
                parameters,
                random_seed=random_seed + fold_data.fold,
                cpu_jobs=cpu_jobs,
                xgboost_device=xgboost_device,
            )
            estimator.fit(fold_data.X_train, fold_data.y_train)
            probabilities = np.clip(
                estimator.predict_proba(fold_data.X_valid)[:, 1], 1e-6, 1 - 1e-6
            )
            fold_log_losses.append(
                log_loss(fold_data.y_valid, probabilities, labels=[0, 1])
            )
            fold_briers.append(brier_score_loss(fold_data.y_valid, probabilities))
            fold_aucs.append(
                roc_auc_score(fold_data.y_valid, probabilities)
                if np.unique(fold_data.y_valid).size == 2
                else np.nan
            )
        trial.set_user_attr("fold_log_losses", fold_log_losses)
        trial.set_user_attr("mean_auc", _mean_or_nan(fold_aucs))
        trial.set_user_attr("mean_brier", _mean_or_nan(fold_briers))
        return float(np.mean(fold_log_losses))

    complete_states = (optuna.trial.TrialState.COMPLETE,)
    completed = sum(trial.state in complete_states for trial in study.trials)
    if completed < target_complete_trials:
        study.optimize(
            objective,
            n_trials=target_complete_trials - completed,
            n_jobs=1,
            gc_after_trial=True,
            show_progress_bar=True,
            callbacks=[lambda current_study, _trial: persist_progress(current_study)],
        )
    completed = sum(trial.state in complete_states for trial in study.trials)
    if completed < target_complete_trials:
        raise RuntimeError(
            f"{model_name}: only {completed}/{target_complete_trials} trials completed."
        )
    persist_progress(study)

    trials = study.trials_dataframe(attrs=("number", "value", "params", "state"))
    _atomic_csv(trials, studies_dir / f"{model_name}_trials.csv")
    best_parameters = dict(study.best_params)
    _atomic_json(
        {
            "model": model_name,
            "study_name": study.study_name,
            "experiment_hash": experiment_hash,
            "completed_trials": completed,
            "best_cv_log_loss": float(study.best_value),
            "best_parameters": best_parameters,
            "xgboost_device": xgboost_device if model_name == "xgboost" else None,
        },
        studies_dir / f"{model_name}_best.json",
    )
    return study, best_parameters


def _evaluate_oof(
    models: Sequence[str],
    best_parameters: dict[str, dict[str, Any]],
    folds: Sequence[FoldData],
    uids: Sequence[str],
    y: np.ndarray,
    groups: Sequence[str],
    output_dir: Path,
    *,
    experiment_hash: str,
    random_seed: int,
    cpu_jobs: int,
    xgboost_device: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    checkpoint_path = output_dir / "oof_fold_checkpoint.csv"
    columns = [
        "uid",
        "is_pathologic",
        "acquisition_family",
        "model",
        "fold",
        "p_oof",
        "experiment_hash",
    ]
    if checkpoint_path.exists():
        checkpoint = pd.read_csv(checkpoint_path, dtype={"uid": "string"})
        if set(columns) - set(checkpoint.columns):
            raise RuntimeError("The Optuna OOF checkpoint has an incompatible schema.")
        checkpoint = checkpoint[columns]
        incompatible = checkpoint["experiment_hash"].astype(str) != experiment_hash
        if incompatible.any():
            raise RuntimeError("The Optuna OOF checkpoint belongs to another configuration.")
    else:
        checkpoint = pd.DataFrame(columns=columns)

    uid_array = np.asarray([str(uid) for uid in uids])
    group_array = np.asarray([str(group) for group in groups])
    for model_name in models:
        for fold_data in folds:
            expected_uids = uid_array[fold_data.valid_indices]
            saved = checkpoint.loc[
                (checkpoint["model"] == model_name)
                & (checkpoint["fold"] == fold_data.fold)
            ]
            reusable = (
                len(saved) == len(expected_uids)
                and set(saved["uid"].astype(str)) == set(expected_uids)
                and np.isfinite(saved["p_oof"].to_numpy(dtype=float)).all()
            )
            if reusable:
                continue
            estimator = _build_estimator(
                model_name,
                best_parameters[model_name],
                random_seed=random_seed + fold_data.fold,
                cpu_jobs=cpu_jobs,
                xgboost_device=xgboost_device,
            )
            estimator.fit(fold_data.X_train, fold_data.y_train)
            probabilities = np.clip(
                estimator.predict_proba(fold_data.X_valid)[:, 1], 1e-6, 1 - 1e-6
            )
            fold_frame = pd.DataFrame(
                {
                    "uid": expected_uids,
                    "is_pathologic": y[fold_data.valid_indices],
                    "acquisition_family": group_array[fold_data.valid_indices],
                    "model": model_name,
                    "fold": fold_data.fold,
                    "p_oof": probabilities,
                    "experiment_hash": experiment_hash,
                }
            )
            checkpoint = checkpoint.loc[
                ~(
                    (checkpoint["model"] == model_name)
                    & (checkpoint["fold"] == fold_data.fold)
                )
            ]
            checkpoint = pd.concat([checkpoint, fold_frame], ignore_index=True)
            _atomic_csv(checkpoint, checkpoint_path)

    expected_rows = len(y) * len(models)
    if len(checkpoint) != expected_rows:
        raise RuntimeError(
            f"Incomplete Optuna OOF coverage: {len(checkpoint)}/{expected_rows} rows."
        )
    wide = checkpoint.pivot(index="uid", columns="model", values="p_oof")
    ordered_uids = pd.Index(uid_array, name="uid")
    wide = wide.reindex(ordered_uids)
    if wide.isna().any().any():
        raise RuntimeError("Optuna OOF predictions contain missing values.")
    wide.columns = [f"p_{column}_oof" for column in wide.columns]
    oof = pd.DataFrame(
        {
            "uid": uid_array,
            "is_pathologic": y,
            "acquisition_family": group_array,
        }
    ).merge(wide.reset_index(), on="uid", how="left", validate="one_to_one")
    probability_columns = [f"p_{model_name}_oof" for model_name in models]
    oof["p_ensemble_oof"] = oof[probability_columns].mean(axis=1).clip(1e-6, 1 - 1e-6)

    metric_rows: list[dict[str, Any]] = []
    for model_name, probability_column in [
        *[(name, f"p_{name}_oof") for name in models],
        ("mean_ensemble", "p_ensemble_oof"),
    ]:
        probabilities = oof[probability_column].to_numpy(dtype=float)
        metric_rows.append(
            {
                "model": model_name,
                "oof_auc": roc_auc_score(y, probabilities),
                "oof_log_loss": log_loss(y, probabilities, labels=[0, 1]),
                "oof_brier": brier_score_loss(y, probabilities),
                "oof_balanced_accuracy_0_5": balanced_accuracy_score(
                    y, probabilities >= 0.5
                ),
            }
        )
    metrics = pd.DataFrame(metric_rows).sort_values("oof_log_loss")
    _atomic_csv(oof, output_dir / "optuna_oof_predictions.csv")
    _atomic_csv(metrics, output_dir / "optuna_oof_metrics.csv")
    return oof, metrics


def run_optuna_experiment(
    folds: Sequence[FoldData],
    *,
    uids: Sequence[str],
    y: np.ndarray,
    groups: Sequence[str],
    output_dir: Path,
    experiment_config: dict[str, Any],
    n_trials_per_model: int = 50,
    models: Sequence[str] = MODEL_NAMES,
    random_seed: int = 20260821,
    cpu_jobs: int = 1,
    prefer_gpu: bool = True,
) -> OptunaExperimentResult:
    """Tune all requested classifiers and create resumable OOF predictions.

    ``n_trials_per_model`` means completed trials per classifier.  Interrupted
    trials remain in the journal, but do not count toward that target.
    """

    if not folds:
        raise ValueError("At least one fold is required.")
    unknown = sorted(set(models) - set(MODEL_NAMES))
    if unknown:
        raise ValueError(f"Unsupported models: {unknown}")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    canonical_config = {
        **experiment_config,
        "models": list(models),
        "n_trials_per_model": int(n_trials_per_model),
        "random_seed": int(random_seed),
    }
    experiment_hash = hashlib.sha256(
        json.dumps(canonical_config, sort_keys=True, default=_json_default).encode("utf-8")
    ).hexdigest()
    xgboost_device = _probe_xgboost_device(prefer_gpu, random_seed)
    canonical_config["experiment_hash"] = experiment_hash
    canonical_config["xgboost_device"] = xgboost_device
    _atomic_json(canonical_config, output_dir / "optuna_experiment_config.json")

    best_parameters: dict[str, dict[str, Any]] = {}
    summary_rows: list[dict[str, Any]] = []
    for model_name in models:
        study, parameters = _study_for_model(
            model_name,
            folds,
            output_dir,
            experiment_hash=experiment_hash,
            target_complete_trials=n_trials_per_model,
            random_seed=random_seed,
            cpu_jobs=cpu_jobs,
            xgboost_device=xgboost_device,
        )
        best_parameters[model_name] = parameters
        completed = sum(
            trial.state == optuna.trial.TrialState.COMPLETE for trial in study.trials
        )
        summary_rows.append(
            {
                "model": model_name,
                "completed_trials": completed,
                "best_cv_log_loss": float(study.best_value),
                "device": xgboost_device if model_name == "xgboost" else "cpu",
                "best_parameters_json": json.dumps(
                    parameters, sort_keys=True, default=_json_default
                ),
            }
        )
    tuning_summary = pd.DataFrame(summary_rows).sort_values("best_cv_log_loss")
    _atomic_csv(tuning_summary, output_dir / "optuna_tuning_summary.csv")
    _atomic_json(best_parameters, output_dir / "optuna_best_parameters.json")

    oof_predictions, metrics = _evaluate_oof(
        models,
        best_parameters,
        folds,
        uids,
        np.asarray(y, dtype=int),
        groups,
        output_dir,
        experiment_hash=experiment_hash,
        random_seed=random_seed,
        cpu_jobs=cpu_jobs,
        xgboost_device=xgboost_device,
    )
    return OptunaExperimentResult(
        tuning_summary=tuning_summary,
        oof_predictions=oof_predictions,
        metrics=metrics,
        best_parameters=best_parameters,
        xgboost_device=xgboost_device,
    )
