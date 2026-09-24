from __future__ import annotations

import json
import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import log_loss

from modeling.cnn.io import atomic_csv, atomic_json, replace_with_retry

from .config import TrainConfig


@dataclass
class HGBFoldResult:
    fold: int
    validation_log_loss: float
    best_iteration: int
    predictions: pd.DataFrame
    model_path: Path


def regional_columns(frame: pd.DataFrame) -> list[str]:
    columns = sorted(column for column in frame if column.startswith("regional930_"))
    if not columns:
        raise ValueError("No se encontraron las variables regional930 del nodo 10.")
    return columns


def _atomic_pickle(payload: Any, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f"{destination.name}.tmp.{os.getpid()}")
    with temporary.open("wb") as stream:
        pickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)
    replace_with_retry(temporary, destination)


def _loss_curve(values: Any) -> np.ndarray:
    scores = np.asarray(values, dtype=float)
    if not len(scores):
        return scores
    # sklearn early stopping maximizes its score. With scoring="loss" this is
    # the negative binary cross entropy; accepting both signs keeps the saved
    # history robust across compatible sklearn releases.
    return -scores if float(np.nanmedian(scores)) <= 0.0 else scores


def _impute_from_train(
    train: np.ndarray, valid: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    median = np.nanmedian(train, axis=0)
    median = np.where(np.isfinite(median), median, 0.0)
    train_clean = np.where(np.isfinite(train), train, median)
    valid_clean = np.where(np.isfinite(valid), valid, median)
    return train_clean, valid_clean, median


def train_hgb_fold(
    cohort: pd.DataFrame,
    folds: pd.DataFrame,
    *,
    fold: int,
    parameters: dict[str, Any],
    train_config: TrainConfig,
    output_dir: Path,
    seed_offset: int = 0,
) -> HGBFoldResult:
    """Select the best boosting iteration on train-only data, then refit.

    The external validation fold is never passed to HGB's internal early
    stopping. The final estimator is rebuilt on the complete external-training
    partition using exactly the best internal iteration.
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = output_dir / "validation_predictions.csv"
    model_path = output_dir / "model.pkl"
    metadata_path = output_dir / "metadata.json"
    history_path = output_dir / "iteration_history.csv"
    if all(path.exists() for path in (prediction_path, model_path, metadata_path, history_path)):
        predictions = pd.read_csv(prediction_path)
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        score = float(
            log_loss(
                predictions["is_pathologic"],
                predictions["probability"].clip(1e-6, 1 - 1e-6),
                labels=[0, 1],
            )
        )
        return HGBFoldResult(fold, score, int(metadata["best_iteration"]), predictions, model_path)

    manifest = folds[["uid", "fold"]].copy()
    manifest["uid"] = manifest["uid"].astype(str)
    merged = cohort.assign(uid=cohort["uid"].astype(str)).merge(
        manifest, on="uid", how="inner", validate="one_to_one"
    )
    train_frame = merged.loc[merged["fold"] != fold].reset_index(drop=True)
    valid_frame = merged.loc[merged["fold"] == fold].reset_index(drop=True)
    if train_frame.empty or valid_frame.empty:
        raise ValueError(f"Fold HGB {fold} vacio.")
    columns = regional_columns(cohort)
    x_train, x_valid, median = _impute_from_train(
        train_frame[columns].to_numpy(dtype=np.float64),
        valid_frame[columns].to_numpy(dtype=np.float64),
    )
    y_train = train_frame["is_pathologic"].to_numpy(dtype=int)
    y_valid = valid_frame["is_pathologic"].to_numpy(dtype=int)
    seed = int(train_config.seed + seed_offset + fold * 3011)
    common = {
        "learning_rate": float(parameters["learning_rate"]),
        "max_leaf_nodes": int(parameters["max_leaf_nodes"]),
        "min_samples_leaf": int(parameters["min_samples_leaf"]),
        "l2_regularization": float(parameters["l2_regularization"]),
        "class_weight": "balanced",
        "random_state": seed,
    }
    selector = HistGradientBoostingClassifier(
        **common,
        max_iter=train_config.hgb_max_iter,
        early_stopping=True,
        scoring="loss",
        validation_fraction=train_config.hgb_validation_fraction,
        n_iter_no_change=train_config.hgb_patience,
        tol=train_config.hgb_tolerance,
    )
    selector.fit(x_train, y_train)
    train_loss = _loss_curve(selector.train_score_)
    validation_loss = _loss_curve(selector.validation_score_)
    if not len(validation_loss) or not np.isfinite(validation_loss).any():
        raise RuntimeError("HGB no produjo una curva interna de validacion util.")
    best_index = int(np.nanargmin(validation_loss))
    # Curves include iteration zero (the class-prior predictor).
    best_iteration = int(np.clip(best_index, 1, selector.n_iter_))
    history = pd.DataFrame(
        {
            "iteration": np.arange(len(validation_loss), dtype=int),
            "train_loss": train_loss,
            "validation_loss": validation_loss,
            "is_best": np.arange(len(validation_loss), dtype=int) == best_index,
        }
    )
    final_model = HistGradientBoostingClassifier(
        **common,
        max_iter=best_iteration,
        early_stopping=False,
    )
    final_model.fit(x_train, y_train)
    probability = final_model.predict_proba(x_valid)[:, 1].clip(1e-6, 1 - 1e-6)
    predictions = pd.DataFrame(
        {
            "uid": valid_frame["uid"].astype(str),
            "is_pathologic": y_valid,
            "probability": probability,
            "fold": int(fold),
            "selected_iteration": best_iteration,
        }
    )
    score = float(log_loss(y_valid, probability, labels=[0, 1]))
    _atomic_pickle(
        {
            "model": final_model,
            "median": median,
            "feature_columns": columns,
            "parameters": parameters,
            "best_iteration": best_iteration,
        },
        model_path,
    )
    atomic_csv(history, history_path)
    atomic_json(
        {
            "fold": fold,
            "seed": seed,
            "parameters": parameters,
            "max_iter": train_config.hgb_max_iter,
            "iterations_observed": int(selector.n_iter_),
            "best_iteration": best_iteration,
            "best_internal_validation_loss": float(validation_loss[best_index]),
            "external_validation_log_loss": score,
            "selection_partition": "internal train-only validation_fraction",
            "external_fold_used_for_early_stopping": False,
        },
        metadata_path,
    )
    # Predictions are the completion marker and are written last.
    atomic_csv(predictions, prediction_path)
    return HGBFoldResult(fold, score, best_iteration, predictions, model_path)
