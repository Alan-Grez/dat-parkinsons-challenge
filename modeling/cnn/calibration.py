from __future__ import annotations

from itertools import pairwise

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import expit
from sklearn.metrics import (
    balanced_accuracy_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)


def expected_calibration_error(y: np.ndarray, probability: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    error = 0.0
    for lower, upper in pairwise(edges):
        selected = (probability >= lower) & (
            probability <= upper if upper == 1.0 else probability < upper
        )
        if not selected.any():
            continue
        error += selected.mean() * abs(probability[selected].mean() - y[selected].mean())
    return float(error)


def binary_metrics(
    frame: pd.DataFrame, probability_column: str = "probability"
) -> dict[str, float]:
    y = frame["is_pathologic"].to_numpy(dtype=int)
    probability = frame[probability_column].to_numpy(dtype=float).clip(1e-6, 1 - 1e-6)
    return {
        "log_loss": float(log_loss(y, probability, labels=[0, 1])),
        "brier": float(brier_score_loss(y, probability)),
        "auc": float(roc_auc_score(y, probability)) if len(np.unique(y)) == 2 else float("nan"),
        "balanced_accuracy_0_5": float(balanced_accuracy_score(y, probability >= 0.5)),
        "ece_10": expected_calibration_error(y, probability, bins=10),
    }


def fit_temperature(logits: np.ndarray, y: np.ndarray) -> float:
    """Bounded global optimum in inverse temperature for binary log loss.

    NLL is convex in beta=1/T. Optimizing a clamped exp(log_T) with LBFGS
    can overshoot to T=.05 and become stuck behind a zero clamp gradient.
    This occurred for two folds of the node10 mean ensemble. Bracket the
    monotone derivative instead; keep identity as an explicit candidate.
    """
    z = np.asarray(logits, dtype=np.float64).reshape(-1)
    target = np.asarray(y, dtype=np.float64).reshape(-1)
    if not len(z) or z.shape != target.shape:
        raise ValueError("Logits y etiquetas deben tener la misma longitud no vacia.")
    if not np.isfinite(z).all() or not np.isin(target, [0.0, 1.0]).all():
        raise ValueError("Calibracion requiere logits finitos y etiquetas binarias.")
    if np.all(z == 0):
        return 1.0

    def derivative(beta: float) -> float:
        return float(np.mean(z * (expit(beta * z) - target)))

    lower, upper = 0.05, 20.0
    candidates = [lower, 1.0, upper]
    if derivative(lower) < 0 < derivative(upper):
        candidates.append(brentq(derivative, lower, upper, xtol=1e-12))
    best_beta = min(
        candidates,
        key=lambda beta: float(np.mean(np.logaddexp(0.0, beta * z) - target * beta * z)),
    )
    return float(1.0 / best_beta)


def apply_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
    scaled = np.asarray(logits, dtype=float) / max(float(temperature), 1e-6)
    return 1.0 / (1.0 + np.exp(-np.clip(scaled, -40, 40)))


def cross_calibrate_oof(frame: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    calibrated = frame.copy()
    calibrated["probability_cross_calibrated"] = np.nan
    for fold in sorted(calibrated["fold"].unique()):
        train = calibrated["fold"] != fold
        valid = calibrated["fold"] == fold
        temperature = fit_temperature(
            calibrated.loc[train, "logit"].to_numpy(dtype=float),
            calibrated.loc[train, "is_pathologic"].to_numpy(dtype=int),
        )
        calibrated.loc[valid, "probability_cross_calibrated"] = apply_temperature(
            calibrated.loc[valid, "logit"].to_numpy(dtype=float), temperature
        )
    final_temperature = fit_temperature(
        calibrated["logit"].to_numpy(dtype=float),
        calibrated["is_pathologic"].to_numpy(dtype=int),
    )
    return calibrated, final_temperature
