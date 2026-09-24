from __future__ import annotations

import pickle
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import pandas as pd
from optuna.exceptions import ExperimentalWarning
from sklearn.linear_model import LogisticRegression

from modeling.cnn.calibration import binary_metrics, cross_calibrate_oof
from modeling.cnn.io import atomic_csv, atomic_json, replace_with_retry

from .config import ExperimentConfig
from .search import (
    _storage,
    _trial_table,
    completed_trials,
    evaluate_family_fold,
    optimize_until_complete,
)


@dataclass
class FinalResult:
    metrics: pd.DataFrame
    oof_predictions: pd.DataFrame
    deployment_manifest: dict[str, Any]


def _atomic_pickle(value: Any, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f"{destination.name}.tmp")
    with temporary.open("wb") as stream:
        pickle.dump(value, stream, protocol=pickle.HIGHEST_PROTOCOL)
    replace_with_retry(temporary, destination)


def _calibrated_oof(frame: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    result = frame.copy()
    probability = result["probability"].to_numpy(dtype=float).clip(1e-6, 1 - 1e-6)
    result["logit"] = np.log(probability / (1.0 - probability))
    calibrated, temperature = cross_calibrate_oof(result)
    return calibrated, temperature


def _metric_row(
    frame: pd.DataFrame,
    *,
    family: str,
    candidate_id: str,
    temperature: float,
) -> dict[str, Any]:
    raw = binary_metrics(frame, "probability")
    calibrated = binary_metrics(frame, "probability_cross_calibrated")
    return {
        "family": family,
        "candidate_id": candidate_id,
        **{f"raw_{key}": value for key, value in raw.items()},
        **{f"calibrated_{key}": value for key, value in calibrated.items()},
        "final_temperature": temperature,
    }


def _stack_matrix(oof_by_family: dict[str, pd.DataFrame]) -> pd.DataFrame:
    merged: pd.DataFrame | None = None
    for family, frame in oof_by_family.items():
        selected = frame[["uid", "is_pathologic", "fold", "probability"]].rename(
            columns={"probability": f"p__{family}"}
        )
        if merged is None:
            merged = selected
        else:
            merged = merged.merge(
                selected.drop(columns=["is_pathologic", "fold"]),
                on="uid",
                validate="one_to_one",
            )
    if merged is None:
        raise RuntimeError("No existen OOF para stacking.")
    return merged.sort_values("uid").reset_index(drop=True)


def _stack_crossfit(
    matrix: pd.DataFrame, parameters: dict[str, Any]
) -> tuple[pd.DataFrame, list[Any]]:
    feature_columns = [column for column in matrix if column.startswith("p__")]
    probabilities = matrix[feature_columns].to_numpy(dtype=float).clip(1e-6, 1 - 1e-6)
    logits = np.log(probabilities / (1 - probabilities))
    output = np.full(len(matrix), np.nan, dtype=float)
    models: list[Any] = []
    for fold in sorted(matrix["fold"].unique()):
        train = matrix["fold"].to_numpy() != fold
        valid = ~train
        model = LogisticRegression(
            C=float(parameters["C"]),
            l1_ratio=float(parameters["l1_ratio"]),
            penalty="elasticnet",
            solver="saga",
            fit_intercept=bool(parameters["fit_intercept"]),
            max_iter=5000,
            random_state=20260831 + int(fold),
        )
        model.fit(logits[train], matrix.loc[train, "is_pathologic"].to_numpy(dtype=int))
        output[valid] = model.predict_proba(logits[valid])[:, 1]
        models.append(model)
    result = matrix[["uid", "is_pathologic", "fold"]].copy()
    result["probability"] = output.clip(1e-6, 1 - 1e-6)
    return result, models


def _prespecified_mean(matrix: pd.DataFrame) -> pd.DataFrame:
    """Label-free ensemble kept eligible for honest outer-fold comparison."""

    feature_columns = [column for column in matrix if column.startswith("p__")]
    result = matrix[["uid", "is_pathologic", "fold"]].copy()
    result["probability"] = (
        matrix[feature_columns].to_numpy(dtype=float).mean(axis=1).clip(1e-6, 1 - 1e-6)
    )
    return result


def _subgroup_audit(all_oof: pd.DataFrame, cohort: pd.DataFrame) -> pd.DataFrame:
    audited = all_oof.merge(
        cohort[["uid", "acquisition_family"]], on="uid", how="left", validate="many_to_one"
    )
    rows: list[dict[str, Any]] = []
    for (family, candidate_id, acquisition), frame in audited.groupby(
        ["family", "candidate_id", "acquisition_family"], sort=False
    ):
        raw = binary_metrics(frame, "probability")
        calibrated = binary_metrics(frame, "probability_cross_calibrated")
        rows.append(
            {
                "family": family,
                "candidate_id": candidate_id,
                "acquisition_family": acquisition,
                "n": len(frame),
                "n_pathologic": int(frame["is_pathologic"].sum()),
                **{f"raw_{key}": value for key, value in raw.items()},
                **{f"calibrated_{key}": value for key, value in calibrated.items()},
            }
        )
    return pd.DataFrame(rows)


def _bootstrap_log_loss(all_oof: pd.DataFrame, *, seed: int, draws: int = 1000) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    for (family, candidate_id), frame in all_oof.groupby(["family", "candidate_id"], sort=False):
        y = frame["is_pathologic"].to_numpy(dtype=int)
        probability = (
            frame["probability_cross_calibrated"].to_numpy(dtype=float).clip(1e-6, 1 - 1e-6)
        )
        losses = -(y * np.log(probability) + (1 - y) * np.log(1 - probability))
        samples = np.empty(draws, dtype=float)
        for draw in range(draws):
            samples[draw] = losses[rng.integers(0, len(losses), size=len(losses))].mean()
        rows.append(
            {
                "family": family,
                "candidate_id": candidate_id,
                "bootstrap_draws": draws,
                "calibrated_log_loss_ci_low": float(np.quantile(samples, 0.025)),
                "calibrated_log_loss_ci_high": float(np.quantile(samples, 0.975)),
            }
        )
    return pd.DataFrame(rows)


def _pairwise_complementarity(all_oof: pd.DataFrame) -> pd.DataFrame:
    pivot = all_oof.pivot(index="uid", columns="family", values="probability_cross_calibrated")
    rows: list[dict[str, Any]] = []
    columns = list(pivot.columns)
    for first_index, first in enumerate(columns):
        for second in columns[first_index + 1 :]:
            first_values = pivot[first].to_numpy(dtype=float)
            second_values = pivot[second].to_numpy(dtype=float)
            rows.append(
                {
                    "first_family": first,
                    "second_family": second,
                    "probability_correlation": float(
                        np.corrcoef(first_values, second_values)[0, 1]
                    ),
                    "mean_absolute_probability_difference": float(
                        np.mean(np.abs(first_values - second_values))
                    ),
                    "decision_disagreement_0_5": float(
                        np.mean((first_values >= 0.5) != (second_values >= 0.5))
                    ),
                }
            )
    return pd.DataFrame(rows)


def _run_stacking_search(
    matrix: pd.DataFrame,
    *,
    experiment: ExperimentConfig,
    run_dir: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    database = run_dir.parent / "search" / "optuna_node10.sqlite3"
    storage = _storage(database, experiment.search)
    study_name = "node10_hybrid_stacking"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ExperimentalWarning)
        sampler = optuna.samplers.TPESampler(
            seed=experiment.train.seed + 99,
            n_startup_trials=min(5, experiment.search.effective_stack_trials),
            multivariate=True,
            constant_liar=True,
        )
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
        direction="minimize",
        sampler=sampler,
    )

    def objective(trial: optuna.Trial) -> float:
        parameters = {
            "C": trial.suggest_float("C", 0.01, 30.0, log=True),
            "l1_ratio": trial.suggest_float("l1_ratio", 0.0, 1.0),
            "fit_intercept": trial.suggest_categorical("fit_intercept", [True, False]),
        }
        predictions, _ = _stack_crossfit(matrix, parameters)
        score = binary_metrics(predictions, "probability")["log_loss"]
        trial.set_user_attr("parameters", parameters)
        return float(score)

    optimize_until_complete(
        study,
        objective,
        target=experiment.search.effective_stack_trials,
        timeout_seconds=experiment.search.optuna_timeout_seconds,
    )
    complete = completed_trials(study)
    best = min(complete, key=lambda item: float(item.value))
    parameters = dict(best.user_attrs["parameters"])
    predictions, models = _stack_crossfit(matrix, parameters)
    atomic_csv(_trial_table(study), run_dir / "stacking_trials.csv")
    _atomic_pickle(
        {
            "crossfit_models": models,
            "parameters": parameters,
            "feature_columns": [column for column in matrix if column.startswith("p__")],
            "warning": "For deployment refit base experts and a final meta-model using training-only OOF logits.",
        },
        run_dir / "stacking_models.pkl",
    )
    metadata = {
        "study_name": study_name,
        "completed_trials": len(complete),
        "total_trials": len(study.trials),
        "best_trial": int(best.number),
        "search_log_loss": float(best.value),
        "parameters": parameters,
        "meta_validation": "cross-fit by frozen outer fold on base OOF logits",
        "selection_validity": (
            "exploratory: outer-fold base predictions are not regenerated inside each "
            "meta-training fold; a strictly nested second level is required for promotion"
        ),
    }
    atomic_json(metadata, run_dir / "stacking_best.json")
    return predictions, metadata


def run_final_cv5(
    cohort: pd.DataFrame,
    folds: pd.DataFrame,
    finalists: list[dict[str, Any]],
    *,
    experiment: ExperimentConfig,
    run_dir: Path,
    device: str | None = None,
) -> FinalResult:
    run_dir.mkdir(parents=True, exist_ok=True)
    oof_by_family: dict[str, pd.DataFrame] = {}
    metric_rows: list[dict[str, Any]] = []
    temperatures: dict[str, float] = {}
    for candidate in finalists:
        family = str(candidate["family"])
        candidate_id = str(candidate["candidate_id"])
        predictions: list[pd.DataFrame] = []
        for fold in range(experiment.search.n_splits_final):
            result = evaluate_family_fold(
                cohort,
                folds,
                family=family,
                fold=fold,
                parameters=dict(candidate["parameters"]),
                experiment=experiment,
                output_dir=run_dir / "models" / family / f"fold_{fold}",
                fixed_epochs=candidate.get("fixed_epochs"),
                device=device,
            )
            predictions.append(result.predictions)
        oof = pd.concat(predictions, ignore_index=True).sort_values("uid").reset_index(drop=True)
        if len(oof) != len(cohort) or oof["uid"].nunique() != len(cohort):
            raise RuntimeError(f"OOF incompleto para {family}.")
        calibrated, temperature = _calibrated_oof(oof)
        calibrated["family"] = family
        calibrated["candidate_id"] = candidate_id
        atomic_csv(calibrated, run_dir / f"oof_{family}.csv")
        oof_by_family[family] = calibrated
        temperatures[family] = float(temperature)
        metric_rows.append(
            _metric_row(
                calibrated,
                family=family,
                candidate_id=candidate_id,
                temperature=temperature,
            )
        )
    matrix = _stack_matrix(oof_by_family)
    mean_oof = _prespecified_mean(matrix)
    mean_calibrated, mean_temperature = _calibrated_oof(mean_oof)
    mean_calibrated["family"] = "hybrid_mean_prespecified"
    mean_calibrated["candidate_id"] = "node10_hybrid_mean_prespecified"
    atomic_csv(mean_calibrated, run_dir / "oof_hybrid_mean_prespecified.csv")
    metric_rows.append(
        _metric_row(
            mean_calibrated,
            family="hybrid_mean_prespecified",
            candidate_id="node10_hybrid_mean_prespecified",
            temperature=mean_temperature,
        )
    )
    stack_oof, stack_metadata = _run_stacking_search(
        matrix,
        experiment=experiment,
        run_dir=run_dir,
    )
    stack_calibrated, stack_temperature = _calibrated_oof(stack_oof)
    stack_calibrated["family"] = "hybrid_stacking"
    stack_calibrated["candidate_id"] = "node10_hybrid_stacking"
    atomic_csv(stack_calibrated, run_dir / "oof_hybrid_stacking.csv")
    metric_rows.append(
        _metric_row(
            stack_calibrated,
            family="hybrid_stacking",
            candidate_id="node10_hybrid_stacking",
            temperature=stack_temperature,
        )
    )
    all_oof = pd.concat(
        [*oof_by_family.values(), mean_calibrated, stack_calibrated], ignore_index=True
    )
    subgroup = _subgroup_audit(all_oof, cohort)
    bootstrap = _bootstrap_log_loss(all_oof, seed=experiment.train.seed)
    complementarity = _pairwise_complementarity(all_oof)
    supported = subgroup.loc[subgroup["n"] >= 10]
    worst_group = (
        supported.groupby(["family", "candidate_id"], as_index=False)["calibrated_log_loss"]
        .max()
        .rename(columns={"calibrated_log_loss": "worst_supported_family_log_loss"})
    )
    metrics = pd.DataFrame(metric_rows).merge(
        bootstrap, on=["family", "candidate_id"], how="left", validate="one_to_one"
    )
    metrics = metrics.merge(
        worst_group, on=["family", "candidate_id"], how="left", validate="one_to_one"
    )
    metrics["promotion_eligible"] = metrics["family"] != "hybrid_stacking"
    metrics = metrics.sort_values("calibrated_log_loss").reset_index(drop=True)
    atomic_csv(metrics, run_dir / "final_metrics.csv")
    atomic_csv(all_oof, run_dir / "all_oof_predictions.csv")
    atomic_csv(subgroup, run_dir / "subgroup_metrics_by_acquisition_family.csv")
    atomic_csv(bootstrap, run_dir / "bootstrap_log_loss.csv")
    atomic_csv(complementarity, run_dir / "pairwise_complementarity.csv")
    primary = metrics.loc[metrics["promotion_eligible"]].iloc[0]
    manifest = {
        "primary_family": str(primary["family"]),
        "primary_candidate_id": str(primary["candidate_id"]),
        "n_base_experts": len(oof_by_family),
        "n_fold_models_per_expert": experiment.search.n_splits_final,
        "base_temperatures": temperatures,
        "stack_temperature": stack_temperature,
        "prespecified_mean_temperature": mean_temperature,
        "stacking": stack_metadata,
        "selection_metric": "five-fold OOF cross-calibrated log loss",
        "uncertainty": "1000 patient-level bootstrap draws on calibrated OOF log loss",
        "subgroup_audit": "acquisition families with n >= 10 summarized by worst log loss",
        "primary_excludes_exploratory_stacking": True,
        "acquisition_family_as_predictor": False,
        "deployment_status": "research evaluation; package only after external/runtime audit",
    }
    atomic_json(manifest, run_dir / "deployment_manifest.json")
    return FinalResult(metrics, all_oof, manifest)
