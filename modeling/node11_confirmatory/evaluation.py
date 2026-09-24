from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

from modeling.cnn.calibration import binary_metrics, cross_calibrate_oof
from modeling.cnn.io import atomic_csv, atomic_json
from modeling.cnn.training import train_one_fold
from modeling.dat_spect_v2.folds import create_or_load_patient_stratified_folds

from .config import ExperimentConfig
from .hgb import train_hgb_fold
from .search import cnn_configs


@dataclass
class FinalResult:
    metrics: pd.DataFrame
    oof_predictions: pd.DataFrame
    bootstrap_differences: pd.DataFrame
    deployment_manifest: dict[str, Any]


def _calibrate(frame: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    result = frame.copy()
    probability = result["probability"].to_numpy(dtype=float).clip(1e-6, 1 - 1e-6)
    result["logit"] = np.log(probability / (1.0 - probability))
    return cross_calibrate_oof(result)


def _metrics_row(
    frame: pd.DataFrame, *, family: str, candidate_id: str, temperature: float
) -> dict[str, Any]:
    raw = binary_metrics(frame, "probability")
    calibrated = binary_metrics(frame, "probability_cross_calibrated")
    return {
        "family": family,
        "candidate_id": candidate_id,
        **{f"raw_{key}": value for key, value in raw.items()},
        **{f"calibrated_{key}": value for key, value in calibrated.items()},
        "final_temperature": float(temperature),
    }


def _paired_bootstrap(
    baseline: pd.DataFrame,
    blend: pd.DataFrame,
    *,
    baseline_name: str,
    probability_column: str,
    seed: int,
    draws: int = 2000,
) -> dict[str, Any]:
    first = baseline.sort_values("uid").reset_index(drop=True)
    second = blend.sort_values("uid").reset_index(drop=True)
    if not first["uid"].equals(second["uid"]):
        raise ValueError("No se puede bootstrappear OOF con UIDs diferentes.")
    y = first["is_pathologic"].to_numpy(dtype=int)
    p_first = first[probability_column].to_numpy(dtype=float).clip(1e-6, 1 - 1e-6)
    p_second = second[probability_column].to_numpy(dtype=float).clip(1e-6, 1 - 1e-6)
    first_loss = -(y * np.log(p_first) + (1 - y) * np.log(1 - p_first))
    second_loss = -(y * np.log(p_second) + (1 - y) * np.log(1 - p_second))
    patient_gain = first_loss - second_loss
    rng = np.random.default_rng(seed)
    samples = np.empty(draws, dtype=float)
    for draw in range(draws):
        samples[draw] = patient_gain[rng.integers(0, len(patient_gain), len(patient_gain))].mean()
    return {
        "baseline": baseline_name,
        "blend": "fixed_raw_50_50",
        "probability_column": probability_column,
        "mean_log_loss_gain": float(patient_gain.mean()),
        "ci_low": float(np.quantile(samples, 0.025)),
        "ci_high": float(np.quantile(samples, 0.975)),
        "probability_gain_gt_zero": float(np.mean(samples > 0)),
        "draws": draws,
    }


def _subgroup_metrics(all_oof: pd.DataFrame, cohort: pd.DataFrame) -> pd.DataFrame:
    audited = all_oof.merge(
        cohort[["uid", "acquisition_family"]],
        on="uid",
        how="left",
        validate="many_to_one",
    )
    rows: list[dict[str, Any]] = []
    for (family, acquisition), frame in audited.groupby(
        ["family", "acquisition_family"], sort=False
    ):
        raw = binary_metrics(frame, "probability")
        calibrated = binary_metrics(frame, "probability_cross_calibrated")
        rows.append(
            {
                "family": family,
                "acquisition_family": acquisition,
                "n": len(frame),
                "n_pathologic": int(frame["is_pathologic"].sum()),
                **{f"raw_{key}": value for key, value in raw.items()},
                **{f"calibrated_{key}": value for key, value in calibrated.items()},
            }
        )
    return pd.DataFrame(rows)


def _weight_sweep(regional: pd.DataFrame, cnn: pd.DataFrame) -> pd.DataFrame:
    first = regional.sort_values("uid").reset_index(drop=True)
    second = cnn.sort_values("uid").reset_index(drop=True)
    if not first[["uid", "fold", "is_pathologic"]].equals(
        second[["uid", "fold", "is_pathologic"]]
    ):
        raise ValueError("Los dos expertos no comparten exactamente los mismos OOF.")
    rows: list[dict[str, float]] = []
    y = first["is_pathologic"].to_numpy(dtype=int)
    p_regional = first["probability"].to_numpy(dtype=float)
    p_cnn = second["probability"].to_numpy(dtype=float)
    for weight in np.linspace(0.0, 1.0, 101):
        probability = weight * p_regional + (1.0 - weight) * p_cnn
        rows.append(
            {
                "regional_weight": float(weight),
                "cnn_weight": float(1.0 - weight),
                "raw_log_loss": float(log_loss(y, probability, labels=[0, 1])),
            }
        )
    return pd.DataFrame(rows)


def _complementarity(regional: pd.DataFrame, cnn: pd.DataFrame) -> dict[str, float]:
    first = regional.sort_values("uid").reset_index(drop=True)
    second = cnn.sort_values("uid").reset_index(drop=True)
    p_first = first["probability"].to_numpy(dtype=float)
    p_second = second["probability"].to_numpy(dtype=float)
    return {
        "raw_probability_correlation": float(np.corrcoef(p_first, p_second)[0, 1]),
        "raw_mean_absolute_difference": float(np.mean(np.abs(p_first - p_second))),
        "raw_decision_disagreement_0_5": float(
            np.mean((p_first >= 0.5) != (p_second >= 0.5))
        ),
    }


def run_final(
    cohort: pd.DataFrame,
    folds: pd.DataFrame,
    finalists: dict[str, dict[str, Any]],
    *,
    experiment: ExperimentConfig,
    run_dir: Path,
    device: str | None = None,
) -> FinalResult:
    run_dir.mkdir(parents=True, exist_ok=True)
    required = {"regional_hgb", "cnn_25d_image_only"}
    if set(finalists) != required:
        raise ValueError(f"El nodo 11 requiere exactamente {sorted(required)}.")
    regional_candidate = finalists["regional_hgb"]
    cnn_candidate = finalists["cnn_25d_image_only"]
    regional_predictions: list[pd.DataFrame] = []
    cnn_predictions: list[pd.DataFrame] = []
    epoch_rows: list[dict[str, Any]] = []
    hgb_rows: list[dict[str, Any]] = []

    regional_root = (
        run_dir / "models" / "regional_hgb" / str(regional_candidate["candidate_id"])
    )
    cnn_root = run_dir / "models" / "cnn_25d_image_only" / str(cnn_candidate["candidate_id"])
    model_config, train_config = cnn_configs(dict(cnn_candidate["parameters"]), experiment)
    for outer_fold in range(experiment.search.n_splits_final):
        hgb_result = train_hgb_fold(
            cohort,
            folds,
            fold=outer_fold,
            parameters=dict(regional_candidate["parameters"]),
            train_config=experiment.train,
            output_dir=regional_root / f"fold_{outer_fold}",
            seed_offset=200_000,
        )
        regional_predictions.append(hgb_result.predictions)
        hgb_rows.append(
            {
                "outer_fold": outer_fold,
                "best_iteration": hgb_result.best_iteration,
                "outer_log_loss": hgb_result.validation_log_loss,
            }
        )

        merged = cohort.merge(folds[["uid", "fold"]], on="uid", validate="one_to_one")
        outer_train = merged.loc[merged["fold"] != outer_fold].drop(columns="fold")
        selection_dir = cnn_root / f"fold_{outer_fold}" / "epoch_selection"
        inner_manifest = create_or_load_patient_stratified_folds(
            outer_train["uid"],
            outer_train["is_pathologic"],
            outer_train["acquisition_family"],
            selection_dir / "inner_folds.csv",
            n_splits=experiment.search.inner_epoch_splits,
            random_seed=experiment.train.seed + 300_000 + outer_fold * 10_007,
            n_candidates=experiment.search.fold_candidates,
            background_valid=outer_train["background_qc_valid"],
        )
        selected_epochs: list[int] = []
        for inner_fold in range(experiment.search.inner_epoch_repeats):
            selection = train_one_fold(
                cohort,
                inner_manifest,
                fold=inner_fold,
                model_config=model_config,
                data_config=experiment.cnn_data,
                train_config=train_config,
                output_dir=selection_dir / f"inner_fold_{inner_fold}",
                pca_variance=None,
                fixed_epochs=None,
                scheduler_epochs=experiment.train.max_epochs,
                device=device,
            )
            selected = int(selection.selected_epoch) + 1
            selected_epochs.append(selected)
            epoch_rows.append(
                {
                    "outer_fold": outer_fold,
                    "inner_fold": inner_fold,
                    "best_epoch": selected,
                    "inner_validation_log_loss": selection.validation_log_loss,
                }
            )
        refit_epochs = max(1, round(float(np.median(selected_epochs))))
        refit = train_one_fold(
            cohort,
            folds,
            fold=outer_fold,
            model_config=model_config,
            data_config=experiment.cnn_data,
            train_config=train_config,
            output_dir=cnn_root / f"fold_{outer_fold}" / "refit",
            pca_variance=None,
            fixed_epochs=refit_epochs,
            scheduler_epochs=experiment.train.max_epochs,
            device=device,
        )
        predictions = refit.predictions.copy()
        predictions["inner_best_epochs"] = ",".join(map(str, selected_epochs))
        predictions["refit_epochs"] = refit_epochs
        cnn_predictions.append(predictions)

    regional_raw = pd.concat(regional_predictions, ignore_index=True).sort_values("uid")
    cnn_raw = pd.concat(cnn_predictions, ignore_index=True).sort_values("uid")
    expected = cohort["uid"].astype(str).nunique()
    for name, frame in (("regional_hgb", regional_raw), ("cnn_25d_image_only", cnn_raw)):
        if len(frame) != expected or frame["uid"].astype(str).nunique() != expected:
            raise RuntimeError(f"OOF incompleto para {name}.")
    comparison_columns = ["uid", "fold", "is_pathologic"]
    if not regional_raw.reset_index(drop=True)[comparison_columns].equals(
        cnn_raw.reset_index(drop=True)[comparison_columns]
    ):
        raise RuntimeError("Los expertos del nodo 11 no terminaron sobre los mismos folds.")

    regional, regional_temperature = _calibrate(regional_raw.reset_index(drop=True))
    cnn, cnn_temperature = _calibrate(cnn_raw.reset_index(drop=True))
    weight = float(experiment.regional_blend_weight)
    blend = regional[["uid", "is_pathologic", "fold"]].copy()
    blend["probability"] = (
        weight * regional["probability"].to_numpy(dtype=float)
        + (1.0 - weight) * cnn["probability"].to_numpy(dtype=float)
    ).clip(1e-6, 1 - 1e-6)
    blend, blend_temperature = _calibrate(blend)

    identities = {
        "regional_hgb": str(regional_candidate["candidate_id"]),
        "cnn_25d_image_only": str(cnn_candidate["candidate_id"]),
        "fixed_raw_50_50": "node11_fixed_raw_50_50",
    }
    frames = {
        "regional_hgb": regional,
        "cnn_25d_image_only": cnn,
        "fixed_raw_50_50": blend,
    }
    temperatures = {
        "regional_hgb": regional_temperature,
        "cnn_25d_image_only": cnn_temperature,
        "fixed_raw_50_50": blend_temperature,
    }
    metric_rows = [
        _metrics_row(
            frame,
            family=family,
            candidate_id=identities[family],
            temperature=temperatures[family],
        )
        for family, frame in frames.items()
    ]
    metrics = pd.DataFrame(metric_rows).sort_values("calibrated_log_loss").reset_index(drop=True)
    all_oof = pd.concat(
        [frame.assign(family=family, candidate_id=identities[family]) for family, frame in frames.items()],
        ignore_index=True,
    )
    bootstrap = pd.DataFrame(
        [
            _paired_bootstrap(
                frames[baseline],
                blend,
                baseline_name=baseline,
                probability_column=column,
                seed=experiment.train.seed + offset,
            )
            for offset, (baseline, column) in enumerate(
                (
                    ("regional_hgb", "probability"),
                    ("cnn_25d_image_only", "probability"),
                    ("regional_hgb", "probability_cross_calibrated"),
                    ("cnn_25d_image_only", "probability_cross_calibrated"),
                )
            )
        ]
    )
    subgroup = _subgroup_metrics(all_oof, cohort)
    supported = subgroup.loc[subgroup["n"] >= 10]
    worst = (
        supported.groupby("family", as_index=False)["calibrated_log_loss"]
        .max()
        .rename(columns={"calibrated_log_loss": "worst_supported_family_log_loss"})
    )
    metrics = metrics.merge(worst, on="family", how="left", validate="one_to_one")
    sweep = _weight_sweep(regional, cnn)
    complementarity = _complementarity(regional, cnn)

    atomic_csv(regional.assign(family="regional_hgb"), run_dir / "oof_regional_hgb.csv")
    atomic_csv(cnn.assign(family="cnn_25d_image_only"), run_dir / "oof_cnn_25d_image_only.csv")
    atomic_csv(blend.assign(family="fixed_raw_50_50"), run_dir / "oof_fixed_raw_50_50.csv")
    atomic_csv(all_oof, run_dir / "all_oof_predictions.csv")
    atomic_csv(metrics, run_dir / "final_metrics.csv")
    atomic_csv(bootstrap, run_dir / "paired_bootstrap_differences.csv")
    atomic_csv(subgroup, run_dir / "subgroup_metrics_by_acquisition_family.csv")
    atomic_csv(sweep, run_dir / "exploratory_weight_sweep.csv")
    atomic_csv(pd.DataFrame(epoch_rows), run_dir / "cnn_epoch_selection.csv")
    atomic_csv(pd.DataFrame(hgb_rows), run_dir / "hgb_iteration_selection.csv")
    atomic_json(complementarity, run_dir / "complementarity.json")
    manifest: dict[str, Any] = {
        "primary_family": "fixed_raw_50_50",
        "primary_candidate_id": "node11_fixed_raw_50_50",
        "regional_candidate_id": identities["regional_hgb"],
        "cnn_candidate_id": identities["cnn_25d_image_only"],
        "regional_blend_weight": weight,
        "cnn_blend_weight": 1.0 - weight,
        "blend_stage": "raw OOF probabilities before one cross-fitted temperature calibration",
        "same_outer_folds": True,
        "same_base_seed": experiment.train.seed,
        "n_outer_folds": experiment.search.n_splits_final,
        "cnn_epoch_selection": (
            "two train-only inner folds per outer fold; median best_epoch; full outer-train refit"
        ),
        "cnn_scheduler_horizon": experiment.train.max_epochs,
        "hgb_iteration_selection": (
            "train-only internal early stopping; full outer-train refit at best_iteration"
        ),
        "outer_fold_used_for_early_stopping": False,
        "weight_sweep_status": "exploratory diagnostic only; never selects the primary weight",
        "selection_metric": "five-fold OOF log loss after common-fold training",
        "calibration": "cross-fitted temperature after blending",
        "temperatures": temperatures,
        "complementarity": complementarity,
        "deployment_status": "development validation; package only after runtime and external audit",
    }
    atomic_json(manifest, run_dir / "deployment_manifest.json")
    return FinalResult(metrics, all_oof, bootstrap, manifest)
