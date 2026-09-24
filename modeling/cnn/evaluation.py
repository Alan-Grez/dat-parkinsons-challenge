from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

from .calibration import binary_metrics, cross_calibrate_oof
from .config import DataConfig, ModelConfig, TrainConfig
from .io import atomic_csv, atomic_json
from .training import train_one_fold


@dataclass
class FinalEvaluationResult:
    metrics: pd.DataFrame
    primary_manifest: dict[str, Any]
    oof_predictions: pd.DataFrame


def _group_diagnostics(frame: pd.DataFrame, minimum_size: int = 10) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for family, group in frame.groupby("acquisition_family"):
        if len(group) < minimum_size or group["is_pathologic"].nunique() < 2:
            continue
        records.append(
            {
                "acquisition_family": str(family),
                "n": len(group),
                "prevalence": float(group["is_pathologic"].mean()),
                "log_loss": float(
                    log_loss(
                        group["is_pathologic"],
                        group["probability_cross_calibrated"].clip(1e-6, 1 - 1e-6),
                        labels=[0, 1],
                    )
                ),
            }
        )
    return pd.DataFrame.from_records(records)


def _background_subgroup_metrics(frame: pd.DataFrame) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for valid, name in ((True, "valid_background"), (False, "invalid_background")):
        subgroup = frame.loc[frame["background_qc_valid"].astype(bool) == valid]
        metrics[f"{name}_n"] = float(len(subgroup))
        if subgroup.empty or subgroup["is_pathologic"].nunique() < 2:
            metrics[f"{name}_log_loss"] = float("nan")
            metrics[f"{name}_brier"] = float("nan")
            continue
        values = binary_metrics(subgroup, "probability_cross_calibrated")
        metrics[f"{name}_log_loss"] = values["log_loss"]
        metrics[f"{name}_brier"] = values["brier"]
    return metrics


def run_finalist_cv5(
    cohort: pd.DataFrame,
    final_folds: pd.DataFrame,
    finalists: list[dict[str, Any]],
    *,
    run_dir: Path,
    data_config: DataConfig,
    device: str | None = None,
) -> FinalEvaluationResult:
    run_dir.mkdir(parents=True, exist_ok=True)
    metric_records: list[dict[str, Any]] = []
    oof_by_candidate: dict[str, pd.DataFrame] = {}
    model_paths: dict[str, list[str]] = {}
    for candidate in finalists:
        candidate_id = str(candidate["candidate_id"])
        model_config = ModelConfig(**candidate["model_config"])
        train_config = TrainConfig(**candidate["train_config"])
        fixed_epochs = int(candidate["fixed_epochs"])
        predictions: list[pd.DataFrame] = []
        paths: list[str] = []
        fold_ids = sorted(final_folds["fold"].astype(int).unique().tolist())
        if fold_ids != list(range(len(fold_ids))):
            raise ValueError(f"Folds finales no consecutivos: {fold_ids}")
        for fold in fold_ids:
            result = train_one_fold(
                cohort,
                final_folds,
                fold=fold,
                model_config=model_config,
                data_config=data_config,
                train_config=train_config,
                output_dir=run_dir / "cv5" / candidate_id / f"fold_{fold}",
                pca_variance=candidate.get("pca_variance"),
                fixed_epochs=fixed_epochs,
                device=device,
            )
            predictions.append(result.predictions)
            paths.append(
                result.checkpoint_path.resolve().relative_to(run_dir.resolve()).as_posix()
            )
        oof = pd.concat(predictions, ignore_index=True)
        oof = oof.merge(
            cohort[["uid", "acquisition_family", "background_qc_valid"]],
            on="uid",
            how="left",
            validate="one_to_one",
        )
        calibrated, final_temperature = cross_calibrate_oof(oof)
        raw_metrics = binary_metrics(calibrated, "probability")
        calibrated_metrics = binary_metrics(calibrated, "probability_cross_calibrated")
        subgroup_metrics = _background_subgroup_metrics(calibrated)
        diagnostics = _group_diagnostics(calibrated)
        worst_group = float(diagnostics["log_loss"].max()) if not diagnostics.empty else np.nan
        metric_records.append(
            {
                "candidate_id": candidate_id,
                "architecture": candidate["architecture"],
                "feature_variant": candidate["feature_variant"],
                "fixed_epochs": fixed_epochs,
                **{f"raw_{key}": value for key, value in raw_metrics.items()},
                **{f"cross_calibrated_{key}": value for key, value in calibrated_metrics.items()},
                **subgroup_metrics,
                "worst_family_log_loss_n10": worst_group,
                "final_temperature": final_temperature,
            }
        )
        candidate_dir = run_dir / "oof" / candidate_id
        atomic_csv(calibrated, candidate_dir / "oof_predictions.csv")
        atomic_csv(diagnostics, candidate_dir / "family_diagnostics.csv")
        atomic_json(
            {"temperature": final_temperature}, candidate_dir / "calibration.json"
        )
        oof_by_candidate[candidate_id] = calibrated
        model_paths[candidate_id] = paths
    metrics = pd.DataFrame.from_records(metric_records).sort_values(
        "cross_calibrated_log_loss"
    )
    metrics["sbr_promotion_pass"] = True
    non_sbr = metrics.loc[metrics["feature_variant"] != "image_radiomics_sbr"]
    for index, row in metrics.loc[
        metrics["feature_variant"] == "image_radiomics_sbr"
    ].iterrows():
        comparable = non_sbr.loc[non_sbr["architecture"] == row["architecture"]]
        if comparable.empty:
            comparable = non_sbr
        if comparable.empty:
            metrics.loc[index, "sbr_promotion_pass"] = False
            continue
        baseline = comparable.sort_values("cross_calibrated_log_loss").iloc[0]
        valid_improves = (
            np.isfinite(row["valid_background_log_loss"])
            and np.isfinite(baseline["valid_background_log_loss"])
            and row["valid_background_log_loss"] < baseline["valid_background_log_loss"]
        )
        global_safe = (
            row["cross_calibrated_log_loss"]
            <= baseline["cross_calibrated_log_loss"] + 0.005
        )
        metrics.loc[index, "sbr_promotion_pass"] = bool(valid_improves and global_safe)
    atomic_csv(metrics, run_dir / "finalist_cv5_metrics.csv")
    if metrics.empty:
        raise RuntimeError("No se evaluo ningun finalista.")
    eligible = metrics.loc[metrics["sbr_promotion_pass"].astype(bool)]
    if eligible.empty:
        eligible = metrics
    primary_id = str(eligible.iloc[0]["candidate_id"])
    primary_candidate = next(
        candidate for candidate in finalists if str(candidate["candidate_id"]) == primary_id
    )
    primary_temperature = float(metrics.iloc[0]["final_temperature"])
    primary_manifest = {
        "candidate_id": primary_id,
        "architecture": primary_candidate["architecture"],
        "feature_variant": primary_candidate["feature_variant"],
        "model_config": primary_candidate["model_config"],
        "train_config": primary_candidate["train_config"],
        "data_config": asdict(data_config),
        "pca_variance": primary_candidate.get("pca_variance"),
        "fixed_epochs": primary_candidate["fixed_epochs"],
        "fold_checkpoints": model_paths[primary_id],
        "n_fold_models": len(model_paths[primary_id]),
        "temperature": primary_temperature,
        "inference_rule": "temperature_scale_each_fold_logit_then_mean_probabilities",
        "acquisition_family_role": "audit_and_splits_only_never_model_input",
        "sbr_promotion_rule": (
            "valid-background log loss must improve and global log loss may degrade <=0.005"
        ),
    }
    atomic_json(primary_manifest, run_dir / "primary_ensemble_manifest.json")
    return FinalEvaluationResult(metrics, primary_manifest, oof_by_candidate[primary_id])
