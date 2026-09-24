from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from modeling.cnn.calibration import binary_metrics, cross_calibrate_oof
from modeling.cnn.io import atomic_csv, atomic_json

from .config import AugmentationConfig, ExperimentConfig, ModelConfig, TrainConfig
from .training import train_one_fold


@dataclass
class FinalEvaluationResult:
    metrics: pd.DataFrame
    oof: pd.DataFrame
    deployment_manifest: dict[str, Any]


def run_finalist_cv5(
    cohort: pd.DataFrame,
    folds: pd.DataFrame,
    finalists: list[dict[str, Any]],
    *,
    experiment: ExperimentConfig,
    run_dir: Path,
    cache_root: Path,
    upstream_hash: str,
    device: str | None = None,
) -> FinalEvaluationResult:
    run_dir.mkdir(parents=True, exist_ok=True)
    all_predictions: list[pd.DataFrame] = []
    metric_records: list[dict[str, Any]] = []
    checkpoint_map: dict[str, list[str]] = {}
    for candidate in finalists:
        candidate_id = str(candidate["candidate_id"])
        model = ModelConfig(**candidate["model_config"])
        train_payload = dict(candidate["train_config"])
        train_payload["augmentation"] = AugmentationConfig(**train_payload["augmentation"])
        train = TrainConfig(**train_payload)
        predictions: list[pd.DataFrame] = []
        checkpoints: list[str] = []
        for fold in range(experiment.search.n_splits_final):
            result = train_one_fold(
                cohort,
                folds,
                fold=fold,
                model_config=model,
                data_config=experiment.data,
                train_config=train,
                cache_root=cache_root / "final",
                output_dir=run_dir / "cv5" / candidate_id / f"fold_{fold}",
                upstream_hash=upstream_hash,
                feature_top_k=int(candidate["feature_top_k"]),
                pca_variance=candidate.get("pca_variance"),
                fixed_epochs=int(candidate["fixed_epochs"]),
                device=device,
            )
            frame = result.predictions.copy()
            frame["candidate_id"] = candidate_id
            frame["architecture"] = candidate["architecture"]
            frame["feature_variant"] = candidate["feature_variant"]
            frame["lateral_strategy"] = candidate["lateral_strategy"]
            predictions.append(frame)
            checkpoints.append(str(result.checkpoint_path.resolve()))
        oof = pd.concat(predictions, ignore_index=True)
        calibrated, temperature = cross_calibrate_oof(oof)
        metrics_raw = binary_metrics(calibrated, "probability")
        metrics_calibrated = binary_metrics(calibrated, "probability_cross_calibrated")
        metric_records.append(
            {
                "candidate_id": candidate_id,
                "architecture": candidate["architecture"],
                "feature_variant": candidate["feature_variant"],
                "lateral_strategy": candidate["lateral_strategy"],
                **{f"raw_{key}": value for key, value in metrics_raw.items()},
                **{f"calibrated_{key}": value for key, value in metrics_calibrated.items()},
                "deployment_temperature": temperature,
            }
        )
        all_predictions.append(calibrated)
        checkpoint_map[candidate_id] = checkpoints
        atomic_csv(calibrated, run_dir / f"oof_{candidate_id}.csv")
    metrics = pd.DataFrame(metric_records).sort_values("calibrated_log_loss").reset_index(drop=True)
    oof_all = pd.concat(all_predictions, ignore_index=True)
    primary_id = str(metrics.iloc[0]["candidate_id"])
    primary_candidate = next(item for item in finalists if item["candidate_id"] == primary_id)
    manifest = {
        "primary_candidate_id": primary_id,
        "primary_candidate": primary_candidate,
        "fold_checkpoints": checkpoint_map[primary_id],
        "n_fold_models": len(checkpoint_map[primary_id]),
        "inference_rule": "calibrate each fold logit then mean five probabilities",
        "selection_metric": "five-fold OOF cross-calibrated log loss",
        "acquisition_family_role": "split and subgroup audit only; never a predictor",
    }
    atomic_csv(metrics, run_dir / "final_metrics.csv")
    atomic_csv(oof_all, run_dir / "all_finalist_oof.csv")
    atomic_json(manifest, run_dir / "deployment_manifest.json")
    return FinalEvaluationResult(metrics, oof_all, manifest)
