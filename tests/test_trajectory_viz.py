from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import optuna
import pandas as pd

from modeling.trajectory_viz import (
    acquisition_family_figure,
    calibration_figure,
    diagnostic_figure,
    final_fold_figure,
    final_metrics_figure,
    final_progress_figure,
    finalist_agreement_figure,
    learning_curve_figure,
    load_trajectory_snapshot,
    optimization_figure,
    probability_distribution_figure,
    write_dashboard,
)


def _stable_hash(payload: dict) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def test_live_optuna_snapshot_and_learning_diagnostic(tmp_path: Path) -> None:
    run_dir = (
        tmp_path
        / "outputs"
        / "private_eda"
        / "cnn_runs"
        / "synthetic_v1"
    )
    search_dir = run_dir / "search"
    database = search_dir / "optuna_cnn.sqlite3"
    database.parent.mkdir(parents=True)
    storage_url = f"sqlite:///{database.as_posix()}"
    study = optuna.create_study(
        study_name="cnn_25d_image_only",
        storage=storage_url,
        direction="minimize",
    )
    candidate = {
        "architecture": "2.5d",
        "feature_variant": "image_only",
        "model_config": {"architecture": "2.5d", "base_channels": 16},
        "train_config": {"learning_rate": 0.0003},
        "pca_variance": None,
    }

    def objective(trial: optuna.Trial) -> float:
        trial.suggest_int("base_channels", 16, 16)
        trial.set_user_attr(
            "candidate",
            {**candidate, "fold_log_loss": [0.55, 0.57, 0.56], "fixed_epochs": 5},
        )
        return 0.56

    study.optimize(objective, n_trials=1)
    parameter_hash = _stable_hash(candidate)[:20]
    history_dir = (
        search_dir
        / "trials"
        / "cnn_25d_image_only"
        / parameter_hash
        / "fold_0"
    )
    history_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "epoch": [0, 1, 2, 3, 4],
            "train_loss": [0.70, 0.62, 0.55, 0.48, 0.42],
            "validation_log_loss": [0.68, 0.60, 0.55, 0.58, 0.61],
            "learning_rate": [3e-4, 2.5e-4, 2e-4, 1.5e-4, 1e-4],
            "improved": [True, True, True, False, False],
        }
    ).to_csv(history_dir / "training_history.csv", index=False)

    snapshot = load_trajectory_snapshot(tmp_path)

    assert snapshot.catalog["run_key"].tolist() == ["node06:synthetic_v1"]
    assert snapshot.trials["state"].tolist() == ["COMPLETE"]
    assert snapshot.trials["parameter_hash"].tolist() == [parameter_hash]
    assert snapshot.histories["trial_number"].dropna().astype(int).unique().tolist() == [0]
    assert snapshot.fold_scores["fold_log_loss"].round(2).tolist() == [0.55, 0.57, 0.56]
    assert snapshot.diagnostics["diagnosis"].tolist() == ["posible sobreajuste"]

    assert len(optimization_figure(snapshot.trials).data) == 2
    assert len(learning_curve_figure(snapshot.histories).data) >= 3
    assert len(diagnostic_figure(snapshot.diagnostics).data) >= 1

    dashboard = write_dashboard(snapshot, tmp_path / "dashboard")
    assert dashboard.exists()
    assert (dashboard.parent / "trials_snapshot.csv").exists()
    assert "Trayectorias CNN" in dashboard.read_text(encoding="utf-8")


def test_node07_partial_and_complete_finalists_are_visualized(tmp_path: Path) -> None:
    run_dir = (
        tmp_path
        / "outputs"
        / "private_eda"
        / "node07_runs"
        / "node07_synthetic"
    )
    config_dir = run_dir / "config"
    search_dir = run_dir / "search"
    final_dir = run_dir / "final"
    config_dir.mkdir(parents=True)
    search_dir.mkdir(parents=True)
    (config_dir / "experiment_config.json").write_text(
        json.dumps({"search": {"n_splits_final": 5}}), encoding="utf-8"
    )
    uids = [f"case_{index:02d}" for index in range(20)]
    labels = [index % 2 for index in range(20)]
    pd.DataFrame(
        {
            "uid": uids,
            "is_pathologic": labels,
            "acquisition_family": ["AF_A"] * 10 + ["AF_B"] * 10,
            "background_qc_valid": [True] * 20,
        }
    ).to_csv(config_dir / "prepared_cohort.csv", index=False)
    finalists = [
        {
            "candidate_id": "slab_candidate_01",
            "study_name": "node07_slab2d_random_flip_image_radiomics_sbr",
            "architecture": "slab2d",
            "feature_variant": "image_radiomics_sbr",
            "lateral_strategy": "random_flip",
            "fixed_epochs": 3,
            "search_log_loss": 0.54,
        },
        {
            "candidate_id": "cnn3d_candidate_2",
            "study_name": "node07_3d_random_flip_image_radiomics_sbr",
            "architecture": "3d",
            "feature_variant": "image_radiomics_sbr",
            "lateral_strategy": "random_flip",
            "fixed_epochs": 3,
            "search_log_loss": 0.57,
        },
    ]
    (search_dir / "finalists.json").write_text(
        json.dumps({"finalists": finalists}), encoding="utf-8"
    )

    completed_oof: list[pd.DataFrame] = []
    for candidate_index, (candidate, completed_folds) in enumerate(
        ((finalists[0], 5), (finalists[1], 2))
    ):
        for fold in range(completed_folds):
            fold_dir = final_dir / "cv5" / candidate["candidate_id"] / f"fold_{fold}"
            fold_dir.mkdir(parents=True)
            pd.DataFrame(
                {
                    "epoch": [0, 1, 2],
                    "train_loss": [0.69, 0.61, 0.55],
                    "validation_log_loss": [np.nan, np.nan, 0.52 + 0.02 * fold],
                    "learning_rate": [3e-4, 2e-4, 1e-4],
                }
            ).to_csv(fold_dir / "history.csv", index=False)
            positions = list(range(fold * 4, fold * 4 + 4))
            probability = [
                0.22 + 0.03 * candidate_index if labels[index] == 0 else 0.78 - 0.03 * candidate_index
                for index in positions
            ]
            predictions = pd.DataFrame(
                {
                    "uid": [uids[index] for index in positions],
                    "is_pathologic": [labels[index] for index in positions],
                    "logit": np.log(np.asarray(probability) / (1 - np.asarray(probability))),
                    "probability": probability,
                    "fold": fold,
                    "selected_epoch": 2,
                }
            )
            predictions.to_csv(
                fold_dir / "validation_predictions.csv", index=False
            )
            if candidate_index == 0:
                completed_oof.append(predictions)
    oof = pd.concat(completed_oof, ignore_index=True)
    oof["probability_cross_calibrated"] = (
        0.5 + (oof["probability"] - 0.5) * 0.9
    )
    oof.to_csv(final_dir / "oof_slab_candidate_01.csv", index=False)

    snapshot = load_trajectory_snapshot(tmp_path)

    assert snapshot.catalog["run_key"].tolist() == ["node07:node07_synthetic"]
    assert len(snapshot.final_progress) == 10
    assert snapshot.final_progress["status"].value_counts().to_dict() == {
        "complete": 7,
        "pending": 3,
    }
    assert len(snapshot.final_predictions) == 28
    metrics = snapshot.final_metrics.set_index("candidate_id")
    assert bool(metrics.loc["slab_candidate_01", "evaluation_complete"])
    assert not bool(metrics.loc["cnn3d_candidate_2", "evaluation_complete"])
    assert pd.notna(metrics.loc["slab_candidate_01", "calibrated_log_loss"])
    assert pd.isna(metrics.loc["cnn3d_candidate_2", "calibrated_log_loss"])

    figures = [
        final_progress_figure(snapshot.final_progress),
        final_fold_figure(snapshot.final_progress),
        final_metrics_figure(snapshot.final_metrics),
        probability_distribution_figure(snapshot.final_predictions),
        calibration_figure(snapshot.final_predictions),
        finalist_agreement_figure(snapshot.final_predictions),
        acquisition_family_figure(snapshot.final_predictions),
        learning_curve_figure(
            snapshot.histories,
            "slab_candidate_01",
            run_key="node07:node07_synthetic",
            phase="final_cv5",
        ),
    ]
    assert all(figure.data for figure in figures)

    dashboard = write_dashboard(snapshot, tmp_path / "dashboard_node07")
    assert dashboard.exists()
    assert (dashboard.parent / "final_cv5_progress.csv").exists()
    assert (dashboard.parent / "final_oof_metrics.csv").exists()


def test_node09_optuna_run_is_discovered(tmp_path: Path) -> None:
    run_dir = (
        tmp_path
        / "outputs"
        / "private_eda"
        / "node09_runs"
        / "fine_synthetic"
    )
    database = run_dir / "search" / "optuna_node09.sqlite3"
    database.parent.mkdir(parents=True)
    study = optuna.create_study(
        study_name="node09_3d_source1234",
        storage=f"sqlite:///{database.as_posix()}",
        direction="minimize",
    )

    def objective(trial: optuna.Trial) -> float:
        trial.suggest_float("learning_rate", 5e-5, 2e-4, log=True)
        trial.set_user_attr(
            "candidate",
            {
                "architecture": "3d",
                "feature_variant": "image_radiomics_sbr",
                "lateral_strategy": "random_flip",
                "source_candidate_id": "source1234",
                "fold_log_loss": [0.54, 0.55, 0.56],
                "fixed_epochs": 12,
            },
        )
        return 0.55

    study.optimize(objective, n_trials=1)
    snapshot = load_trajectory_snapshot(
        tmp_path, run_keys=["node09:fine_synthetic"]
    )

    assert snapshot.catalog["run_key"].tolist() == ["node09:fine_synthetic"]
    assert snapshot.trials["node"].tolist() == ["node09"]
    assert snapshot.trials["architecture"].tolist() == ["3d"]
    assert snapshot.trials["study_label"].str.startswith("09 · ").all()


def test_node10_search_and_final_model_paths_are_visualized(tmp_path: Path) -> None:
    run_dir = tmp_path / "outputs" / "private_eda" / "node10_runs" / "hybrid_synthetic"
    config_dir = run_dir / "config"
    search_dir = run_dir / "search"
    final_dir = run_dir / "final"
    config_dir.mkdir(parents=True)
    (config_dir / "experiment_config.json").write_text(
        json.dumps({"search": {"n_splits_final": 2}}), encoding="utf-8"
    )
    uids = [f"case_{index}" for index in range(8)]
    labels = [index % 2 for index in range(8)]
    pd.DataFrame(
        {
            "uid": uids,
            "is_pathologic": labels,
            "acquisition_family": ["A"] * 4 + ["B"] * 4,
        }
    ).to_csv(config_dir / "prepared_cohort.csv", index=False)
    database = search_dir / "optuna_node10.sqlite3"
    database.parent.mkdir(parents=True)
    study = optuna.create_study(
        study_name="node10_dual_stream_multitask",
        storage=f"sqlite:///{database.as_posix()}",
        direction="minimize",
    )
    parameters = {"base_channels": 8, "learning_rate": 1e-4}
    candidate = {
        "family": "dual_stream_multitask",
        "parameters": parameters,
        "fold_log_loss": [0.55, 0.57, 0.56],
        "fixed_epochs": 3,
    }

    def objective(trial: optuna.Trial) -> float:
        trial.suggest_int("base_channels", 8, 8)
        trial.set_user_attr("candidate", candidate)
        return 0.56

    study.optimize(objective, n_trials=1)
    candidate_id = "dual_candidate_01"
    finalist = {
        **candidate,
        "candidate_id": candidate_id,
        "study_name": "node10_dual_stream_multitask",
        "search_log_loss": 0.56,
        "trial_number": 0,
    }
    (search_dir / "finalists.json").write_text(
        json.dumps({"finalists": [finalist]}), encoding="utf-8"
    )
    for fold in range(2):
        fold_dir = final_dir / "models" / "dual_stream_multitask" / f"fold_{fold}"
        fold_dir.mkdir(parents=True)
        pd.DataFrame(
            {
                "epoch": [0, 1, 2],
                "train_loss": [0.69, 0.62, 0.56],
                "validation_log_loss": [0.66, 0.59, 0.55],
            }
        ).to_csv(fold_dir / "history.csv", index=False)
        positions = list(range(fold * 4, fold * 4 + 4))
        pd.DataFrame(
            {
                "uid": [uids[index] for index in positions],
                "is_pathologic": [labels[index] for index in positions],
                "probability": [0.2 if labels[index] == 0 else 0.8 for index in positions],
                "fold": fold,
            }
        ).to_csv(fold_dir / "validation_predictions.csv", index=False)
    snapshot = load_trajectory_snapshot(tmp_path, run_keys=["node10:hybrid_synthetic"])
    assert snapshot.trials["architecture"].tolist() == ["3d_dual"]
    assert snapshot.trials["feature_variant"].tolist() == ["dual_stream_multitask"]
    assert snapshot.trials["study_label"].str.startswith("10 · ").all()
    assert len(snapshot.final_progress) == 2
    assert snapshot.final_progress["status"].eq("complete").all()
    assert snapshot.histories["phase"].eq("final_cv5").all()
