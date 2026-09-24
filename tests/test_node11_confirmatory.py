from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import optuna
import pandas as pd

from modeling.node11_confirmatory.config import ExperimentConfig, SearchConfig, TrainConfig
from modeling.node11_confirmatory.diagnostics import optuna_pairwise_scatter_figures
from modeling.node11_confirmatory.evaluation import run_final
from modeling.node11_confirmatory.hgb import HGBFoldResult, train_hgb_fold
from modeling.node11_confirmatory.pipeline import _merge_source_cohorts
from modeling.node11_confirmatory.search import _enqueue_failed_parameters_for_resume


def _cohort(n: int = 120) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    label = np.arange(n) % 2
    signal = label[:, None] * 0.45 + rng.normal(0, 1, size=(n, 8))
    frame = pd.DataFrame(
        {
            "uid": [f"case_{index:04d}" for index in range(n)],
            "is_pathologic": label,
            "acquisition_family": [f"AF{index % 9:02d}" for index in range(n)],
            "background_qc_valid": np.arange(n) % 3 != 0,
            "cache_path": ["unused.npz"] * n,
            "cnn_upstream_hash": ["synthetic"] * n,
        }
    )
    for column in range(signal.shape[1]):
        frame[f"regional930_f{column:03d}"] = signal[:, column]
    return frame


def _folds(frame: pd.DataFrame, n_splits: int) -> pd.DataFrame:
    result = frame[["uid", "is_pathologic", "acquisition_family", "background_qc_valid"]].copy()
    result["fold"] = np.arange(len(frame)) % n_splits
    return result


def test_default_contract_is_long_patient_and_requires_ten_complete_trials() -> None:
    experiment = ExperimentConfig()
    assert experiment.train.max_epochs == 500
    assert experiment.train.cnn_patience >= 60
    assert experiment.train.hgb_max_iter == 500
    assert experiment.search.effective_completed_trials >= 10
    assert experiment.regional_blend_weight == 0.5


def test_interrupted_trial_parameters_are_enqueued_once_for_resume() -> None:
    study = optuna.create_study(direction="minimize")
    failed = study.ask()
    failed.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    study.tell(failed, state=optuna.trial.TrialState.FAIL)

    _enqueue_failed_parameters_for_resume(study)
    _enqueue_failed_parameters_for_resume(study)

    waiting = [
        trial for trial in study.trials if trial.state == optuna.trial.TrialState.WAITING
    ]
    assert len(waiting) == 1
    assert waiting[0].system_attrs["fixed_params"] == failed.params


def test_source_merge_requires_identical_uids_and_labels() -> None:
    regional = _cohort(12).drop(columns=["cache_path", "cnn_upstream_hash"])
    cnn = _cohort(12)
    merged = _merge_source_cohorts(regional, cnn)
    assert len(merged) == 12
    assert merged["uid"].nunique() == 12
    assert {"cache_path", "regional930_f000"}.issubset(merged.columns)


def test_hgb_selects_internal_best_iteration_then_refits(tmp_path: Path) -> None:
    cohort = _cohort()
    folds = _folds(cohort, 3)
    train = TrainConfig(
        max_epochs=20,
        cnn_patience=5,
        hgb_max_iter=40,
        hgb_patience=6,
        num_workers=1,
    )
    result = train_hgb_fold(
        cohort,
        folds,
        fold=0,
        parameters={
            "learning_rate": 0.04,
            "max_leaf_nodes": 8,
            "min_samples_leaf": 8,
            "l2_regularization": 0.01,
        },
        train_config=train,
        output_dir=tmp_path / "hgb",
    )
    assert 1 <= result.best_iteration <= train.hgb_max_iter
    assert len(result.predictions) == len(cohort) // 3
    assert (tmp_path / "hgb" / "iteration_history.csv").exists()
    metadata = json.loads((tmp_path / "hgb" / "metadata.json").read_text())
    assert metadata["external_fold_used_for_early_stopping"] is False
    assert metadata["best_iteration"] == result.best_iteration


def test_final_stage_keeps_common_outer_fold_and_fixed_primary_weight(
    tmp_path: Path, monkeypatch
) -> None:
    cohort = _cohort(100)
    folds = _folds(cohort, 5)
    experiment = ExperimentConfig(
        train=TrainConfig(
            max_epochs=12,
            cnn_patience=4,
            hgb_max_iter=20,
            hgb_patience=4,
            num_workers=1,
        ),
        search=SearchConfig(
            n_splits_search=3,
            n_splits_final=5,
            inner_epoch_splits=4,
            inner_epoch_repeats=2,
            completed_trials_per_expert=10,
            fold_candidates=4,
        ),
    )

    def fake_hgb(
        cohort: pd.DataFrame,
        folds: pd.DataFrame,
        *,
        fold: int,
        output_dir: Path,
        **_: object,
    ) -> HGBFoldResult:
        valid = cohort.merge(folds[["uid", "fold"]], on="uid").query("fold == @fold")
        probability = np.where(valid["is_pathologic"].to_numpy() == 1, 0.72, 0.28)
        predictions = pd.DataFrame(
            {
                "uid": valid["uid"].astype(str),
                "is_pathologic": valid["is_pathologic"].astype(int),
                "probability": probability,
                "fold": fold,
                "selected_iteration": 7,
            }
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        return HGBFoldResult(fold, 0.3, 7, predictions, output_dir / "model.pkl")

    def fake_cnn(
        cohort: pd.DataFrame,
        fold_manifest: pd.DataFrame,
        *,
        fold: int,
        fixed_epochs: int | None,
        output_dir: Path,
        **_: object,
    ) -> SimpleNamespace:
        valid = cohort.merge(fold_manifest[["uid", "fold"]], on="uid").query("fold == @fold")
        probability = np.where(valid["is_pathologic"].to_numpy() == 1, 0.68, 0.32)
        predictions = pd.DataFrame(
            {
                "uid": valid["uid"].astype(str),
                "is_pathologic": valid["is_pathologic"].astype(int),
                "probability": probability,
                "fold": fold,
                "selected_epoch": 5,
            }
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        return SimpleNamespace(
            selected_epoch=4 if fixed_epochs is None else fixed_epochs - 1,
            validation_log_loss=0.4,
            predictions=predictions,
        )

    monkeypatch.setattr("modeling.node11_confirmatory.evaluation.train_hgb_fold", fake_hgb)
    monkeypatch.setattr("modeling.node11_confirmatory.evaluation.train_one_fold", fake_cnn)
    finalists = {
        "regional_hgb": {
            "candidate_id": "regional",
            "parameters": {
                "learning_rate": 0.03,
                "max_leaf_nodes": 16,
                "min_samples_leaf": 40,
                "l2_regularization": 0.01,
            },
        },
        "cnn_25d_image_only": {
            "candidate_id": "cnn",
            "parameters": {
                "base_channels": 12,
                "image_embedding_dim": 128,
                "pooling": "gap_max",
                "dropout": 0.35,
                "learning_rate": 1e-4,
                "weight_decay": 1e-3,
                "consistency_weight": 0.01,
                "noise_std": 0.02,
                "rotation_degrees": 5.0,
            },
        },
    }
    result = run_final(
        cohort,
        folds,
        finalists,
        experiment=experiment,
        run_dir=tmp_path / "final",
        device="cpu",
    )
    assert set(result.metrics["family"]) == {
        "regional_hgb",
        "cnn_25d_image_only",
        "fixed_raw_50_50",
    }
    assert result.deployment_manifest["regional_blend_weight"] == 0.5
    assert result.deployment_manifest["same_outer_folds"] is True
    assert result.deployment_manifest["outer_fold_used_for_early_stopping"] is False
    assert len(result.oof_predictions) == 3 * len(cohort)


def test_optuna_diagnostic_builds_one_scatter_matrix_per_study(tmp_path: Path) -> None:
    studies = tmp_path / "search" / "studies"
    studies.mkdir(parents=True)
    pd.DataFrame(
        {
            "trial": [0, 1, 2],
            "state": ["COMPLETE"] * 3,
            "objective": [0.54, 0.52, 0.53],
            "param__learning_rate": [1e-4, 8e-5, 1.2e-4],
            "param__dropout": [0.30, 0.35, 0.40],
            "param__pooling": ["gap_max", "gap_gem", "gap_max"],
        }
    ).to_csv(studies / "node11_test_trials.csv", index=False)
    figures = optuna_pairwise_scatter_figures(tmp_path)
    assert set(figures) == {"node11_test"}
    assert figures["node11_test"].data[0].type == "splom"
