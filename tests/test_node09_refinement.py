from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from modeling.dat_spect_v2.config import ModelConfig, TrainConfig
from modeling.node09_refinement import RefinementExperiment, refinement_space
from modeling.node09_refinement.search import run_refinement_search
from modeling.node09_refinement.selection import select_node07_references


def _candidate(candidate_id: str, architecture: str, study_name: str) -> dict:
    model = ModelConfig(
        architecture=architecture,
        feature_variant="image_radiomics_sbr",
        lateral_strategy="random_flip",
        pooling="gmp" if architecture == "slab2d" else "gap_max",
        tabular_embedding_dim=24,
        dropout=0.30,
    )
    train = TrainConfig(
        learning_rate=3.2e-4,
        weight_decay=3e-6,
        max_epochs_search=18,
        patience=5,
    )
    return {
        "candidate_id": candidate_id,
        "architecture": architecture,
        "feature_variant": "image_radiomics_sbr",
        "lateral_strategy": "random_flip",
        "study_name": study_name,
        "model_config": asdict(model),
        "train_config": asdict(train),
        "feature_top_k": 64,
        "pca_variance": None,
        "fixed_epochs": 6,
        "search_log_loss": 0.57,
    }


def test_selects_top_two_cv5_plus_best_3d(monkeypatch, tmp_path: Path) -> None:
    source_dir = (
        tmp_path / "outputs" / "private_eda" / "node07_runs" / "source_v1"
    )
    (source_dir / "search").mkdir(parents=True)
    (source_dir / "final").mkdir(parents=True)
    candidates = [
        _candidate("slab", "slab2d", "node07_slab2d_random_flip_image_radiomics_sbr"),
        _candidate("twod", "2.5d", "node07_25d_random_flip_image_radiomics_sbr"),
        _candidate("three", "3d", "node07_3d_random_flip_image_radiomics_sbr"),
    ]
    (source_dir / "search" / "finalists.json").write_text(
        json.dumps({"finalists": candidates}), encoding="utf-8"
    )
    pd.DataFrame(
        {
            "candidate_id": ["slab", "three", "twod"],
            "architecture": ["slab2d", "3d", "2.5d"],
            "calibrated_log_loss": [0.51, 0.52, 0.53],
        }
    ).to_csv(source_dir / "final" / "final_metrics.csv", index=False)
    trial_rows = []
    for candidate in candidates:
        trial_rows.append(
            {
                "study_name": candidate["study_name"],
                "state": "COMPLETE",
                "objective_log_loss": 0.55,
                "param__fusion_learning_rate": 1.8e-4,
                "param__fusion_dropout": 0.28,
            }
        )
    monkeypatch.setattr(
        "modeling.node09_refinement.selection.load_trajectory_snapshot",
        lambda *_args, **_kwargs: SimpleNamespace(trials=pd.DataFrame(trial_rows)),
    )

    selected = select_node07_references(tmp_path, "source_v1")

    assert [item.candidate["candidate_id"] for item in selected] == [
        "slab",
        "three",
        "twod",
    ]
    assert selected[0].selection_roles == ("top_1_overall",)
    assert selected[1].selection_roles == ("top_2_overall", "best_3d")
    assert selected[2].selection_roles == ("diversity_fill",)
    assert selected[0].trajectory_context["parameter_centers"][
        "fusion_learning_rate"
    ] == 1.8e-4


def test_refinement_space_lowers_learning_rate() -> None:
    reference = {
        **_candidate("slab", "slab2d", "source_study"),
        "source_candidate_id": "slab",
        "trajectory_context": {
            "parameter_centers": {
                "fusion_learning_rate": 1.8e-4,
                "fusion_weight_decay": 2e-5,
                "fusion_dropout": 0.27,
            }
        },
    }
    space = refinement_space(reference)
    assert 0 < space["learning_rate"][0] < space["learning_rate"][1]
    assert space["learning_rate"][1] < reference["train_config"]["learning_rate"]
    assert reference["model_config"]["tabular_embedding_dim"] in space[
        "tabular_embedding_dim"
    ]
    assert reference["feature_top_k"] in space["feature_top_k"]


def test_three_studies_resume_to_completed_trial_target(monkeypatch, tmp_path: Path) -> None:
    references = []
    for index, architecture in enumerate(("slab2d", "2.5d", "3d")):
        candidate = _candidate(
            f"source_{index}", architecture, f"node07_{architecture}_{index}"
        )
        references.append(
            {
                **candidate,
                "source_candidate_id": candidate["candidate_id"],
                "node07_rank": index + 1,
                "node07_calibrated_log_loss": 0.51 + index * 0.01,
                "selection_roles": ["test"],
                "trajectory_context": {"parameter_centers": {}},
            }
        )

    def fake_train(*_args, fold: int, train_config: TrainConfig, **_kwargs):
        return SimpleNamespace(
            validation_log_loss=0.50 + fold * 0.01 + train_config.learning_rate,
            selected_epoch=9,
        )

    monkeypatch.setattr(
        "modeling.node09_refinement.search.train_one_fold", fake_train
    )
    experiment = RefinementExperiment(run_id="synthetic")
    experiment = replace(
        experiment,
        search=replace(
            experiment.search,
            trials_per_model=2,
            pruning_startup_trials=2,
            maximum_attempt_multiplier=3,
        ),
    )
    arguments = {
        "cohort": pd.DataFrame({"uid": ["a", "b"]}),
        "folds": pd.DataFrame({"uid": ["a", "b"], "fold": [0, 1]}),
        "references": references,
        "experiment": experiment,
        "run_dir": tmp_path / "search",
        "cache_root": tmp_path / "cache",
        "upstream_hash": "upstream",
        "device": "cpu",
    }
    first = run_refinement_search(**arguments)
    second = run_refinement_search(**arguments)

    assert len(first.finalists) == 3
    assert len(second.finalists) == 3
    assert first.summary["completed_trials"].tolist() == [2, 2, 2]
    assert second.summary["completed_trials"].tolist() == [2, 2, 2]
    assert all(candidate["fixed_epochs"] >= 8 for candidate in first.finalists)

