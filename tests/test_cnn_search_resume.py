from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from modeling.cnn import DataConfig, SearchConfig, TrainConfig
from modeling.cnn.search import run_staged_search


def test_staged_optuna_search_covers_six_variants_and_resumes(tmp_path, monkeypatch) -> None:
    calls: list[tuple[str, str, int]] = []

    def fake_train_one_fold(*_args, fold, model_config, **_kwargs):
        calls.append((model_config.architecture, model_config.feature_variant, fold))
        architecture_penalty = 0.01 if model_config.architecture == "3d" else 0.02
        variant_penalty = {
            "image_only": 0.03,
            "image_radiomics": 0.02,
            "image_radiomics_sbr": 0.01,
        }[model_config.feature_variant]
        return SimpleNamespace(
            validation_log_loss=0.5 + architecture_penalty + variant_penalty + fold / 100,
            selected_epoch=1,
        )

    monkeypatch.setattr("modeling.cnn.search.train_one_fold", fake_train_one_fold)
    cohort = pd.DataFrame({"uid": ["a", "b"]})
    folds = pd.DataFrame({"uid": ["a", "b"], "fold": [0, 1]})
    search = SearchConfig(
        n_splits_search=3,
        trials_image=1,
        trials_fusion=1,
        trials_sbr=1,
        finalists=3,
        pruning_startup_trials=10,
    )
    arguments = {
        "cohort": cohort,
        "search_folds": folds,
        "run_dir": tmp_path / "search",
        "data_config": DataConfig(),
        "train_config": TrainConfig(max_epochs_search=2),
        "search_config": search,
        "device": "cpu",
    }

    first = run_staged_search(**arguments)
    calls_after_first = len(calls)
    second = run_staged_search(**arguments)

    expected = {
        (architecture, variant)
        for architecture in ("2.5d", "3d")
        for variant in ("image_only", "image_radiomics", "image_radiomics_sbr")
    }
    assert set(zip(first.summary["architecture"], first.summary["feature_variant"])) == expected
    assert len(first.finalists) == 3
    assert first.database_path.exists()
    assert calls_after_first == 6 * 3
    assert len(calls) == calls_after_first
    pd.testing.assert_frame_equal(
        first.summary.reset_index(drop=True), second.summary.reset_index(drop=True)
    )
