from __future__ import annotations

import numpy as np
from sklearn.model_selection import StratifiedKFold

from modeling import FoldData, run_optuna_experiment


def test_optuna_models_resume_and_oof(tmp_path) -> None:
    rng = np.random.default_rng(20260821)
    X = rng.normal(size=(80, 6)).astype(np.float32)
    y = (X[:, 0] - 0.5 * X[:, 1] + rng.normal(scale=0.7, size=80) > 0).astype(int)
    uids = [f"case_{index:03d}" for index in range(len(y))]
    groups = [f"family_{index % 8}" for index in range(len(y))]
    splitter = StratifiedKFold(n_splits=2, shuffle=True, random_state=20260821)
    folds = [
        FoldData(
            fold=fold,
            train_indices=train,
            valid_indices=valid,
            X_train=X[train],
            X_valid=X[valid],
            y_train=y[train],
            y_valid=y[valid],
        )
        for fold, (train, valid) in enumerate(splitter.split(X, y))
    ]
    arguments = {
        "folds": folds,
        "uids": uids,
        "y": y,
        "groups": groups,
        "output_dir": tmp_path / "experiment",
        "experiment_config": {"test": "resume"},
        "n_trials_per_model": 1,
        "random_seed": 20260821,
        "cpu_jobs": 2,
        "prefer_gpu": True,
    }

    first = run_optuna_experiment(**arguments)
    second = run_optuna_experiment(**arguments)

    assert set(first.metrics["model"]) == {
        "logistic",
        "random_forest",
        "xgboost",
        "mean_ensemble",
    }
    assert len(first.oof_predictions) == len(y)
    assert first.oof_predictions.filter(like="p_").notna().all().all()
    assert second.tuning_summary.set_index("model")["completed_trials"].eq(1).all()
    assert second.xgboost_device == first.xgboost_device
