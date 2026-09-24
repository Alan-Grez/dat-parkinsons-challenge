from __future__ import annotations

import json
from dataclasses import replace

from modeling.cnn import ExperimentConfig
from modeling.cnn.pipeline import _load_upstream


def test_extending_optuna_budget_keeps_resume_contract() -> None:
    baseline = ExperimentConfig(run_id="resume_test")
    extended = replace(
        baseline,
        search=replace(
            baseline.search,
            trials_image=50,
            trials_fusion=50,
            trials_sbr=50,
            optuna_timeout_seconds=86_400,
        ),
    )
    assert baseline.config_hash == extended.config_hash


def test_changing_model_or_fold_contract_changes_hash() -> None:
    baseline = ExperimentConfig(run_id="resume_test")
    changed_folds = replace(
        baseline,
        search=replace(baseline.search, n_splits_final=4),
    )
    changed_shape = replace(
        baseline,
        data=replace(baseline.data, image_shape_3d=(48, 64, 48)),
    )
    assert baseline.config_hash != changed_folds.config_hash
    assert baseline.config_hash != changed_shape.config_hash


def test_upstream_fingerprint_changes_with_derived_features(tmp_path) -> None:
    node4 = tmp_path / "full_cohort_v3"
    crops = node4 / "registered_crops"
    crops.mkdir(parents=True)
    (crops / "case_1.npz").write_bytes(b"derived crop inventory")
    (node4 / "registration_radiomics_full_config.json").write_text(
        json.dumps({"config_hash": "declared"}), encoding="utf-8"
    )
    features = (
        "uid,is_pathologic,acquisition_family,background_qc_valid,value\n"
        "case_1,0,AF001,True,1.0\n"
    )
    path = node4 / "dat_radiomics_features_full.csv"
    path.write_text(features, encoding="utf-8")
    _frame, _config, first = _load_upstream(node4)

    path.write_text(features.replace("1.0", "2.0"), encoding="utf-8")
    _frame, _config, second = _load_upstream(node4)

    assert first != second
