from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from modeling.cnn import ExperimentConfig
from modeling.cnn.io import atomic_json, atomic_npz
from modeling.cnn.pipeline import prepare_experiment


def _synthetic_node4(root: Path) -> None:
    (root / "pyproject.toml").write_text("[project]\nname='synthetic'\nversion='0'\n")
    node4 = root / "outputs" / "private_eda" / "full_cohort_v3"
    crops = node4 / "registered_crops"
    crops.mkdir(parents=True)
    records = []
    rng = np.random.default_rng(20260821)
    for family_index in range(10):
        for member, label in enumerate((0, 0, 1, 1)):
            uid = f"case_{family_index:02d}_{member}"
            volume = rng.gamma(2.0 + label * 0.3, 1.0, size=(16, 16, 16)).astype(
                np.float32
            )
            target = np.zeros_like(volume, dtype=np.uint8)
            target[3:13, 3:13, 3:13] = 1
            atomic_npz(
                crops / f"{uid}.npz",
                volume_ratio=volume,
                volume_intensity01=volume,
                target_mask=target,
                crop_start_zyx=np.zeros(3, dtype=np.int32),
                spacing_xyz=np.full(3, 2.5, dtype=np.float32),
            )
            records.append(
                {
                    "uid": uid,
                    "is_pathologic": label,
                    "acquisition_family": f"AF{family_index:03d}",
                    "background_qc_valid": member == 0,
                    "semiquant_sbr": 2.0 + label,
                    "semiquant_right_sbr": 2.1 + label,
                    "semiquant_left_sbr": 1.9 + label,
                    "firstorder_ratio_mean": 1.5 + label,
                }
            )
    pd.DataFrame(records).to_csv(node4 / "dat_radiomics_features_full.csv", index=False)
    atomic_json(
        {"config_hash": "synthetic_node4_v1"},
        node4 / "registration_radiomics_full_config.json",
    )


def test_prepare_pipeline_is_complete_grouped_and_idempotent(tmp_path: Path) -> None:
    _synthetic_node4(tmp_path)
    base = ExperimentConfig(run_id="synthetic_prepare")
    experiment = replace(
        base,
        data=replace(
            base.data,
            image_shape_3d=(16, 20, 16),
            view_size_2d=20,
            texture_levels=8,
            cache_checkpoint_every=7,
        ),
        train=replace(base.train, num_workers=0),
        search=replace(base.search, fold_candidates=12),
    )

    first = prepare_experiment(experiment, project_root=tmp_path)
    second = prepare_experiment(experiment, project_root=tmp_path)

    assert len(first.cohort) == 40
    assert first.cohort["cache_path"].map(lambda value: Path(value).exists()).all()
    assert "core_texture3d_contrast" in first.cohort
    assert "core_highuptake_sphericity" in first.cohort
    assert not any(column.startswith("stability_") for column in first.cohort)
    assert first.upstream_hash == second.upstream_hash
    assert first.cohort["uid"].tolist() == second.cohort["uid"].tolist()
    for manifest in (first.search_folds, first.final_folds):
        assert manifest.groupby("acquisition_family")["fold"].nunique().eq(1).all()
        assert manifest.groupby("fold")["is_pathologic"].nunique().eq(2).all()
