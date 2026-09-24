from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
import pytest

from modeling.cnn import (
    create_or_load_balanced_group_folds,
    validate_fold_manifest,
)
from modeling.cnn.pipeline import validate_path_component


def _synthetic_cohort() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create balanced labels while keeping each acquisition family indivisible."""

    uids: list[str] = []
    labels: list[int] = []
    groups: list[str] = []
    for group_index in range(20):
        group = f"AF{group_index + 1:03d}"
        group_labels = [0, 0, 0, 1] if group_index % 2 == 0 else [1, 1, 1, 0]
        for case_index, label in enumerate(group_labels):
            uids.append(f"case_{group_index:02d}_{case_index}")
            labels.append(label)
            groups.append(group)
    return (
        np.asarray(uids, dtype=str),
        np.asarray(labels, dtype=np.int64),
        np.asarray(groups, dtype=str),
    )


def _sorted_manifest(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.sort_values("uid").reset_index(drop=True)


def test_balanced_group_folds_cover_cohort_and_keep_groups_disjoint(tmp_path) -> None:
    uids, y, groups = _synthetic_cohort()
    manifest_path = tmp_path / "final_folds_5_v1.csv"

    manifest = create_or_load_balanced_group_folds(
        uids=uids,
        y=y,
        groups=groups,
        manifest_path=manifest_path,
        n_splits=5,
        random_seed=20260821,
    )
    validate_fold_manifest(
        manifest,
        uids=uids,
        y=y,
        groups=groups,
        n_splits=5,
    )

    assert manifest_path.is_file()
    assert {"uid", "is_pathologic", "acquisition_family", "fold"} <= set(
        manifest.columns
    )
    assert len(manifest) == len(uids)
    assert manifest["uid"].is_unique
    assert set(manifest["uid"].astype(str)) == set(uids)
    assert set(manifest["fold"].astype(int)) == set(range(5))

    # A technical acquisition family must never appear in two validation folds.
    folds_per_group = manifest.groupby("acquisition_family")["fold"].nunique()
    assert folds_per_group.eq(1).all()

    # The synthetic construction permits every fold to contain useful support
    # for both classes without requiring exact equality.
    fold_summary = manifest.groupby("fold")["is_pathologic"].agg(["count", "mean"])
    assert (fold_summary["count"] > 0).all()
    assert fold_summary["mean"].between(0.25, 0.75).all()
    assert int(fold_summary["count"].max() - fold_summary["count"].min()) <= 4


def test_fold_manifest_resume_is_idempotent(tmp_path) -> None:
    uids, y, groups = _synthetic_cohort()
    manifest_path = tmp_path / "search_folds_3_v1.csv"
    arguments = {
        "uids": uids,
        "y": y,
        "groups": groups,
        "manifest_path": manifest_path,
        "n_splits": 3,
        "random_seed": 20260821,
    }

    first = create_or_load_balanced_group_folds(**arguments)
    first_bytes = manifest_path.read_bytes()
    first_hash = hashlib.sha256(first_bytes).hexdigest()

    second = create_or_load_balanced_group_folds(**arguments)
    second_bytes = manifest_path.read_bytes()

    pd.testing.assert_frame_equal(_sorted_manifest(first), _sorted_manifest(second))
    assert second_bytes == first_bytes
    assert hashlib.sha256(second_bytes).hexdigest() == first_hash


def test_fold_validator_rejects_group_leakage(tmp_path) -> None:
    uids, y, groups = _synthetic_cohort()
    manifest = create_or_load_balanced_group_folds(
        uids=uids,
        y=y,
        groups=groups,
        manifest_path=tmp_path / "folds.csv",
        n_splits=5,
        random_seed=20260821,
    )
    corrupted = manifest.copy()
    group = str(corrupted.iloc[0]["acquisition_family"])
    group_rows = corrupted.index[corrupted["acquisition_family"].astype(str) == group]
    corrupted.loc[group_rows[0], "fold"] = (
        int(corrupted.loc[group_rows[0], "fold"]) + 1
    ) % 5

    with pytest.raises(ValueError):
        validate_fold_manifest(
            corrupted,
            uids=uids,
            y=y,
            groups=groups,
            n_splits=5,
        )


@pytest.mark.parametrize("value", ["../escape", "..", ".", "bad/name", " bad"])
def test_path_component_rejects_unsafe_values(value: str) -> None:
    with pytest.raises(ValueError):
        validate_path_component(value, "run_id")


@pytest.mark.parametrize("value", ["cnn_compact_v1", "v3", "trial-01.alpha"])
def test_path_component_accepts_safe_values(value: str) -> None:
    assert validate_path_component(value, "run_id") == value
