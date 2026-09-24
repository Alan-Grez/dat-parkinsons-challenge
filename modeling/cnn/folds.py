from __future__ import annotations

import hashlib
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from .io import atomic_csv, atomic_json


def _cohort_hash(uids: Sequence[str], y: Sequence[int], groups: Sequence[str]) -> str:
    rows = sorted(zip(map(str, uids), map(int, y), map(str, groups), strict=True))
    encoded = "\n".join(f"{uid}|{label}|{group}" for uid, label, group in rows)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _background_hash(
    uids: Sequence[str], background_valid: Sequence[bool] | None
) -> str | None:
    if background_valid is None:
        return None
    rows = sorted(zip(map(str, uids), map(bool, background_valid), strict=True))
    encoded = "\n".join(f"{uid}|{int(value)}" for uid, value in rows)
    return hashlib.sha256(encoded.encode("ascii")).hexdigest()


def validate_fold_manifest(
    manifest: pd.DataFrame,
    *,
    uids: Sequence[str],
    y: Sequence[int],
    groups: Sequence[str],
    n_splits: int,
) -> None:
    required = {"uid", "is_pathologic", "acquisition_family", "fold"}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"Faltan columnas de folds: {sorted(missing)}")
    if manifest["uid"].astype(str).duplicated().any():
        raise ValueError("Cada UID debe aparecer exactamente una vez en los folds.")
    expected = pd.DataFrame(
        {
            "uid": list(map(str, uids)),
            "is_pathologic_expected": list(map(int, y)),
            "acquisition_family_expected": list(map(str, groups)),
        }
    )
    observed = manifest.copy()
    observed["uid"] = observed["uid"].astype(str)
    merged = expected.merge(observed, on="uid", how="outer", indicator=True)
    if not (merged["_merge"] == "both").all():
        raise ValueError("El manifiesto de folds no coincide con la cohorte solicitada.")
    if not np.array_equal(
        merged["is_pathologic_expected"].to_numpy(dtype=int),
        merged["is_pathologic"].to_numpy(dtype=int),
    ):
        raise ValueError("Las etiquetas del manifiesto de folds cambiaron.")
    if not np.array_equal(
        merged["acquisition_family_expected"].astype(str).to_numpy(),
        merged["acquisition_family"].astype(str).to_numpy(),
    ):
        raise ValueError("Las familias del manifiesto de folds cambiaron.")
    fold_values = sorted(manifest["fold"].astype(int).unique().tolist())
    if fold_values != list(range(n_splits)):
        raise ValueError(f"Se esperaban folds 0..{n_splits - 1}, se obtuvo {fold_values}.")
    family_fold_count = manifest.groupby("acquisition_family")["fold"].nunique()
    if int(family_fold_count.max()) != 1:
        raise ValueError("Una familia de adquisicion aparece en mas de un fold.")
    class_counts = manifest.groupby("fold")["is_pathologic"].nunique()
    if (class_counts < 2).any():
        raise ValueError("Cada fold debe contener ambas clases.")


def _assignment_score(manifest: pd.DataFrame) -> float:
    summary = manifest.groupby("fold").agg(
        n=("uid", "size"),
        prevalence=("is_pathologic", "mean"),
        families=("acquisition_family", "nunique"),
    )
    size_cv = float(summary["n"].std(ddof=0) / max(summary["n"].mean(), 1.0))
    prevalence_sd = float(summary["prevalence"].std(ddof=0))
    family_cv = float(
        summary["families"].std(ddof=0) / max(summary["families"].mean(), 1.0)
    )
    score = size_cv + 2.5 * prevalence_sd + 0.25 * family_cv
    if "background_qc_valid" in manifest:
        background_sd = float(
            manifest.groupby("fold")["background_qc_valid"].mean().std(ddof=0)
        )
        score += 0.75 * background_sd
    return score


def create_or_load_balanced_group_folds(
    uids: Sequence[str],
    y: Sequence[int],
    groups: Sequence[str],
    manifest_path: Path,
    *,
    n_splits: int,
    random_seed: int,
    n_candidates: int = 256,
    background_valid: Sequence[bool] | None = None,
) -> pd.DataFrame:
    uids_array = np.asarray(uids, dtype=str)
    y_array = np.asarray(y, dtype=int)
    group_array = np.asarray(groups, dtype=str)
    cohort_hash = _cohort_hash(uids_array, y_array, group_array)
    background_hash = _background_hash(uids_array, background_valid)
    metadata_path = manifest_path.with_suffix(".meta.json")
    if manifest_path.exists():
        manifest = pd.read_csv(manifest_path)
        validate_fold_manifest(
            manifest,
            uids=uids_array,
            y=y_array,
            groups=group_array,
            n_splits=n_splits,
        )
        if metadata_path.exists():
            import json

            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata.get("cohort_hash") != cohort_hash:
                raise RuntimeError("Los folds existentes pertenecen a otra cohorte.")
            previous_background = metadata.get("background_valid_hash")
            if previous_background is not None and previous_background != background_hash:
                raise RuntimeError(
                    "La validez de fondo cambio desde que se congelaron los folds. "
                    "Versiona un nuevo manifiesto de folds."
                )
        return manifest.sort_values("uid").reset_index(drop=True)

    base = pd.DataFrame(
        {
            "uid": uids_array,
            "is_pathologic": y_array,
            "acquisition_family": group_array,
        }
    )
    if background_valid is not None:
        base["background_qc_valid"] = np.asarray(background_valid, dtype=bool)
    best: pd.DataFrame | None = None
    best_score = float("inf")
    indices = np.arange(len(base))
    for candidate in range(max(1, n_candidates)):
        splitter = StratifiedGroupKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=random_seed + candidate,
        )
        assignment = np.full(len(base), -1, dtype=int)
        for fold, (_, valid_indices) in enumerate(
            splitter.split(indices, y_array, group_array)
        ):
            assignment[valid_indices] = fold
        trial = base.assign(fold=assignment)
        try:
            validate_fold_manifest(
                trial,
                uids=uids_array,
                y=y_array,
                groups=group_array,
                n_splits=n_splits,
            )
        except ValueError:
            continue
        score = _assignment_score(trial)
        if score < best_score:
            best = trial.copy()
            best_score = score
    if best is None:
        raise RuntimeError("No fue posible construir folds agrupados con ambas clases.")
    best = best.sort_values("uid").reset_index(drop=True)
    atomic_csv(best, manifest_path)
    atomic_json(
        {
            "cohort_hash": cohort_hash,
            "background_valid_hash": background_hash,
            "n_splits": n_splits,
            "random_seed": random_seed,
            "n_candidates": n_candidates,
            "assignment_score": best_score,
            "group_column_role": "split_and_audit_only_not_a_predictor",
        },
        metadata_path,
    )
    return best


def fold_summary(manifest: pd.DataFrame) -> pd.DataFrame:
    aggregations: dict[str, tuple[str, str]] = {
        "n": ("uid", "size"),
        "pathologic": ("is_pathologic", "sum"),
        "prevalence": ("is_pathologic", "mean"),
        "families": ("acquisition_family", "nunique"),
    }
    if "background_qc_valid" in manifest:
        aggregations["background_valid_fraction"] = ("background_qc_valid", "mean")
    return manifest.groupby("fold").agg(**aggregations).reset_index()
