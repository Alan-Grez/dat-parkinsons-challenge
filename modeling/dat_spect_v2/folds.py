from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from modeling.cnn.io import atomic_csv, atomic_json


def _cohort_hash(
    uids: Sequence[str],
    y: Sequence[int],
    families: Sequence[str],
    background_valid: Sequence[bool] | None,
) -> str:
    background = (
        np.zeros(len(uids), dtype=bool)
        if background_valid is None
        else np.asarray(background_valid, dtype=bool)
    )
    rows = sorted(
        zip(
            map(str, uids),
            map(int, y),
            map(str, families),
            map(int, background),
            strict=True,
        )
    )
    encoded = "\n".join(
        f"{uid}|{label}|{family}|{valid}"
        for uid, label, family, valid in rows
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def validate_patient_stratified_folds(
    manifest: pd.DataFrame,
    *,
    uids: Sequence[str],
    y: Sequence[int],
    families: Sequence[str],
    n_splits: int,
) -> None:
    required = {"uid", "is_pathologic", "acquisition_family", "fold"}
    missing = required - set(manifest)
    if missing:
        raise ValueError(f"Faltan columnas de folds: {sorted(missing)}")
    if manifest["uid"].astype(str).duplicated().any():
        raise ValueError("Cada paciente debe aparecer exactamente una vez.")
    expected = pd.DataFrame(
        {
            "uid": np.asarray(uids, dtype=str),
            "expected_y": np.asarray(y, dtype=int),
            "expected_family": np.asarray(families, dtype=str),
        }
    )
    observed = manifest.assign(uid=manifest["uid"].astype(str))
    merged = expected.merge(observed, on="uid", how="outer", indicator=True)
    if not (merged["_merge"] == "both").all():
        raise ValueError("El manifiesto no coincide con la cohorte.")
    if not np.array_equal(
        merged["expected_y"].to_numpy(dtype=int),
        merged["is_pathologic"].to_numpy(dtype=int),
    ):
        raise ValueError("Las etiquetas cambiaron desde que se congelaron los folds.")
    if not np.array_equal(
        merged["expected_family"].astype(str).to_numpy(),
        merged["acquisition_family"].astype(str).to_numpy(),
    ):
        raise ValueError("Las familias cambiaron desde que se congelaron los folds.")
    if sorted(observed["fold"].astype(int).unique()) != list(range(n_splits)):
        raise ValueError("Los identificadores de fold no son consecutivos.")
    if (observed.groupby("fold")["is_pathologic"].nunique() < 2).any():
        raise ValueError("Cada fold debe contener ambas clases.")
    sizes = observed.groupby("fold").size().to_numpy(dtype=int)
    if int(sizes.max() - sizes.min()) > 1:
        raise ValueError("Los folds de pacientes no estan balanceados en tamano.")


def _family_balance_penalty(manifest: pd.DataFrame, n_splits: int) -> float:
    counts = (
        manifest.groupby(["acquisition_family", "fold"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=range(n_splits), fill_value=0)
    )
    totals = counts.sum(axis=1).to_numpy(dtype=float)
    proportions = counts.to_numpy(dtype=float) / np.maximum(totals[:, None], 1.0)
    target = np.full_like(proportions, 1.0 / n_splits)
    # Large protocol families matter more, but a single giant family cannot dominate.
    weights = np.sqrt(totals)
    return float(np.average(np.mean(np.abs(proportions - target), axis=1), weights=weights))


def _assignment_score(manifest: pd.DataFrame, n_splits: int) -> float:
    prevalence_sd = float(
        manifest.groupby("fold")["is_pathologic"].mean().std(ddof=0)
    )
    background_sd = (
        float(manifest.groupby("fold")["background_qc_valid"].mean().std(ddof=0))
        if "background_qc_valid" in manifest
        else 0.0
    )
    return (
        3.0 * prevalence_sd
        + 1.5 * background_sd
        + _family_balance_penalty(manifest, n_splits)
    )


def create_or_load_patient_stratified_folds(
    uids: Sequence[str],
    y: Sequence[int],
    families: Sequence[str],
    manifest_path: Path,
    *,
    n_splits: int,
    random_seed: int,
    n_candidates: int = 256,
    background_valid: Sequence[bool] | None = None,
) -> pd.DataFrame:
    """Freeze balanced patient folds while spreading acquisition families.

    Acquisition family is used only to select the most balanced assignment among
    label-stratified candidates. It is never passed to a predictive model.
    """
    uid_array = np.asarray(uids, dtype=str)
    y_array = np.asarray(y, dtype=int)
    family_array = np.asarray(families, dtype=str)
    cohort_hash = _cohort_hash(uid_array, y_array, family_array, background_valid)
    metadata_path = manifest_path.with_suffix(".meta.json")
    if manifest_path.exists():
        manifest = pd.read_csv(manifest_path)
        validate_patient_stratified_folds(
            manifest,
            uids=uid_array,
            y=y_array,
            families=family_array,
            n_splits=n_splits,
        )
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata.get("cohort_hash") != cohort_hash:
                raise RuntimeError("Los folds congelados pertenecen a otra cohorte.")
        return manifest.sort_values("uid").reset_index(drop=True)

    base = pd.DataFrame(
        {
            "uid": uid_array,
            "is_pathologic": y_array,
            "acquisition_family": family_array,
        }
    )
    if background_valid is not None:
        base["background_qc_valid"] = np.asarray(background_valid, dtype=bool)
    indices = np.arange(len(base))
    best: pd.DataFrame | None = None
    best_score = float("inf")
    for candidate in range(max(1, n_candidates)):
        splitter = StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=random_seed + candidate,
        )
        assignment = np.full(len(base), -1, dtype=int)
        for fold, (_, valid_indices) in enumerate(splitter.split(indices, y_array)):
            assignment[valid_indices] = fold
        trial = base.assign(fold=assignment)
        score = _assignment_score(trial, n_splits)
        if score < best_score:
            best = trial.copy()
            best_score = score
    if best is None:
        raise RuntimeError("No fue posible construir folds balanceados por paciente.")
    validate_patient_stratified_folds(
        best,
        uids=uid_array,
        y=y_array,
        families=family_array,
        n_splits=n_splits,
    )
    best = best.sort_values("uid").reset_index(drop=True)
    atomic_csv(best, manifest_path)
    atomic_json(
        {
            "cohort_hash": cohort_hash,
            "n_splits": n_splits,
            "random_seed": random_seed,
            "n_candidates": n_candidates,
            "assignment_score": best_score,
            "split_unit": "unique_patient",
            "stratification": "label with acquisition-family distribution balancing",
            "acquisition_family_role": "fold_balance_and_audit_only_not_a_predictor",
        },
        metadata_path,
    )
    return best

