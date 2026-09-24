from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from modeling.cnn.folds import create_or_load_balanced_group_folds, fold_summary
from modeling.cnn.io import atomic_csv, atomic_json

from .config import ExperimentConfig, stable_hash
from .evaluation import FinalEvaluationResult, run_finalist_cv5
from .folds import create_or_load_patient_stratified_folds
from .preprocessing import audit_base_cache, prepare_base_cache
from .search import SearchResult, run_staged_search

_SAFE_COMPONENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


@dataclass
class PreparedNode07:
    project_root: Path
    node4_dir: Path
    run_dir: Path
    cache_root: Path
    cohort: pd.DataFrame
    search_folds: pd.DataFrame
    final_folds: pd.DataFrame
    family_stress_folds: pd.DataFrame
    upstream_hash: str


def resolve_project_root(path: Path | None = None) -> Path:
    root = (path or Path.cwd()).resolve()
    if root.name.lower() == "notebooks":
        root = root.parent
    if not (root / "pyproject.toml").exists():
        raise FileNotFoundError(f"No se encontro pyproject.toml en {root}")
    return root


def _validate_component(value: str, name: str) -> None:
    if value in {".", ".."} or _SAFE_COMPONENT.fullmatch(value) is None:
        raise ValueError(f"{name} contiene caracteres no permitidos.")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_upstream(node4_dir: Path) -> tuple[pd.DataFrame, dict[str, Any], str]:
    features_path = node4_dir / "dat_radiomics_features_full.csv"
    config_path = node4_dir / "registration_radiomics_full_config.json"
    crops_dir = node4_dir / "registered_crops"
    for required in (features_path, config_path, crops_dir):
        if not required.exists():
            raise FileNotFoundError(required)
    frame = pd.read_csv(features_path)
    frame["uid"] = frame["uid"].astype(str)
    frame["is_pathologic"] = frame["is_pathologic"].astype(int)
    frame["acquisition_family"] = frame["acquisition_family"].astype(str)
    frame["background_qc_valid"] = frame["background_qc_valid"].fillna(False).astype(bool)
    if frame["uid"].duplicated().any():
        raise ValueError("El nodo 04 contiene UIDs duplicados.")
    crop_paths = sorted(crops_dir.glob("*.npz"), key=lambda value: value.name)
    missing = sorted(set(frame["uid"]) - {path.stem for path in crop_paths})
    if missing:
        raise FileNotFoundError(f"Faltan crops del nodo 04; ejemplo: {missing[0]}")
    upstream_config = json.loads(config_path.read_text(encoding="utf-8"))
    fingerprint = {
        "config_sha256": _sha256(config_path),
        "features_sha256": _sha256(features_path),
        "crop_inventory": stable_hash(
            [(path.name, path.stat().st_size, path.stat().st_mtime_ns) for path in crop_paths]
        ),
        "n_crops": len(crop_paths),
    }
    return frame, upstream_config, stable_hash(fingerprint)


def prepare_experiment(
    experiment: ExperimentConfig,
    *,
    project_root: Path | None = None,
) -> PreparedNode07:
    root = resolve_project_root(project_root)
    _validate_component(experiment.run_id, "run_id")
    _validate_component(experiment.data.node4_profile, "node4_profile")
    private = root / "outputs" / "private_eda"
    node4_dir = private / f"full_cohort_{experiment.data.node4_profile}"
    run_dir = private / "node07_runs" / experiment.run_id
    cache_root = private / "node07_cache" / experiment.data.node4_profile
    split_root = private / "node07_splits" / experiment.data.node4_profile
    run_dir.mkdir(parents=True, exist_ok=True)
    frame, upstream_config, upstream_hash = _load_upstream(node4_dir)
    config_path = run_dir / "config" / "experiment_config.json"
    if config_path.exists():
        previous = json.loads(config_path.read_text(encoding="utf-8"))
        if previous.get("config_hash") != experiment.config_hash:
            raise RuntimeError(
                "El run_id del nodo 07 pertenece a otra configuracion; usa otro run_id."
            )
        if previous.get("upstream_hash") != upstream_hash:
            raise RuntimeError("El nodo 04 cambio; usa otro run_id del nodo 07.")
    atomic_json(
        {
            **experiment.to_dict(),
            "config_hash": experiment.config_hash,
            "upstream_hash": upstream_hash,
            "privacy": "derived private outputs only; raw NIfTI are never read by node 07",
            "acquisition_family_role": "fold balance and audit only; forbidden predictor",
            "scientific_contract": {
                "self_normalization": "sqrt(d)*x/||x||2 on positive voxels inside a fixed physical striatal ROI with bounded label-independent recentering",
                "slab": "six 2-mm axial samples averaged to 12 mm",
                "multitemplate": "fold-specific label-independent prototypes plus bounded affine refinement",
                "multitemplate_equivalence": "engineering approximation; not an exact SPM12 linear-template implementation",
                "regional_features": "930 fixed definitions; selector/scaler/PCA fit inside each fold",
            },
        },
        config_path,
    )
    base = prepare_base_cache(
        node4_dir / "registered_crops",
        frame[["uid"]],
        cache_root,
        config=experiment.data,
        upstream_hash=upstream_hash,
    )
    base_dir = Path(base["base_cache_path"].iloc[0]).parent
    base_audit = audit_base_cache(
        base,
        config=experiment.data,
        destination=base_dir / "base_cache_audit.csv",
    )
    failed_base = base_audit.loc[~base_audit["node07_base_qc_pass"]]
    if not failed_base.empty:
        raise RuntimeError(
            f"El cache base del nodo 07 fallo QC en {len(failed_base)} casos; "
            f"ejemplo: {failed_base.iloc[0]['uid']}"
        )
    cohort = (
        frame.merge(base, on="uid", how="inner", validate="one_to_one")
        .merge(base_audit, on="uid", how="inner", validate="one_to_one")
    )
    cohort["node07_upstream_hash"] = upstream_hash
    if len(cohort) != len(frame):
        raise RuntimeError("El cache base del nodo 07 no cubre la cohorte completa.")
    split_root.mkdir(parents=True, exist_ok=True)
    search_folds = create_or_load_patient_stratified_folds(
        cohort["uid"],
        cohort["is_pathologic"],
        cohort["acquisition_family"],
        split_root / "search_patient_3fold_v3.csv",
        n_splits=experiment.search.n_splits_search,
        random_seed=experiment.train.seed,
        n_candidates=experiment.search.fold_candidates,
        background_valid=cohort["background_qc_valid"],
    )
    final_folds = create_or_load_patient_stratified_folds(
        cohort["uid"],
        cohort["is_pathologic"],
        cohort["acquisition_family"],
        split_root / "evaluation_patient_5fold_v3.csv",
        n_splits=experiment.search.n_splits_final,
        random_seed=experiment.train.seed + 104729,
        n_candidates=experiment.search.fold_candidates,
        background_valid=cohort["background_qc_valid"],
    )
    # Secondary stress test: complete acquisition families are held out. Because
    # one protocol family has 460 cases, these folds are intentionally not used
    # as the primary balanced CV.
    family_stress_folds = create_or_load_balanced_group_folds(
        cohort["uid"],
        cohort["is_pathologic"],
        cohort["acquisition_family"],
        split_root / "family_holdout_stress_3fold_v3.csv",
        n_splits=3,
        random_seed=experiment.train.seed + 15485863,
        n_candidates=experiment.search.fold_candidates,
        background_valid=cohort["background_qc_valid"],
    )
    atomic_csv(cohort, run_dir / "config" / "prepared_cohort.csv")
    atomic_csv(fold_summary(search_folds), run_dir / "config" / "search_fold_summary.csv")
    atomic_csv(fold_summary(final_folds), run_dir / "config" / "final_fold_summary.csv")
    atomic_csv(
        fold_summary(family_stress_folds),
        run_dir / "config" / "family_stress_fold_summary.csv",
    )
    atomic_json(
        {
            "n_cases": len(cohort),
            "n_families": int(cohort["acquisition_family"].nunique()),
            "background_valid_fraction": float(cohort["background_qc_valid"].mean()),
            "base_cache_qc_pass_fraction": float(cohort["node07_base_qc_pass"].mean()),
            "upstream_config": upstream_config,
            "forbidden_predictors": [
            "acquisition_family",
                "spacing_x_mm",
                "spacing_y_mm",
                "spacing_z_mm",
                "fov_x_mm",
                "fov_y_mm",
                "fov_z_mm",
                "registration_backend",
                "technical_outlier_score",
            ],
            "primary_cv": "unique-patient label-stratified folds selected for family/background balance",
            "secondary_stress_test": "family-disjoint 3-fold manifest; descriptive because a 460-case family prevents balanced 5-fold CV",
        },
        run_dir / "config" / "data_contract.json",
    )
    return PreparedNode07(
        root,
        node4_dir,
        run_dir,
        cache_root,
        cohort,
        search_folds,
        final_folds,
        family_stress_folds,
        upstream_hash,
    )


def run_search_stage(
    prepared: PreparedNode07,
    experiment: ExperimentConfig,
    *,
    device: str | None = None,
) -> SearchResult:
    return run_staged_search(
        prepared.cohort,
        prepared.search_folds,
        experiment=experiment,
        run_dir=prepared.run_dir / "search",
        cache_root=prepared.cache_root,
        upstream_hash=prepared.upstream_hash,
        device=device,
    )


def load_finalists(prepared: PreparedNode07) -> list[dict[str, Any]]:
    path = prepared.run_dir / "search" / "finalists.json"
    if not path.exists():
        raise FileNotFoundError("Primero ejecuta la busqueda del nodo 07.")
    return list(json.loads(path.read_text(encoding="utf-8"))["finalists"])


def run_final_stage(
    prepared: PreparedNode07,
    experiment: ExperimentConfig,
    *,
    device: str | None = None,
) -> FinalEvaluationResult:
    return run_finalist_cv5(
        prepared.cohort,
        prepared.final_folds,
        load_finalists(prepared),
        experiment=experiment,
        run_dir=prepared.run_dir / "final",
        cache_root=prepared.cache_root,
        upstream_hash=prepared.upstream_hash,
        device=device,
    )
