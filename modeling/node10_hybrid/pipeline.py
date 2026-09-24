from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from modeling.cnn.folds import fold_summary
from modeling.cnn.io import atomic_csv, atomic_json
from modeling.dat_spect_v2.folds import create_or_load_patient_stratified_folds

from .config import ExperimentConfig
from .evaluation import FinalResult, run_final_cv5
from .features import build_hybrid_cache
from .search import SearchResult, run_hybrid_search

_SAFE_COMPONENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


@dataclass
class PreparedNode10:
    project_root: Path
    node4_dir: Path
    run_dir: Path
    cache_root: Path
    cohort: pd.DataFrame
    search_folds: pd.DataFrame
    final_folds: pd.DataFrame
    upstream_hash: str


def resolve_project_root(path: Path | None = None) -> Path:
    root = (path or Path.cwd()).resolve()
    if root.name.lower() == "notebooks":
        root = root.parent
    if not (root / "pyproject.toml").exists():
        raise FileNotFoundError(f"No se encontro pyproject.toml en {root}")
    return root


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_upstream(node4_dir: Path) -> tuple[pd.DataFrame, str]:
    features_path = node4_dir / "dat_radiomics_features_full.csv"
    config_path = node4_dir / "registration_radiomics_full_config.json"
    crops_dir = node4_dir / "registered_crops"
    for path in (features_path, config_path, crops_dir):
        if not path.exists():
            raise FileNotFoundError(path)
    frame = pd.read_csv(features_path)
    frame["uid"] = frame["uid"].astype(str)
    frame["is_pathologic"] = frame["is_pathologic"].astype(int)
    frame["acquisition_family"] = frame["acquisition_family"].astype(str)
    frame["background_qc_valid"] = frame["background_qc_valid"].fillna(False).astype(bool)
    if frame["uid"].duplicated().any():
        raise ValueError("El nodo 04 contiene UIDs duplicados.")
    crops = sorted(crops_dir.glob("*.npz"), key=lambda value: value.name)
    missing = sorted(set(frame["uid"]) - {path.stem for path in crops})
    if missing:
        raise FileNotFoundError(f"Falta crop de nodo 04 para {missing[0]}")
    fingerprint = {
        "features_sha256": _sha256(features_path),
        "config_sha256": _sha256(config_path),
        "crop_inventory": [
            (path.name, path.stat().st_size, path.stat().st_mtime_ns) for path in crops
        ],
    }
    from modeling.dat_spect_v2.config import stable_hash

    return frame, stable_hash(fingerprint)


def prepare_experiment(
    experiment: ExperimentConfig,
    *,
    project_root: Path | None = None,
) -> PreparedNode10:
    root = resolve_project_root(project_root)
    if _SAFE_COMPONENT.fullmatch(experiment.run_id) is None:
        raise ValueError("run_id invalido.")
    private = root / "outputs" / "private_eda"
    node4_dir = private / f"full_cohort_{experiment.data.node4_profile}"
    run_dir = private / "node10_runs" / experiment.run_id
    cache_root = private / "node10_cache" / experiment.data.node4_profile
    split_root = private / "node10_splits" / experiment.data.node4_profile
    run_dir.mkdir(parents=True, exist_ok=True)
    frame, upstream_hash = _load_upstream(node4_dir)
    config_path = run_dir / "config" / "experiment_config.json"
    if config_path.exists():
        previous = json.loads(config_path.read_text(encoding="utf-8"))
        if previous.get("config_hash") != experiment.config_hash:
            raise RuntimeError("El run_id del nodo 10 pertenece a otra configuracion.")
        if previous.get("upstream_hash") != upstream_hash:
            raise RuntimeError("Cambio el nodo 04; usa otro run_id del nodo 10.")
    atomic_json(
        {
            **experiment.to_dict(),
            "config_hash": experiment.config_hash,
            "upstream_hash": upstream_hash,
            "scientific_contract": {
                "intensity_branch": (
                    "registered volume_intensity01 x foreground_p99_registered; clipped raw-count "
                    "magnitude retained and scaled from each fold training partition only"
                ),
                "pattern_branch": "same physical crop projected with sqrt(d)*x/||x||2",
                "auxiliary_heads": "six regional uptake proxies, affected side, asymmetry, fragmentation",
                "graph": "12 fixed region-by-z-band nodes; no voxel graph and no torch-geometric dependency",
                "fold_safety": "all scalers, selectors, diffusion maps and calibration fit inside training folds",
                "optuna_completion": "at least 10 COMPLETE trials per base family and stacking; PRUNED trials are replaced",
            },
        },
        config_path,
    )
    manifest, topology, regional, graph = build_hybrid_cache(
        node4_dir / "registered_crops",
        frame[["uid", "foreground_p99_registered"]],
        cache_root,
        config=experiment.data,
        upstream_hash=upstream_hash,
    )
    cohort = (
        frame.merge(manifest, on="uid", validate="one_to_one")
        .merge(topology, on="uid", validate="one_to_one")
        .merge(regional, on="uid", validate="one_to_one")
        .merge(graph, on="uid", validate="one_to_one")
    )
    if len(cohort) != len(frame):
        raise RuntimeError("El nodo 10 no conserva la cohorte completa.")
    split_root.mkdir(parents=True, exist_ok=True)
    search_folds = create_or_load_patient_stratified_folds(
        cohort["uid"],
        cohort["is_pathologic"],
        cohort["acquisition_family"],
        split_root / "search_patient_3fold_v1.csv",
        n_splits=experiment.search.n_splits_search,
        random_seed=experiment.train.seed,
        n_candidates=experiment.search.fold_candidates,
        background_valid=cohort["background_qc_valid"],
    )
    final_folds = create_or_load_patient_stratified_folds(
        cohort["uid"],
        cohort["is_pathologic"],
        cohort["acquisition_family"],
        split_root / "evaluation_patient_5fold_v1.csv",
        n_splits=experiment.search.n_splits_final,
        random_seed=experiment.train.seed + 104729,
        n_candidates=experiment.search.fold_candidates,
        background_valid=cohort["background_qc_valid"],
    )
    atomic_csv(cohort, run_dir / "config" / "prepared_cohort.csv")
    atomic_csv(fold_summary(search_folds), run_dir / "config" / "search_fold_summary.csv")
    atomic_csv(fold_summary(final_folds), run_dir / "config" / "final_fold_summary.csv")
    atomic_json(
        {
            "n_cases": len(cohort),
            "n_unique_uids": int(cohort["uid"].nunique()),
            "n_families": int(cohort["acquisition_family"].nunique()),
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
            "acquisition_family_role": "fold construction and subgroup audit only",
            "regional_proxy_warning": "fixed atlas proxies are not clinical segmentations",
        },
        run_dir / "config" / "data_contract.json",
    )
    return PreparedNode10(
        root, node4_dir, run_dir, cache_root, cohort, search_folds, final_folds, upstream_hash
    )


def run_search_stage(
    prepared: PreparedNode10,
    experiment: ExperimentConfig,
    *,
    device: str | None = None,
) -> SearchResult:
    return run_hybrid_search(
        prepared.cohort,
        prepared.search_folds,
        experiment=experiment,
        run_dir=prepared.run_dir / "search",
        device=device,
    )


def load_finalists(prepared: PreparedNode10) -> list[dict[str, Any]]:
    path = prepared.run_dir / "search" / "finalists.json"
    if not path.exists():
        raise FileNotFoundError("Primero ejecuta la busqueda del nodo 10.")
    return list(json.loads(path.read_text(encoding="utf-8"))["finalists"])


def run_final_stage(
    prepared: PreparedNode10,
    experiment: ExperimentConfig,
    *,
    device: str | None = None,
) -> FinalResult:
    return run_final_cv5(
        prepared.cohort,
        prepared.final_folds,
        load_finalists(prepared),
        experiment=experiment,
        run_dir=prepared.run_dir / "final",
        device=device,
    )
