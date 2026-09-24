from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .config import ExperimentConfig, stable_hash
from .data import prepare_image_cache
from .evaluation import FinalEvaluationResult, run_finalist_cv5
from .features import build_derived_feature_cache
from .folds import create_or_load_balanced_group_folds, fold_summary
from .io import atomic_csv, atomic_json
from .search import StagedSearchResult, run_staged_search

_SAFE_COMPONENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


def validate_path_component(value: str, name: str) -> str:
    if value in {".", ".."} or _SAFE_COMPONENT.fullmatch(value) is None:
        raise ValueError(
            f"{name} debe comenzar con letra o numero y usar solo letras, numeros, ., _ o -."
        )
    return value


@dataclass
class PreparedCNNExperiment:
    project_root: Path
    run_dir: Path
    node4_dir: Path
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


def _sha256_file(path: Path) -> str:
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
    features = pd.read_csv(features_path)
    features["uid"] = features["uid"].astype(str)
    if features["uid"].duplicated().any():
        raise ValueError("El nodo 04 contiene UIDs duplicados.")
    features["is_pathologic"] = features["is_pathologic"].astype(int)
    features["acquisition_family"] = features["acquisition_family"].astype(str)
    features["background_qc_valid"] = (
        features["background_qc_valid"].fillna(False).astype(bool)
    )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    crop_paths = sorted(crops_dir.glob("*.npz"), key=lambda path: path.name)
    crop_uids = {path.stem for path in crop_paths}
    missing = sorted(set(features["uid"]) - crop_uids)
    if missing:
        raise FileNotFoundError(f"Faltan {len(missing)} crops del nodo 04; ejemplo: {missing[0]}")
    # Hash the feature table contents and a lightweight crop inventory. The
    # latter catches normal recomputations without decompressing 1,362 NPZs.
    crop_inventory = [
        (path.name, path.stat().st_size, path.stat().st_mtime_ns) for path in crop_paths
    ]
    fingerprint = {
        "declared_config_hash": config.get("config_hash"),
        "config_file_sha256": _sha256_file(config_path),
        "features_file_sha256": _sha256_file(features_path),
        "crop_inventory_hash": stable_hash(crop_inventory),
        "n_crops": len(crop_paths),
    }
    upstream_hash = stable_hash(fingerprint)
    config = {**config, "resolved_cnn_upstream_fingerprint": fingerprint}
    return features, config, upstream_hash


def prepare_experiment(
    experiment: ExperimentConfig,
    *,
    project_root: Path | None = None,
) -> PreparedCNNExperiment:
    root = resolve_project_root(project_root)
    validate_path_component(experiment.run_id, "run_id")
    validate_path_component(experiment.data.node4_profile, "node4_profile")
    private_dir = root / "outputs" / "private_eda"
    node4_dir = private_dir / f"full_cohort_{experiment.data.node4_profile}"
    run_dir = private_dir / "cnn_runs" / experiment.run_id
    cache_root = private_dir / "cnn_cache" / experiment.data.node4_profile
    split_root = private_dir / "cnn_splits" / experiment.data.node4_profile
    run_dir.mkdir(parents=True, exist_ok=True)
    features, upstream_config, upstream_hash = _load_upstream(node4_dir)
    config_path = run_dir / "config" / "experiment_config.json"
    if config_path.exists():
        previous = json.loads(config_path.read_text(encoding="utf-8"))
        if previous.get("config_hash") != experiment.config_hash:
            raise RuntimeError(
                "Este run_id ya pertenece a otra configuracion CNN. Usa un run_id nuevo."
            )
        previous_upstream = previous.get("upstream_hash")
        if previous_upstream is not None and previous_upstream != upstream_hash:
            raise RuntimeError(
                "El nodo 04 cambio desde que se creo este run CNN. Usa un run_id nuevo "
                "para no mezclar derivados ni checkpoints incompatibles."
            )
    atomic_json(
        {
            **experiment.to_dict(),
            "config_hash": experiment.config_hash,
            "upstream_hash": upstream_hash,
            "privacy": "derived_private_outputs_only",
            "upstream_leakage_status": (
                "exploratory: node4 template/mask was frozen from the complete training cohort"
            ),
            "acquisition_family_role": "split_and_audit_only_never_predictor",
        },
        config_path,
    )
    crops_dir = node4_dir / "registered_crops"
    image_cache = prepare_image_cache(
        crops_dir,
        features[["uid"]],
        cache_root,
        config=experiment.data,
        upstream_hash=upstream_hash,
    )
    derived_key = stable_hash(
        {
            "upstream_hash": upstream_hash,
            "data": experiment.data.__dict__,
            "uids": sorted(features["uid"].tolist()),
        }
    )[:16]
    derived_path = cache_root / f"derived_radiomics_{derived_key}.csv"
    derived = build_derived_feature_cache(
        crops_dir,
        features[["uid"]],
        derived_path,
        config=experiment.data,
        upstream_hash=upstream_hash,
    )
    cohort = features.merge(derived, on="uid", how="inner", validate="one_to_one")
    cohort = cohort.merge(image_cache, on="uid", how="inner", validate="one_to_one")
    cohort["cnn_upstream_hash"] = upstream_hash
    if len(cohort) != len(features):
        raise RuntimeError("El cache CNN no cubre toda la cohorte del nodo 04.")
    split_root.mkdir(parents=True, exist_ok=True)
    search_folds = create_or_load_balanced_group_folds(
        cohort["uid"],
        cohort["is_pathologic"],
        cohort["acquisition_family"],
        split_root / "search_3fold_v1.csv",
        n_splits=experiment.search.n_splits_search,
        random_seed=experiment.train.seed,
        n_candidates=experiment.search.fold_candidates,
        background_valid=cohort["background_qc_valid"],
    )
    final_folds = create_or_load_balanced_group_folds(
        cohort["uid"],
        cohort["is_pathologic"],
        cohort["acquisition_family"],
        split_root / "evaluation_5fold_v1.csv",
        n_splits=experiment.search.n_splits_final,
        random_seed=experiment.train.seed + 104729,
        n_candidates=experiment.search.fold_candidates,
        background_valid=cohort["background_qc_valid"],
    )
    prepared_path = run_dir / "config" / "prepared_cohort_manifest.csv"
    atomic_csv(cohort, prepared_path)
    atomic_csv(fold_summary(search_folds), run_dir / "config" / "search_fold_summary.csv")
    atomic_csv(fold_summary(final_folds), run_dir / "config" / "final_fold_summary.csv")
    atomic_json(
        {
            "upstream_hash": upstream_hash,
            "upstream_config": upstream_config,
            "n_cases": len(cohort),
            "n_families": int(cohort["acquisition_family"].nunique()),
            "background_valid_fraction": float(cohort["background_qc_valid"].mean()),
            "forbidden_predictor_columns": [
                "acquisition_family",
                "spacing",
                "fov",
                "orientation",
                "registration_backend",
                "technical_qc",
            ],
        },
        run_dir / "config" / "upstream_contract.json",
    )
    return PreparedCNNExperiment(
        root,
        run_dir,
        node4_dir,
        cohort,
        search_folds,
        final_folds,
        upstream_hash,
    )


def run_search_stage(
    prepared: PreparedCNNExperiment,
    experiment: ExperimentConfig,
    *,
    device: str | None = None,
) -> StagedSearchResult:
    return run_staged_search(
        prepared.cohort,
        prepared.search_folds,
        run_dir=prepared.run_dir / "search",
        data_config=experiment.data,
        train_config=experiment.train,
        search_config=experiment.search,
        device=device,
    )


def load_finalists(prepared: PreparedCNNExperiment) -> list[dict[str, Any]]:
    path = prepared.run_dir / "search" / "finalists.json"
    if not path.exists():
        raise FileNotFoundError("Primero ejecuta la etapa search.")
    return list(json.loads(path.read_text(encoding="utf-8"))["finalists"])


def run_final_stage(
    prepared: PreparedCNNExperiment,
    experiment: ExperimentConfig,
    *,
    device: str | None = None,
) -> FinalEvaluationResult:
    return run_finalist_cv5(
        prepared.cohort,
        prepared.final_folds,
        load_finalists(prepared),
        run_dir=prepared.run_dir / "final",
        data_config=experiment.data,
        device=device,
    )
