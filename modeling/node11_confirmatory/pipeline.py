from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from modeling.cnn.config import ExperimentConfig as SourceCNNExperiment
from modeling.cnn.folds import fold_summary
from modeling.cnn.io import atomic_csv, atomic_json
from modeling.cnn.pipeline import prepare_experiment as prepare_source_cnn
from modeling.dat_spect_v2.config import stable_hash
from modeling.dat_spect_v2.folds import create_or_load_patient_stratified_folds
from modeling.node10_hybrid.config import ExperimentConfig as SourceRegionalExperiment
from modeling.node10_hybrid.pipeline import prepare_experiment as prepare_source_regional

from .config import ExperimentConfig
from .evaluation import FinalResult, run_final
from .search import SearchResult, run_search

_SAFE_COMPONENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


@dataclass
class PreparedNode11:
    project_root: Path
    run_dir: Path
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


def _validate_component(value: str, name: str) -> None:
    if value in {".", ".."} or _SAFE_COMPONENT.fullmatch(value) is None:
        raise ValueError(f"{name} contiene caracteres no permitidos.")


def _merge_source_cohorts(regional: pd.DataFrame, cnn: pd.DataFrame) -> pd.DataFrame:
    left = regional.copy()
    right_columns = ["uid", "is_pathologic", "cache_path", "cnn_upstream_hash"]
    missing = set(right_columns) - set(cnn)
    if missing:
        raise ValueError(f"El origen CNN no contiene {sorted(missing)}.")
    right = cnn[right_columns].copy().rename(columns={"is_pathologic": "cnn_label"})
    left["uid"] = left["uid"].astype(str)
    right["uid"] = right["uid"].astype(str)
    merged = left.merge(right, on="uid", how="inner", validate="one_to_one")
    if len(merged) != len(left) or len(merged) != len(right):
        raise RuntimeError("Los nodos 06 y 10 no contienen exactamente la misma cohorte.")
    if not (merged["is_pathologic"].astype(int) == merged["cnn_label"].astype(int)).all():
        raise RuntimeError("Las etiquetas difieren entre los orígenes de los nodos 06 y 10.")
    merged = merged.drop(columns="cnn_label")
    if merged["uid"].duplicated().any():
        raise RuntimeError("La cohorte combinada contiene UIDs duplicados.")
    return merged.sort_values("uid").reset_index(drop=True)


def prepare_experiment(
    experiment: ExperimentConfig,
    *,
    project_root: Path | None = None,
) -> PreparedNode11:
    root = resolve_project_root(project_root)
    for value, name in (
        (experiment.run_id, "run_id"),
        (experiment.source_node06_run_id, "source_node06_run_id"),
        (experiment.source_node10_run_id, "source_node10_run_id"),
    ):
        _validate_component(value, name)
    source_cnn = prepare_source_cnn(
        SourceCNNExperiment(
            run_id=experiment.source_node06_run_id,
            data=experiment.cnn_data,
        ),
        project_root=root,
    )
    source_regional = prepare_source_regional(
        SourceRegionalExperiment(
            run_id=experiment.source_node10_run_id,
            data=experiment.regional_data,
        ),
        project_root=root,
    )
    cohort = _merge_source_cohorts(source_regional.cohort, source_cnn.cohort)
    upstream_hash = stable_hash(
        {
            "node06": source_cnn.upstream_hash,
            "node10": source_regional.upstream_hash,
            "uids": sorted(cohort["uid"].astype(str)),
        }
    )
    run_dir = root / "outputs" / "private_eda" / "node11_runs" / experiment.run_id
    config_dir = run_dir / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / "experiment_config.json"
    if config_path.exists():
        previous = json.loads(config_path.read_text(encoding="utf-8"))
        if previous.get("config_hash") != experiment.config_hash:
            raise RuntimeError("El run_id del nodo 11 pertenece a otra configuracion.")
        if previous.get("upstream_hash") != upstream_hash:
            raise RuntimeError("Cambio un origen del nodo 11; usa un run_id nuevo.")
    atomic_json(
        {
            **experiment.to_dict(),
            "config_hash": experiment.config_hash,
            "upstream_hash": upstream_hash,
            "source_node06_upstream_hash": source_cnn.upstream_hash,
            "source_node10_upstream_hash": source_regional.upstream_hash,
            "scientific_contract": {
                "experts": "regional_hgb plus cnn_25d_image_only",
                "same_folds": "both search and final predictions use identical UID manifests",
                "same_seed": experiment.train.seed,
                "cnn_horizon": experiment.train.max_epochs,
                "cnn_best_epoch": (
                    "selected on train-only inner folds and refit on complete outer train"
                ),
                "hgb_best_iteration": (
                    "selected on train-only internal validation and refit on complete outer train"
                ),
                "primary_blend": (
                    f"raw probabilities; regional={experiment.regional_blend_weight:.2f}; "
                    f"cnn={1.0 - experiment.regional_blend_weight:.2f}; calibrate once after blend"
                ),
                "optuna_completion": (
                    "at least 10 COMPLETE trials per expert; PRUNED trials are replaced"
                ),
            },
        },
        config_path,
    )
    search_folds = create_or_load_patient_stratified_folds(
        cohort["uid"],
        cohort["is_pathologic"],
        cohort["acquisition_family"],
        config_dir / "search_common_3fold.csv",
        n_splits=experiment.search.n_splits_search,
        random_seed=experiment.train.seed,
        n_candidates=experiment.search.fold_candidates,
        background_valid=cohort["background_qc_valid"],
    )
    final_folds = create_or_load_patient_stratified_folds(
        cohort["uid"],
        cohort["is_pathologic"],
        cohort["acquisition_family"],
        config_dir / "final_common_5fold.csv",
        n_splits=experiment.search.n_splits_final,
        random_seed=experiment.train.seed + 104_729,
        n_candidates=experiment.search.fold_candidates,
        background_valid=cohort["background_qc_valid"],
    )
    atomic_csv(cohort, config_dir / "prepared_cohort.csv")
    atomic_csv(fold_summary(search_folds), config_dir / "search_fold_summary.csv")
    atomic_csv(fold_summary(final_folds), config_dir / "final_fold_summary.csv")
    atomic_json(
        {
            "n_cases": len(cohort),
            "n_unique_uids": int(cohort["uid"].nunique()),
            "n_regional_features": len(
                [column for column in cohort if column.startswith("regional930_")]
            ),
            "image_cache": "node06 physical registered crop cache",
            "regional_cache": "node10 fold-safe regional930 feature cache",
            "acquisition_family_role": "fold balance and post-OOF audit only",
            "outer_validation_used_for_early_stopping": False,
        },
        config_dir / "data_contract.json",
    )
    return PreparedNode11(root, run_dir, cohort, search_folds, final_folds, upstream_hash)


def run_search_stage(
    prepared: PreparedNode11,
    experiment: ExperimentConfig,
    *,
    device: str | None = None,
) -> SearchResult:
    return run_search(
        prepared.cohort,
        prepared.search_folds,
        experiment=experiment,
        run_dir=prepared.run_dir / "search",
        device=device,
    )


def load_finalists(prepared: PreparedNode11) -> dict[str, dict[str, Any]]:
    path = prepared.run_dir / "search" / "finalists.json"
    if not path.exists():
        raise FileNotFoundError("Primero ejecuta la busqueda Optuna del nodo 11.")
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {str(key): dict(value) for key, value in payload["finalists"].items()}


def run_final_stage(
    prepared: PreparedNode11,
    experiment: ExperimentConfig,
    *,
    device: str | None = None,
) -> FinalResult:
    return run_final(
        prepared.cohort,
        prepared.final_folds,
        load_finalists(prepared),
        experiment=experiment,
        run_dir=prepared.run_dir / "final",
        device=device,
    )
