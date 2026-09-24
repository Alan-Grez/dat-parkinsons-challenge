from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from modeling.cnn.folds import fold_summary
from modeling.cnn.io import atomic_csv, atomic_json
from modeling.dat_spect_v2.evaluation import FinalEvaluationResult, run_finalist_cv5
from modeling.dat_spect_v2.folds import validate_patient_stratified_folds

from .config import RefinementExperiment
from .search import RefinementSearchResult, refinement_space, run_refinement_search
from .selection import select_node07_references

_SAFE_COMPONENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


@dataclass
class PreparedNode09:
    project_root: Path
    source_run_dir: Path
    run_dir: Path
    cache_root: Path
    cohort: pd.DataFrame
    search_folds: pd.DataFrame
    final_folds: pd.DataFrame
    references: list[dict[str, Any]]
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


def _load_source_config(source_run_dir: Path) -> dict[str, Any]:
    path = source_run_dir / "config" / "experiment_config.json"
    if not path.exists():
        raise FileNotFoundError(
            "No existe la configuracion del nodo 07 seleccionado: " f"{path}"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_folds(frame: pd.DataFrame, folds: pd.DataFrame, n_splits: int) -> None:
    validate_patient_stratified_folds(
        folds,
        uids=frame["uid"].astype(str),
        y=frame["is_pathologic"].astype(int),
        families=frame["acquisition_family"].astype(str),
        n_splits=n_splits,
    )


def prepare_refinement(
    experiment: RefinementExperiment,
    *,
    project_root: Path | None = None,
) -> PreparedNode09:
    root = resolve_project_root(project_root)
    _validate_component(experiment.run_id, "run_id")
    _validate_component(experiment.source_node07_run_id, "source_node07_run_id")
    source_run_dir = (
        root
        / "outputs"
        / "private_eda"
        / "node07_runs"
        / experiment.source_node07_run_id
    )
    source = _load_source_config(source_run_dir)
    source_data = dict(source.get("data", {}))
    if source_data != json.loads(json.dumps(asdict(experiment.data))):
        raise RuntimeError(
            "La configuracion de datos del nodo 09 no coincide con el nodo 07 fuente."
        )
    upstream_hash = str(source.get("upstream_hash", ""))
    if not upstream_hash:
        raise ValueError("El nodo 07 no registra upstream_hash.")
    cohort_path = source_run_dir / "config" / "prepared_cohort.csv"
    if not cohort_path.exists():
        raise FileNotFoundError(cohort_path)
    cohort = pd.read_csv(cohort_path)
    cohort["uid"] = cohort["uid"].astype(str)
    cohort["is_pathologic"] = cohort["is_pathologic"].astype(int)
    cohort["acquisition_family"] = cohort["acquisition_family"].astype(str)
    if cohort["uid"].duplicated().any():
        raise ValueError("El nodo 07 fuente contiene pacientes duplicados.")

    private_root = root / "outputs" / "private_eda"
    split_root = private_root / "node07_splits" / experiment.data.node4_profile
    search_fold_path = split_root / "search_patient_3fold_v3.csv"
    final_fold_path = split_root / "evaluation_patient_5fold_v3.csv"
    for path in (search_fold_path, final_fold_path):
        if not path.exists():
            raise FileNotFoundError(path)
    search_folds = pd.read_csv(search_fold_path)
    final_folds = pd.read_csv(final_fold_path)
    _validate_folds(cohort, search_folds, experiment.search.n_splits_search)
    _validate_folds(cohort, final_folds, experiment.search.n_splits_final)

    # This is deliberately evaluated after all source contracts. It refuses to
    # rank partial CV5 results or the less reliable 3-fold search scores.
    selected = select_node07_references(root, experiment.source_node07_run_id)
    references = [item.to_dict() for item in selected]
    if len(references) != 3 or len(
        {str(item["source_candidate_id"]) for item in references}
    ) != 3:
        raise RuntimeError("La seleccion del nodo 09 no produjo tres referencias unicas.")

    run_dir = private_root / "node09_runs" / experiment.run_id
    cache_root = private_root / "node07_cache" / experiment.data.node4_profile
    config_path = run_dir / "config" / "experiment_config.json"
    if config_path.exists():
        previous = json.loads(config_path.read_text(encoding="utf-8"))
        if previous.get("config_hash") != experiment.config_hash:
            raise RuntimeError(
                "El run_id del nodo 09 pertenece a otra configuracion; usa otro run_id."
            )
        if previous.get("upstream_hash") != upstream_hash:
            raise RuntimeError("El nodo 07 fuente cambio; usa otro run_id del nodo 09.")
    atomic_json(
        {
            **experiment.to_dict(),
            "config_hash": experiment.config_hash,
            "upstream_hash": upstream_hash,
            "source_node07_config_hash": source.get("config_hash"),
            "selection_contract": (
                "top-2 by node07 five-fold cross-calibrated OOF log loss plus best 3D; "
                "fill next-ranked candidate if the best 3D is already top-2"
            ),
            "search_contract": (
                "three independent local Optuna studies; 3-fold tuning then frozen 5-fold "
                "evaluation; exactly one winner per source branch"
            ),
        },
        config_path,
    )
    atomic_csv(cohort, run_dir / "config" / "prepared_cohort.csv")
    atomic_csv(fold_summary(search_folds), run_dir / "config" / "search_fold_summary.csv")
    atomic_csv(fold_summary(final_folds), run_dir / "config" / "final_fold_summary.csv")
    atomic_json({"references": references}, run_dir / "config" / "selected_references.json")
    atomic_json(
        {
            "spaces": {
                str(reference["source_candidate_id"]): refinement_space(reference)
                for reference in references
            }
        },
        run_dir / "config" / "refinement_spaces.json",
    )
    atomic_csv(
        pd.DataFrame(
            [
                {
                    "source_candidate_id": reference["source_candidate_id"],
                    "node07_rank": reference["node07_rank"],
                    "node07_calibrated_log_loss": reference[
                        "node07_calibrated_log_loss"
                    ],
                    "selection_roles": ",".join(reference["selection_roles"]),
                    "architecture": reference["architecture"],
                    "feature_variant": reference["feature_variant"],
                    "lateral_strategy": reference["lateral_strategy"],
                    "source_learning_rate": reference["train_config"]["learning_rate"],
                    "source_fixed_epochs": reference["fixed_epochs"],
                    "source_completed_trials": reference["trajectory_context"][
                        "completed_trials"
                    ],
                }
                for reference in references
            ]
        ),
        run_dir / "config" / "selected_references.csv",
    )
    atomic_json(
        {
            "n_cases": len(cohort),
            "n_unique_patients": int(cohort["uid"].nunique()),
            "n_source_models": 3,
            "n_search_folds": experiment.search.n_splits_search,
            "n_final_folds": experiment.search.n_splits_final,
            "completed_trials_target_per_model": experiment.search.trials_per_model,
            "maximum_search_epochs": experiment.search.max_epochs_search,
            "forbidden_predictors": ["acquisition_family", "spacing_x_mm", "spacing_y_mm"],
            "leakage_guard": (
                "node07 final metrics select branches only; selectors/scalers/PCA are fit "
                "inside each node09 fold"
            ),
        },
        run_dir / "config" / "data_contract.json",
    )
    return PreparedNode09(
        project_root=root,
        source_run_dir=source_run_dir,
        run_dir=run_dir,
        cache_root=cache_root,
        cohort=cohort,
        search_folds=search_folds,
        final_folds=final_folds,
        references=references,
        upstream_hash=upstream_hash,
    )


def run_search_stage(
    prepared: PreparedNode09,
    experiment: RefinementExperiment,
    *,
    device: str | None = None,
) -> RefinementSearchResult:
    return run_refinement_search(
        prepared.cohort,
        prepared.search_folds,
        prepared.references,
        experiment=experiment,
        run_dir=prepared.run_dir / "search",
        cache_root=prepared.cache_root,
        upstream_hash=prepared.upstream_hash,
        device=device,
    )


def load_finalists(prepared: PreparedNode09) -> list[dict[str, Any]]:
    path = prepared.run_dir / "search" / "finalists.json"
    if not path.exists():
        raise FileNotFoundError("Primero ejecuta la busqueda fina del nodo 09.")
    finalists = list(json.loads(path.read_text(encoding="utf-8"))["finalists"])
    if len(finalists) != 3:
        raise RuntimeError("El nodo 09 debe evaluar exactamente tres finalistas.")
    return finalists


def run_final_stage(
    prepared: PreparedNode09,
    experiment: RefinementExperiment,
    *,
    device: str | None = None,
) -> FinalEvaluationResult:
    return run_finalist_cv5(
        prepared.cohort,
        prepared.final_folds,
        load_finalists(prepared),
        experiment=experiment,  # type: ignore[arg-type]
        run_dir=prepared.run_dir / "final",
        cache_root=prepared.cache_root,
        upstream_hash=prepared.upstream_hash,
        device=device,
    )
