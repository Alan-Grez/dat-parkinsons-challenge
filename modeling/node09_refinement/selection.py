from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from modeling.trajectory_viz import load_trajectory_snapshot


@dataclass(frozen=True)
class SelectedReference:
    candidate: dict[str, Any]
    selection_roles: tuple[str, ...]
    node07_rank: int
    node07_calibrated_log_loss: float
    trajectory_context: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.candidate,
            "source_candidate_id": str(self.candidate["candidate_id"]),
            "selection_roles": list(self.selection_roles),
            "node07_rank": self.node07_rank,
            "node07_calibrated_log_loss": self.node07_calibrated_log_loss,
            "trajectory_context": self.trajectory_context,
        }


def _read_source_candidates(source_run_dir: Path) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    metrics_path = source_run_dir / "final" / "final_metrics.csv"
    finalists_path = source_run_dir / "search" / "finalists.json"
    if not finalists_path.exists():
        raise FileNotFoundError("El nodo 07 aun no tiene finalistas de busqueda.")
    if not metrics_path.exists():
        raise RuntimeError(
            "El nodo 07 aun no termino la evaluacion CV5 de sus tres finalistas. "
            "Espera a que exista final/final_metrics.csv antes de seleccionar el nodo 09."
        )
    metrics = pd.read_csv(metrics_path)
    required = {"candidate_id", "architecture", "calibrated_log_loss"}
    missing = required.difference(metrics.columns)
    if missing:
        raise ValueError(f"Faltan columnas en final_metrics.csv: {sorted(missing)}")
    metrics["candidate_id"] = metrics["candidate_id"].astype(str)
    metrics["calibrated_log_loss"] = pd.to_numeric(
        metrics["calibrated_log_loss"], errors="coerce"
    )
    if not np.isfinite(metrics["calibrated_log_loss"]).all():
        raise ValueError("Las metricas CV5 del nodo 07 contienen log loss no finito.")
    candidates = list(
        json.loads(finalists_path.read_text(encoding="utf-8"))["finalists"]
    )
    candidate_ids = {str(item["candidate_id"]) for item in candidates}
    missing_candidates = sorted(set(metrics["candidate_id"]) - candidate_ids)
    if missing_candidates:
        raise ValueError(
            "Las metricas del nodo 07 no coinciden con finalists.json: "
            f"{missing_candidates}"
        )
    ordered = metrics.sort_values("calibrated_log_loss").reset_index(drop=True)
    return ordered, candidates


def _trajectory_context(
    project_root: Path,
    source_run_id: str,
    candidate: dict[str, Any],
) -> dict[str, Any]:
    """Summarize the strongest local trajectory without fitting on CV5 labels."""

    snapshot = load_trajectory_snapshot(
        project_root, run_keys=[f"node07:{source_run_id}"]
    )
    trials = snapshot.trials
    if trials.empty:
        return {"completed_trials": 0, "top_trials": 0, "parameter_centers": {}}
    subset = trials.loc[
        trials["study_name"].eq(str(candidate["study_name"]))
        & trials["state"].eq("COMPLETE")
        & np.isfinite(trials["objective_log_loss"])
    ].sort_values("objective_log_loss")
    if subset.empty:
        return {"completed_trials": 0, "top_trials": 0, "parameter_centers": {}}
    top_count = min(len(subset), max(2, int(np.ceil(len(subset) * 0.5))))
    strongest = subset.head(top_count)
    centers: dict[str, Any] = {}
    param_columns = [column for column in strongest if column.startswith("param__")]
    for column in param_columns:
        name = column.removeprefix("param__")
        numeric = pd.to_numeric(strongest[column], errors="coerce")
        finite = numeric[np.isfinite(numeric)]
        if len(finite):
            centers[name] = float(finite.median())
        else:
            values = strongest[column].dropna().astype(str)
            if len(values):
                centers[name] = str(values.mode().iloc[0])
    return {
        "completed_trials": len(subset),
        "top_trials": top_count,
        "best_log_loss": float(subset.iloc[0]["objective_log_loss"]),
        "top_log_loss_median": float(strongest["objective_log_loss"].median()),
        "parameter_centers": centers,
    }


def select_node07_references(
    project_root: Path,
    source_run_id: str,
) -> list[SelectedReference]:
    """Choose top-2 CV5 candidates plus the best 3D, always yielding 3 unique models."""

    source_run_dir = (
        project_root
        / "outputs"
        / "private_eda"
        / "node07_runs"
        / source_run_id
    )
    metrics, candidates = _read_source_candidates(source_run_dir)
    if len(metrics) < 3:
        raise RuntimeError("El nodo 09 necesita al menos tres finalistas CV5 del nodo 07.")
    top_two = metrics.head(2)["candidate_id"].astype(str).tolist()
    three_d = metrics.loc[metrics["architecture"].astype(str).eq("3d")]
    if three_d.empty:
        raise RuntimeError("El nodo 07 no contiene un finalista 3D evaluado por CV5.")
    best_3d = str(three_d.iloc[0]["candidate_id"])
    selected_ids = list(dict.fromkeys([*top_two, best_3d]))
    for candidate_id in metrics["candidate_id"].astype(str):
        if len(selected_ids) >= 3:
            break
        if candidate_id not in selected_ids:
            selected_ids.append(candidate_id)
    candidates_by_id = {str(item["candidate_id"]): item for item in candidates}
    rank_by_id = {
        str(row.candidate_id): index + 1
        for index, row in enumerate(metrics.itertuples(index=False))
    }
    loss_by_id = metrics.set_index("candidate_id")["calibrated_log_loss"].to_dict()
    selected: list[SelectedReference] = []
    for candidate_id in selected_ids[:3]:
        roles: list[str] = []
        if candidate_id in top_two:
            roles.append(f"top_{top_two.index(candidate_id) + 1}_overall")
        if candidate_id == best_3d:
            roles.append("best_3d")
        if not roles:
            roles.append("diversity_fill")
        candidate = candidates_by_id[candidate_id]
        selected.append(
            SelectedReference(
                candidate=candidate,
                selection_roles=tuple(roles),
                node07_rank=int(rank_by_id[candidate_id]),
                node07_calibrated_log_loss=float(loss_by_id[candidate_id]),
                trajectory_context=_trajectory_context(
                    project_root, source_run_id, candidate
                ),
            )
        )
    return selected
