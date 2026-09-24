from __future__ import annotations

import hashlib
import json
import sqlite3
import tempfile
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import pandas as pd

from modeling.cnn.calibration import binary_metrics

FEATURE_VARIANTS = ("image_radiomics_sbr", "image_radiomics", "image_only")
LATERAL_STRATEGIES = ("random_flip", "canonical")


@dataclass(frozen=True)
class RunSpec:
    node: str
    run_id: str
    run_dir: Path
    database_path: Path | None

    @property
    def run_key(self) -> str:
        return f"{self.node}:{self.run_id}"


@dataclass(frozen=True)
class TrajectorySnapshot:
    created_at: str
    catalog: pd.DataFrame
    trials: pd.DataFrame
    histories: pd.DataFrame
    fold_scores: pd.DataFrame
    diagnostics: pd.DataFrame
    trajectory_summary: pd.DataFrame
    final_progress: pd.DataFrame
    final_predictions: pd.DataFrame
    final_metrics: pd.DataFrame
    warnings: tuple[str, ...]


def _stable_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _mtime(path: Path | None) -> pd.Timestamp | pd.NaT:
    if path is None or not path.exists():
        return pd.NaT
    return pd.Timestamp(path.stat().st_mtime, unit="s", tz="UTC")


def _database_for(node: str, run_dir: Path) -> Path | None:
    filename = {
        "node06": "optuna_cnn.sqlite3",
        "node07": "optuna_node07.sqlite3",
        "node09": "optuna_node09.sqlite3",
        "node10": "optuna_node10.sqlite3",
    }[node]
    candidate = run_dir / "search" / filename
    return candidate if candidate.exists() else None


def _run_specs(project_root: Path) -> list[RunSpec]:
    private_root = project_root / "outputs" / "private_eda"
    specs: list[RunSpec] = []
    for node, folder in (
        ("node06", "cnn_runs"),
        ("node07", "node07_runs"),
        ("node09", "node09_runs"),
        ("node10", "node10_runs"),
    ):
        parent = private_root / folder
        if not parent.exists():
            continue
        for run_dir in sorted(path for path in parent.iterdir() if path.is_dir()):
            specs.append(
                RunSpec(
                    node=node,
                    run_id=run_dir.name,
                    run_dir=run_dir,
                    database_path=_database_for(node, run_dir),
                )
            )
    return specs


def _history_paths(spec: RunSpec) -> list[Path]:
    names = {"training_history.csv", "history.csv"}
    paths: list[Path] = []
    for phase_dir in (
        spec.run_dir / "search" / "trials",
        spec.run_dir / "final" / "cv5",
        spec.run_dir / "final" / "models",
    ):
        if not phase_dir.exists():
            continue
        paths.extend(
            path for path in phase_dir.rglob("*.csv") if path.name in names
        )
    return paths


def discover_runs(project_root: str | Path) -> pd.DataFrame:
    root = Path(project_root).resolve()
    records: list[dict[str, Any]] = []
    for spec in _run_specs(root):
        histories = _history_paths(spec)
        activity = [_mtime(spec.database_path)]
        activity.extend(_mtime(path) for path in histories)
        finite_activity = [value for value in activity if not pd.isna(value)]
        records.append(
            {
                "run_key": spec.run_key,
                "node": spec.node,
                "run_id": spec.run_id,
                "run_dir": str(spec.run_dir),
                "has_optuna_database": spec.database_path is not None,
                "history_files": len(histories),
                "latest_activity_utc": max(finite_activity) if finite_activity else pd.NaT,
            }
        )
    columns = [
        "run_key",
        "node",
        "run_id",
        "run_dir",
        "has_optuna_database",
        "history_files",
        "latest_activity_utc",
    ]
    if not records:
        return pd.DataFrame(columns=columns)
    return (
        pd.DataFrame.from_records(records, columns=columns)
        .sort_values(["node", "latest_activity_utc", "run_id"], ascending=[True, False, False])
        .reset_index(drop=True)
    )


def _default_run_keys(catalog: pd.DataFrame) -> list[str]:
    if catalog.empty:
        return []
    selected: list[str] = []
    for node in ("node06", "node07", "node09", "node10"):
        subset = catalog.loc[catalog["node"].eq(node)]
        if not subset.empty:
            selected.append(str(subset.iloc[0]["run_key"]))
    return selected


def _copy_sqlite_snapshot(source: Path, destination: Path) -> None:
    source_uri = f"{source.resolve().as_uri()}?mode=ro"
    source_connection = sqlite3.connect(source_uri, uri=True, timeout=8.0)
    destination_connection = sqlite3.connect(destination)
    try:
        source_connection.execute("PRAGMA busy_timeout=8000")
        source_connection.backup(destination_connection, pages=256, sleep=0.05)
    finally:
        destination_connection.close()
        source_connection.close()


def _parse_study(study_name: str, node: str) -> tuple[str, str, str]:
    variant = next(
        (candidate for candidate in FEATURE_VARIANTS if study_name.endswith(candidate)),
        "unknown",
    )
    if node == "node06":
        prefix = study_name.removeprefix("cnn_")
        architecture = "2.5d" if prefix.startswith("25d_") else "3d" if prefix.startswith("3d_") else "unknown"
        return architecture, variant, "not_applicable"
    if node == "node09":
        prefix = study_name.removeprefix("node09_")
        architecture = next(
            (value for value in ("slab2d", "25d", "3d") if prefix.startswith(f"{value}_")),
            "unknown",
        )
        architecture = "2.5d" if architecture == "25d" else architecture
        return architecture, variant, "unknown"
    if node == "node10":
        family = study_name.removeprefix("node10_")
        if family == "dual_stream_multitask":
            architecture = "3d_dual"
        elif family.startswith("graph_"):
            architecture = "graph"
        elif family == "topology_hgb":
            architecture = "topology"
        elif family == "diffusion_map":
            architecture = "diffusion"
        elif family in {"subtype_mixture", "hybrid_stacking"}:
            architecture = "mixture"
        else:
            architecture = "tabular"
        return architecture, family, "not_applicable"
    prefix = study_name.removeprefix("node07_")
    architecture = next(
        (value for value in ("slab2d", "25d", "3d") if prefix.startswith(f"{value}_")),
        "unknown",
    )
    architecture = "2.5d" if architecture == "25d" else architecture
    lateral = next(
        (candidate for candidate in LATERAL_STRATEGIES if f"_{candidate}_" in study_name),
        "unknown",
    )
    return architecture, variant, lateral


def _candidate_parameter_hash(candidate: dict[str, Any]) -> str | None:
    if not candidate:
        return None
    payload = dict(candidate)
    payload.pop("fold_log_loss", None)
    payload.pop("fixed_epochs", None)
    return _stable_hash(payload)[:20]


def _json_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return {}
    try:
        loaded = json.loads(str(value))
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _trial_record(
    *,
    spec: RunSpec,
    study_name: str,
    number: int,
    state: str,
    value: float | None,
    duration_seconds: float | None,
    params: dict[str, Any],
    candidate: dict[str, Any],
    datetime_start: Any = None,
    datetime_complete: Any = None,
) -> dict[str, Any]:
    inferred_architecture, inferred_variant, inferred_lateral = _parse_study(
        study_name, spec.node
    )
    architecture = candidate.get("architecture", inferred_architecture)
    feature_variant = candidate.get("feature_variant", inferred_variant)
    lateral_strategy = candidate.get("lateral_strategy", inferred_lateral)
    record: dict[str, Any] = {
        "run_key": spec.run_key,
        "node": spec.node,
        "run_id": spec.run_id,
        "study_name": study_name,
        "study_label": (
            study_name.replace("cnn_", "06 · ")
            .replace("node07_", "07 · ")
            .replace("node09_", "09 · ")
            .replace("node10_", "10 · ")
        ),
        "trial_number": int(number),
        "state": str(state).upper(),
        "objective_log_loss": float(value) if value is not None and np.isfinite(value) else np.nan,
        "duration_seconds": (
            float(duration_seconds)
            if duration_seconds is not None and np.isfinite(duration_seconds)
            else np.nan
        ),
        "datetime_start": datetime_start,
        "datetime_complete": datetime_complete,
        "architecture": architecture,
        "feature_variant": feature_variant,
        "lateral_strategy": lateral_strategy,
        "parameter_hash": _candidate_parameter_hash(candidate),
        "fixed_epochs": candidate.get("fixed_epochs", np.nan),
        "fold_scores_json": json.dumps(candidate.get("fold_log_loss", [])),
        "params_json": json.dumps(params, sort_keys=True, default=str),
        "candidate_json": json.dumps(candidate, sort_keys=True, default=str),
    }
    for key, item in params.items():
        record[f"param__{key}"] = item
    return record


def _load_trials_from_database(spec: RunSpec) -> pd.DataFrame:
    if spec.database_path is None:
        return pd.DataFrame()
    with tempfile.TemporaryDirectory(prefix="dat-optuna-read-") as temporary_dir:
        snapshot_path = Path(temporary_dir) / spec.database_path.name
        _copy_sqlite_snapshot(spec.database_path, snapshot_path)
        storage_url = f"sqlite:///{snapshot_path.as_posix()}"
        storage = optuna.storages.RDBStorage(url=storage_url)
        records: list[dict[str, Any]] = []
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                summaries = optuna.study.get_all_study_summaries(storage=storage)
                for summary in summaries:
                    study = optuna.load_study(study_name=summary.study_name, storage=storage)
                    for trial in study.trials:
                        records.append(
                            _trial_record(
                                spec=spec,
                                study_name=summary.study_name,
                                number=trial.number,
                                state=trial.state.name,
                                value=trial.value,
                                duration_seconds=(
                                    trial.duration.total_seconds()
                                    if trial.duration is not None
                                    else None
                                ),
                                params=dict(trial.params),
                                candidate=dict(trial.user_attrs.get("candidate", {})),
                                datetime_start=trial.datetime_start,
                                datetime_complete=trial.datetime_complete,
                            )
                        )
        finally:
            storage.remove_session()
            storage.engine.dispose()
    return pd.DataFrame.from_records(records)


def _load_trials_from_csv(spec: RunSpec) -> pd.DataFrame:
    studies_dir = spec.run_dir / "search" / "studies"
    if not studies_dir.exists():
        return pd.DataFrame()
    records: list[dict[str, Any]] = []
    for path in sorted(studies_dir.glob("*_trials.csv")):
        study_name = path.name.removesuffix("_trials.csv")
        frame = pd.read_csv(path)
        for row in frame.to_dict("records"):
            params = _json_mapping(row.get("params_json", row.get("parameters")))
            candidate = _json_mapping(row.get("candidate_json", row.get("candidate")))
            number = row.get("number", row.get("trial", -1))
            records.append(
                _trial_record(
                    spec=spec,
                    study_name=study_name,
                    number=int(number),
                    state=str(row.get("state", "UNKNOWN")),
                    value=pd.to_numeric(row.get("value"), errors="coerce"),
                    duration_seconds=pd.to_numeric(
                        row.get("duration_seconds"), errors="coerce"
                    ),
                    params=params,
                    candidate=candidate,
                )
            )
    return pd.DataFrame.from_records(records)


def _load_trials(spec: RunSpec, messages: list[str]) -> pd.DataFrame:
    if spec.database_path is not None:
        try:
            return _load_trials_from_database(spec)
        except (OSError, sqlite3.Error, RuntimeError, ValueError) as error:
            messages.append(
                f"{spec.run_key}: no se pudo leer el snapshot SQLite ({type(error).__name__}); "
                "se usaron los CSV exportados."
            )
    return _load_trials_from_csv(spec)


def _load_finalist_metadata(spec: RunSpec) -> dict[str, dict[str, Any]]:
    path = spec.run_dir / "search" / "finalists.json"
    if not path.exists():
        return {}
    try:
        finalists = json.loads(path.read_text(encoding="utf-8")).get("finalists", [])
    except (OSError, json.JSONDecodeError):
        return {}
    return {
        str(candidate.get("candidate_id")): candidate
        for candidate in finalists
        if candidate.get("candidate_id")
    }


def _expected_final_folds(spec: RunSpec) -> int:
    candidates = (
        spec.run_dir / "config" / "experiment_config.json",
        spec.run_dir / "cnn_experiment_config.json",
    )
    for path in candidates:
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            return int(payload.get("search", {}).get("n_splits_final", 5))
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            continue
    return 5


def _candidate_label(candidate: dict[str, Any], candidate_id: str) -> str:
    family = str(candidate.get("family", ""))
    inferred_architecture, inferred_variant, _ = _parse_study(
        str(candidate.get("study_name", f"node10_{family}")),
        "node10" if family else "node07",
    )
    architecture = str(candidate.get("architecture", inferred_architecture))
    variant = str(candidate.get("feature_variant", family or inferred_variant))
    lateral = str(candidate.get("lateral_strategy", "not_applicable"))
    parts = [architecture, variant]
    if lateral in LATERAL_STRATEGIES:
        parts.append(lateral)
    parts.append(candidate_id[:8])
    return " · ".join(parts)


def _read_csv_snapshot(
    path: Path,
    messages: list[str],
    *,
    label: str,
) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return _read_history(path)
    except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError) as error:
        messages.append(f"{path}: {label} omitido ({type(error).__name__}).")
        return pd.DataFrame()


def _prediction_metrics(frame: pd.DataFrame, probability_column: str) -> dict[str, float]:
    usable = frame.loc[
        frame["is_pathologic"].notna() & frame[probability_column].notna()
    ].copy()
    if usable.empty:
        return {
            "log_loss": np.nan,
            "brier": np.nan,
            "auc": np.nan,
            "balanced_accuracy_0_5": np.nan,
            "ece_10": np.nan,
        }
    return binary_metrics(usable, probability_column)


def _cohort_metadata(spec: RunSpec, messages: list[str]) -> pd.DataFrame:
    path = spec.run_dir / "config" / "prepared_cohort.csv"
    frame = _read_csv_snapshot(path, messages, label="cohorte preparada")
    if frame.empty or "uid" not in frame:
        return pd.DataFrame()
    wanted = [
        column
        for column in (
            "uid",
            "acquisition_family",
            "background_qc_valid",
            "mask_mode",
            "lateral_flip_applied",
            "affected_side_proxy",
        )
        if column in frame
    ]
    return frame[wanted].drop_duplicates("uid")


def _load_final_artifacts(
    spec: RunSpec,
    messages: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    finalists = _load_finalist_metadata(spec)
    if not finalists:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    expected_folds = _expected_final_folds(spec)
    final_dir = spec.run_dir / "final"
    progress_records: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    cohort_metadata = _cohort_metadata(spec, messages)

    for candidate_id, candidate in finalists.items():
        candidate_label = _candidate_label(candidate, candidate_id)
        fold_prediction_frames: list[pd.DataFrame] = []
        for fold in range(expected_folds):
            fold_dir = (
                final_dir / "models" / str(candidate["family"]) / f"fold_{fold}"
                if spec.node == "node10"
                else final_dir / "cv5" / candidate_id / f"fold_{fold}"
            )
            history = _read_csv_snapshot(
                fold_dir / "history.csv", messages, label="historial final"
            )
            predictions = _read_csv_snapshot(
                fold_dir / "validation_predictions.csv",
                messages,
                label="predicciones de fold",
            )
            epochs_recorded = (
                int(pd.to_numeric(history["epoch"], errors="coerce").nunique())
                if not history.empty and "epoch" in history
                else 0
            )
            planned_epochs = int(candidate.get("fixed_epochs", 0) or 0)
            if not predictions.empty:
                status = "complete"
            elif epochs_recorded:
                status = "training"
            else:
                status = "pending"
            fold_metrics = (
                _prediction_metrics(predictions, "probability")
                if not predictions.empty and "probability" in predictions
                else {}
            )
            progress_records.append(
                {
                    "run_key": spec.run_key,
                    "node": spec.node,
                    "run_id": spec.run_id,
                    "candidate_id": candidate_id,
                    "candidate_label": candidate_label,
                    "study_name": candidate.get("study_name", "unknown"),
                    "architecture": _parse_study(
                        str(candidate.get("study_name", "unknown")), spec.node
                    )[0],
                    "feature_variant": candidate.get(
                        "feature_variant", candidate.get("family", "unknown")
                    ),
                    "lateral_strategy": candidate.get(
                        "lateral_strategy", "not_applicable"
                    ),
                    "search_log_loss": pd.to_numeric(
                        candidate.get("search_log_loss"), errors="coerce"
                    ),
                    "fold": fold,
                    "status": status,
                    "epochs_recorded": epochs_recorded,
                    "planned_epochs": planned_epochs,
                    "epoch_fraction": (
                        min(1.0, epochs_recorded / planned_epochs)
                        if planned_epochs > 0
                        else 0.0
                    ),
                    "n_predictions": len(predictions),
                    "fold_log_loss": fold_metrics.get("log_loss", np.nan),
                    "fold_auc": fold_metrics.get("auc", np.nan),
                    "fold_brier": fold_metrics.get("brier", np.nan),
                    "fold_ece_10": fold_metrics.get("ece_10", np.nan),
                }
            )
            if not predictions.empty:
                predictions = predictions.copy()
                predictions["fold"] = fold
                fold_prediction_frames.append(predictions)

        oof_paths = (
            final_dir / f"oof_{candidate.get('family')}.csv",
            final_dir / f"oof_{candidate_id}.csv",
            final_dir / "oof" / candidate_id / "oof_predictions.csv",
        )
        candidate_predictions = pd.DataFrame()
        for oof_path in oof_paths:
            candidate_predictions = _read_csv_snapshot(
                oof_path, messages, label="OOF del candidato"
            )
            if not candidate_predictions.empty:
                break
        prediction_source = "candidate_oof"
        if candidate_predictions.empty and fold_prediction_frames:
            candidate_predictions = pd.concat(
                fold_prediction_frames, ignore_index=True, sort=False
            )
            prediction_source = "completed_folds"
        if candidate_predictions.empty:
            continue
        candidate_predictions = candidate_predictions.copy()
        candidate_predictions["run_key"] = spec.run_key
        candidate_predictions["node"] = spec.node
        candidate_predictions["run_id"] = spec.run_id
        candidate_predictions["candidate_id"] = candidate_id
        candidate_predictions["candidate_label"] = candidate_label
        candidate_predictions["architecture"] = candidate.get(
            "architecture",
            _parse_study(str(candidate.get("study_name", "unknown")), spec.node)[0],
        )
        candidate_predictions["feature_variant"] = candidate.get(
            "feature_variant", candidate.get("family", "unknown")
        )
        candidate_predictions["lateral_strategy"] = candidate.get(
            "lateral_strategy", "not_applicable"
        )
        candidate_predictions["prediction_source"] = prediction_source
        if not cohort_metadata.empty:
            missing_metadata = [
                column
                for column in cohort_metadata.columns
                if column != "uid" and column not in candidate_predictions
            ]
            if missing_metadata:
                candidate_predictions = candidate_predictions.merge(
                    cohort_metadata[["uid", *missing_metadata]],
                    on="uid",
                    how="left",
                    validate="many_to_one",
                )
        prediction_frames.append(candidate_predictions)

    progress = pd.DataFrame.from_records(progress_records)
    predictions = (
        pd.concat(prediction_frames, ignore_index=True, sort=False)
        if prediction_frames
        else pd.DataFrame()
    )
    if spec.node == "node10":
        combined = _read_csv_snapshot(
            final_dir / "all_oof_predictions.csv",
            messages,
            label="OOF combinado del nodo 10",
        )
        if not combined.empty and "candidate_id" in combined:
            existing_ids = (
                set(predictions["candidate_id"].astype(str))
                if not predictions.empty and "candidate_id" in predictions
                else set()
            )
            ensemble = combined.loc[~combined["candidate_id"].astype(str).isin(existing_ids)].copy()
            if not ensemble.empty:
                ensemble["run_key"] = spec.run_key
                ensemble["node"] = spec.node
                ensemble["run_id"] = spec.run_id
                ensemble["candidate_label"] = ensemble["family"].astype(str)
                ensemble["architecture"] = "ensemble"
                ensemble["feature_variant"] = ensemble["family"].astype(str)
                ensemble["lateral_strategy"] = "not_applicable"
                ensemble["prediction_source"] = "combined_oof"
                if not cohort_metadata.empty:
                    ensemble = ensemble.merge(
                        cohort_metadata,
                        on="uid",
                        how="left",
                        validate="many_to_one",
                    )
                predictions = pd.concat([predictions, ensemble], ignore_index=True, sort=False)
    metric_records: list[dict[str, Any]] = []
    if not predictions.empty:
        for candidate_id, frame in predictions.groupby("candidate_id", sort=False):
            candidate = finalists.get(
                str(candidate_id),
                {
                    "family": str(frame["family"].iloc[0]) if "family" in frame else str(candidate_id),
                    "architecture": "ensemble",
                    "feature_variant": (
                        str(frame["family"].iloc[0]) if "family" in frame else str(candidate_id)
                    ),
                },
            )
            raw = _prediction_metrics(frame, "probability")
            calibrated = (
                _prediction_metrics(frame, "probability_cross_calibrated")
                if "probability_cross_calibrated" in frame
                and frame["probability_cross_calibrated"].notna().all()
                else {}
            )
            folds_completed = int(frame["fold"].nunique())
            metric_records.append(
                {
                    "run_key": spec.run_key,
                    "node": spec.node,
                    "run_id": spec.run_id,
                    "candidate_id": str(candidate_id),
                    "candidate_label": _candidate_label(candidate, str(candidate_id)),
                    "architecture": candidate.get(
                        "architecture",
                        _parse_study(
                            str(candidate.get("study_name", "unknown")), spec.node
                        )[0],
                    ),
                    "feature_variant": candidate.get(
                        "feature_variant", candidate.get("family", "unknown")
                    ),
                    "lateral_strategy": candidate.get(
                        "lateral_strategy", "not_applicable"
                    ),
                    "search_log_loss": pd.to_numeric(
                        candidate.get("search_log_loss"), errors="coerce"
                    ),
                    "folds_completed": folds_completed,
                    "expected_folds": expected_folds,
                    "n_predictions": len(frame),
                    "evaluation_complete": folds_completed == expected_folds,
                    "metrics_source": (
                        "cross_calibrated_oof"
                        if calibrated
                        else (
                            "complete_raw_oof"
                            if folds_completed == expected_folds
                            else "partial_raw_oof"
                        )
                    ),
                    **{f"raw_{key}": value for key, value in raw.items()},
                    **{
                        f"calibrated_{key}": value
                        for key, value in calibrated.items()
                    },
                }
            )
    metrics = pd.DataFrame.from_records(metric_records)
    official_metrics = pd.DataFrame()
    for metrics_path in (
        final_dir / "final_metrics.csv",
        final_dir / "finalist_cv5_metrics.csv",
    ):
        official_metrics = _read_csv_snapshot(
            metrics_path, messages, label="métricas finales"
        )
        if not official_metrics.empty:
            break
    if not official_metrics.empty:
        official_metrics = official_metrics.rename(
            columns={
                column: column.replace("cross_calibrated_", "calibrated_", 1)
                for column in official_metrics.columns
                if column.startswith("cross_calibrated_")
            }
        )
    if not metrics.empty and not official_metrics.empty and "candidate_id" in official_metrics:
        official = official_metrics.set_index("candidate_id")
        for row_index, candidate_id in metrics["candidate_id"].items():
            if candidate_id not in official.index:
                continue
            row = official.loc[candidate_id]
            for column, value in row.items():
                if column == "candidate_id":
                    continue
                metrics.loc[row_index, column] = value
            metrics.loc[row_index, "metrics_source"] = "official_final_metrics"
    return progress, predictions, metrics


def _history_metadata(spec: RunSpec, path: Path) -> dict[str, Any]:
    search_root = spec.run_dir / "search" / "trials"
    final_root = spec.run_dir / "final" / "cv5"
    if search_root in path.parents:
        relative = path.relative_to(search_root)
        study_name, trajectory_id, fold_name = relative.parts[:3]
        architecture, variant, lateral = _parse_study(study_name, spec.node)
        return {
            "phase": "search_cv3",
            "study_name": study_name,
            "trajectory_id": trajectory_id,
            "fold": int(fold_name.removeprefix("fold_")),
            "architecture": architecture,
            "feature_variant": variant,
            "lateral_strategy": lateral,
        }
    if spec.node == "node10" and (spec.run_dir / "final" / "models") in path.parents:
        relative = path.relative_to(spec.run_dir / "final" / "models")
        family, fold_name = relative.parts[:2]
        candidates = _load_finalist_metadata(spec)
        candidate_id, candidate = next(
            (
                (candidate_id, value)
                for candidate_id, value in candidates.items()
                if value.get("family") == family
            ),
            (family, {}),
        )
        study_name = str(candidate.get("study_name", f"node10_{family}"))
        architecture, variant, lateral = _parse_study(study_name, spec.node)
        return {
            "phase": "final_cv5",
            "study_name": study_name,
            "trajectory_id": candidate_id,
            "fold": int(fold_name.removeprefix("fold_")),
            "architecture": architecture,
            "feature_variant": family or variant,
            "lateral_strategy": lateral,
            "trial_number": candidate.get("trial_number", np.nan),
            "objective_log_loss": candidate.get("search_log_loss", np.nan),
            "fixed_epochs": candidate.get("fixed_epochs", np.nan),
        }
    relative = path.relative_to(final_root)
    candidate_id, fold_name = relative.parts[:2]
    candidate = _load_finalist_metadata(spec).get(candidate_id, {})
    study_name = str(candidate.get("study_name", f"final_{candidate_id}"))
    architecture, variant, lateral = _parse_study(study_name, spec.node)
    return {
        "phase": "final_cv5",
        "study_name": study_name,
        "trajectory_id": candidate_id,
        "fold": int(fold_name.removeprefix("fold_")),
        "architecture": candidate.get("architecture", architecture),
        "feature_variant": candidate.get("feature_variant", variant),
        "lateral_strategy": candidate.get("lateral_strategy", lateral),
        "trial_number": candidate.get("trial_number", np.nan),
        "objective_log_loss": candidate.get("search_log_loss", np.nan),
        "fixed_epochs": candidate.get("fixed_epochs", np.nan),
    }


def _read_history(path: Path, attempts: int = 4) -> pd.DataFrame:
    error: BaseException | None = None
    for _ in range(attempts):
        try:
            return pd.read_csv(path)
        except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError) as caught:
            error = caught
    if error is not None:
        raise error
    return pd.DataFrame()


def _load_histories(spec: RunSpec, trial_lookup: pd.DataFrame, messages: list[str]) -> pd.DataFrame:
    records: list[pd.DataFrame] = []
    for path in sorted(_history_paths(spec)):
        try:
            frame = _read_history(path)
        except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError) as error:
            messages.append(f"{path}: historial omitido ({type(error).__name__}).")
            continue
        if frame.empty or "epoch" not in frame:
            continue
        metadata = _history_metadata(spec, path)
        for key, value in metadata.items():
            frame[key] = value
        frame["run_key"] = spec.run_key
        frame["node"] = spec.node
        frame["run_id"] = spec.run_id
        frame["history_path"] = str(path)
        frame["epoch"] = pd.to_numeric(frame["epoch"], errors="coerce")
        frame["epoch_number"] = frame["epoch"] + 1
        frame["train_loss"] = pd.to_numeric(frame.get("train_loss"), errors="coerce")
        frame["validation_log_loss"] = pd.to_numeric(
            frame.get("validation_log_loss"), errors="coerce"
        )
        frame["learning_rate"] = pd.to_numeric(
            frame.get("learning_rate"), errors="coerce"
        )
        frame["best_validation_so_far"] = frame["validation_log_loss"].cummin()
        records.append(frame)
    if not records:
        return pd.DataFrame()
    histories = pd.concat(records, ignore_index=True, sort=False)
    if trial_lookup.empty:
        return histories
    lookup_columns = [
        "run_key",
        "study_name",
        "parameter_hash",
        "trial_number",
        "state",
        "objective_log_loss",
        "fixed_epochs",
    ]
    available = [column for column in lookup_columns if column in trial_lookup]
    lookup = trial_lookup.loc[
        trial_lookup["parameter_hash"].notna(), available
    ].copy()
    if lookup.empty:
        return histories
    lookup["state_priority"] = lookup["state"].map(
        {"COMPLETE": 0, "RUNNING": 1, "WAITING": 2, "PRUNED": 3, "FAIL": 4}
    ).fillna(9)
    lookup = (
        lookup.sort_values(["state_priority", "trial_number"])
        .drop_duplicates(["run_key", "study_name", "parameter_hash"])
        .drop(columns="state_priority")
        .rename(columns={"parameter_hash": "trajectory_id"})
    )
    histories = histories.merge(
        lookup,
        how="left",
        on=["run_key", "study_name", "trajectory_id"],
        suffixes=("", "_trial"),
    )
    for column in ("trial_number", "objective_log_loss", "fixed_epochs"):
        trial_column = f"{column}_trial"
        if trial_column in histories:
            histories[column] = histories.get(column, pd.Series(index=histories.index)).combine_first(
                histories[trial_column]
            )
            histories = histories.drop(columns=trial_column)
    histories["state"] = histories.get("state", pd.Series(index=histories.index)).fillna(
        "UNMAPPED_OR_ACTIVE"
    )
    return histories


def _fold_scores(trials: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    if trials.empty:
        return pd.DataFrame()
    for row in trials.to_dict("records"):
        try:
            values = json.loads(row.get("fold_scores_json", "[]"))
        except (TypeError, json.JSONDecodeError):
            values = []
        if not isinstance(values, list):
            continue
        for fold, value in enumerate(values):
            numeric = pd.to_numeric(value, errors="coerce")
            if pd.isna(numeric):
                continue
            records.append(
                {
                    "run_key": row["run_key"],
                    "node": row["node"],
                    "run_id": row["run_id"],
                    "study_name": row["study_name"],
                    "study_label": row["study_label"],
                    "architecture": row["architecture"],
                    "feature_variant": row["feature_variant"],
                    "lateral_strategy": row["lateral_strategy"],
                    "trial_number": row["trial_number"],
                    "fold": fold,
                    "fold_log_loss": float(numeric),
                }
            )
    return pd.DataFrame.from_records(records)


def _linear_slope(frame: pd.DataFrame, y_column: str, tail: int = 4) -> float:
    finite = frame.loc[np.isfinite(frame[y_column]), ["epoch_number", y_column]].tail(tail)
    if len(finite) < 2 or finite["epoch_number"].nunique() < 2:
        return np.nan
    return float(np.polyfit(finite["epoch_number"], finite[y_column], 1)[0])


def diagnose_learning_curves(histories: pd.DataFrame) -> pd.DataFrame:
    """Resume señales heurísticas; no reemplaza una evaluación OOF externa."""

    columns = [
        "run_key",
        "node",
        "run_id",
        "phase",
        "study_name",
        "architecture",
        "feature_variant",
        "lateral_strategy",
        "trajectory_id",
        "trial_number",
        "fold",
    ]
    if histories.empty:
        return pd.DataFrame(columns=columns + ["diagnosis"])
    records: list[dict[str, Any]] = []
    for keys, group in histories.groupby(columns, dropna=False, sort=False):
        ordered = group.sort_values("epoch_number")
        train = ordered.loc[np.isfinite(ordered["train_loss"]), "train_loss"]
        valid = ordered.loc[
            np.isfinite(ordered["validation_log_loss"]),
            ["epoch_number", "validation_log_loss"],
        ]
        record = dict(zip(columns, keys, strict=True))
        record.update(
            {
                "epochs_recorded": int(ordered["epoch_number"].nunique()),
                "validation_points": len(valid),
                "train_first": float(train.iloc[0]) if len(train) else np.nan,
                "train_last": float(train.iloc[-1]) if len(train) else np.nan,
                "train_drop": float(train.iloc[0] - train.iloc[-1]) if len(train) else np.nan,
                "validation_first": float(valid["validation_log_loss"].iloc[0]) if len(valid) else np.nan,
                "validation_best": float(valid["validation_log_loss"].min()) if len(valid) else np.nan,
                "validation_last": float(valid["validation_log_loss"].iloc[-1]) if len(valid) else np.nan,
                "best_epoch": (
                    int(valid.loc[valid["validation_log_loss"].idxmin(), "epoch_number"])
                    if len(valid)
                    else np.nan
                ),
                "train_tail_slope": _linear_slope(ordered, "train_loss"),
                "validation_tail_slope": _linear_slope(ordered, "validation_log_loss"),
            }
        )
        if len(valid):
            record["validation_improvement"] = record["validation_first"] - record["validation_best"]
            record["validation_rebound"] = record["validation_last"] - record["validation_best"]
            record["epochs_after_best"] = int(
                ordered["epoch_number"].max() - record["best_epoch"]
            )
            differences = np.diff(valid["validation_log_loss"].to_numpy(dtype=float))
            record["validation_step_std"] = float(np.std(differences)) if len(differences) else np.nan
        else:
            record.update(
                {
                    "validation_improvement": np.nan,
                    "validation_rebound": np.nan,
                    "epochs_after_best": np.nan,
                    "validation_step_std": np.nan,
                }
            )
        if len(valid) < 4:
            diagnosis = "validación insuficiente"
        elif (
            record["validation_rebound"] >= 0.03
            and record["epochs_after_best"] >= 2
            and record["train_tail_slope"] < -0.001
        ):
            diagnosis = "posible sobreajuste"
        elif record["validation_step_std"] >= 0.07:
            diagnosis = "validación inestable"
        elif (
            record["best_epoch"] >= ordered["epoch_number"].max() - 1
            and record["validation_tail_slope"] < -0.002
        ):
            diagnosis = "aún estaba mejorando"
        elif (
            record["train_drop"] < 0.03
            and record["validation_best"] >= 0.67
        ):
            diagnosis = "posible subajuste u optimización débil"
        elif (
            abs(record["validation_tail_slope"]) <= 0.003
            and record["validation_rebound"] < 0.02
        ):
            diagnosis = "meseta estable"
        else:
            diagnosis = "señal mixta"
        record["diagnosis"] = diagnosis
        records.append(record)
    return pd.DataFrame.from_records(records)


def _trajectory_summary(diagnostics: pd.DataFrame) -> pd.DataFrame:
    if diagnostics.empty:
        return pd.DataFrame()
    keys = [
        "run_key",
        "node",
        "run_id",
        "phase",
        "study_name",
        "architecture",
        "feature_variant",
        "lateral_strategy",
        "trajectory_id",
        "trial_number",
    ]
    summary = (
        diagnostics.groupby(keys, dropna=False)
        .agg(
            folds_observed=("fold", "nunique"),
            mean_best_validation=("validation_best", "mean"),
            std_best_validation=("validation_best", "std"),
            mean_last_validation=("validation_last", "mean"),
            mean_rebound=("validation_rebound", "mean"),
            median_best_epoch=("best_epoch", "median"),
            overfit_fraction=("diagnosis", lambda values: float(np.mean(values.eq("posible sobreajuste")))),
            underfit_fraction=(
                "diagnosis",
                lambda values: float(np.mean(values.eq("posible subajuste u optimización débil"))),
            ),
            unstable_fraction=("diagnosis", lambda values: float(np.mean(values.eq("validación inestable")))),
        )
        .reset_index()
    )
    return summary.sort_values(
        ["phase", "mean_best_validation", "std_best_validation"], na_position="last"
    ).reset_index(drop=True)


def _decorate_trials(trials: pd.DataFrame) -> pd.DataFrame:
    if trials.empty:
        return trials
    trials = trials.sort_values(["run_key", "study_name", "trial_number"]).reset_index(drop=True)
    trials["completed"] = trials["state"].eq("COMPLETE") & trials["objective_log_loss"].notna()
    trials["best_so_far"] = np.nan
    for index in trials.groupby(["run_key", "study_name"], sort=False).groups.values():
        subset = trials.loc[index]
        completed_values = subset["objective_log_loss"].where(subset["completed"])
        trials.loc[index, "best_so_far"] = completed_values.cummin().ffill().to_numpy()
    trials["network"] = (
        trials["node"].str.replace("node", "", regex=False)
        + " · "
        + trials["architecture"].astype(str)
        + " · "
        + trials["feature_variant"].astype(str)
        + np.where(
            trials["lateral_strategy"].isin(["canonical", "random_flip"]),
            " · " + trials["lateral_strategy"].astype(str),
            "",
        )
    )
    return trials


def load_trajectory_snapshot(
    project_root: str | Path,
    run_keys: Sequence[str] | None = None,
) -> TrajectorySnapshot:
    root = Path(project_root).resolve()
    catalog = discover_runs(root)
    selected_keys = list(run_keys) if run_keys is not None else _default_run_keys(catalog)
    specs_by_key = {spec.run_key: spec for spec in _run_specs(root)}
    missing = sorted(set(selected_keys).difference(specs_by_key))
    if missing:
        raise KeyError(f"Runs desconocidos: {missing}")
    messages: list[str] = []
    trial_frames: list[pd.DataFrame] = []
    for key in selected_keys:
        trial_frames.append(_load_trials(specs_by_key[key], messages))
    trials = _decorate_trials(
        pd.concat([frame for frame in trial_frames if not frame.empty], ignore_index=True, sort=False)
        if any(not frame.empty for frame in trial_frames)
        else pd.DataFrame()
    )
    history_frames: list[pd.DataFrame] = []
    for key in selected_keys:
        history_frames.append(_load_histories(specs_by_key[key], trials, messages))
    histories = (
        pd.concat([frame for frame in history_frames if not frame.empty], ignore_index=True, sort=False)
        if any(not frame.empty for frame in history_frames)
        else pd.DataFrame()
    )
    diagnostics = diagnose_learning_curves(histories)
    summary = _trajectory_summary(diagnostics)
    fold_scores = _fold_scores(trials)
    final_progress_frames: list[pd.DataFrame] = []
    final_prediction_frames: list[pd.DataFrame] = []
    final_metric_frames: list[pd.DataFrame] = []
    for key in selected_keys:
        progress, predictions, metrics = _load_final_artifacts(
            specs_by_key[key], messages
        )
        final_progress_frames.append(progress)
        final_prediction_frames.append(predictions)
        final_metric_frames.append(metrics)
    final_progress = (
        pd.concat(
            [frame for frame in final_progress_frames if not frame.empty],
            ignore_index=True,
            sort=False,
        )
        if any(not frame.empty for frame in final_progress_frames)
        else pd.DataFrame()
    )
    final_predictions = (
        pd.concat(
            [frame for frame in final_prediction_frames if not frame.empty],
            ignore_index=True,
            sort=False,
        )
        if any(not frame.empty for frame in final_prediction_frames)
        else pd.DataFrame()
    )
    final_metrics = (
        pd.concat(
            [frame for frame in final_metric_frames if not frame.empty],
            ignore_index=True,
            sort=False,
        )
        if any(not frame.empty for frame in final_metric_frames)
        else pd.DataFrame()
    )
    selected_catalog = catalog.loc[catalog["run_key"].isin(selected_keys)].copy()
    if not selected_catalog.empty:
        counts = (
            trials.groupby(["run_key", "state"]).size().unstack(fill_value=0)
            if not trials.empty
            else pd.DataFrame()
        )
        for state in ("COMPLETE", "RUNNING", "PRUNED", "FAIL", "WAITING"):
            selected_catalog[f"trials_{state.lower()}"] = selected_catalog["run_key"].map(
                counts.get(state, {})
            ).fillna(0).astype(int)
    return TrajectorySnapshot(
        created_at=datetime.now(UTC).isoformat(),
        catalog=selected_catalog.reset_index(drop=True),
        trials=trials,
        histories=histories,
        fold_scores=fold_scores,
        diagnostics=diagnostics,
        trajectory_summary=summary,
        final_progress=final_progress,
        final_predictions=final_predictions,
        final_metrics=final_metrics,
        warnings=tuple(messages),
    )
