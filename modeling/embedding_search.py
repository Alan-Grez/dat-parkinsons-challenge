"""Resumable, auditable search for exploratory t-SNE and UMAP projections.

The search deliberately keeps two questions separate:

* ``structure_score`` rewards neighborhood preservation and stability while
  penalizing acquisition-family and voxel-spacing clustering.
* ``label_score`` measures exploratory class separation.  It uses the target
  only to evaluate a finished projection; labels are never passed to t-SNE or
  UMAP during fitting.

Consequently these embeddings are visualization/QC artefacts, not predictors
and not estimates of out-of-sample discrimination.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import pandas as pd
from optuna.storages.journal import (
    JournalFileBackend,
    JournalFileOpenLock,
    JournalStorage,
)
from scipy.stats import rankdata
from sklearn.manifold import TSNE, trustworthiness
from sklearn.metrics import balanced_accuracy_score, silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

METHODS = ("tsne", "umap")
PARAMETER_SPACE_VERSION = "tsne_umap_v1"


@dataclass(frozen=True)
class EmbeddingSearchResult:
    """Durable outputs returned to notebook 05 and the standalone runner."""

    comparison: pd.DataFrame
    coordinates: pd.DataFrame
    trials: pd.DataFrame
    best_parameters: dict[str, dict[str, dict[str, Any]]]
    output_dir: Path


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Cannot serialize {type(value)!r}")


def _replace_with_retry(temporary: Path, destination: Path, attempts: int = 10) -> None:
    delay_seconds = 0.25
    for attempt in range(attempts):
        try:
            temporary.replace(destination)
            return
        except PermissionError:
            if attempt == attempts - 1:
                raise
            time.sleep(delay_seconds)
            delay_seconds = min(delay_seconds * 1.8, 3.0)


def _atomic_csv(frame: pd.DataFrame, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    _replace_with_retry(temporary, destination)


def _atomic_json(payload: dict[str, Any], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_json_default),
        encoding="utf-8",
    )
    _replace_with_retry(temporary, destination)


def _candidate_components(n_features: int) -> list[int]:
    candidates = [2, 3, 4, 6, 8, 10, 12, n_features]
    return sorted({min(max(value, 2), n_features) for value in candidates})


def _prepare_input(
    X: np.ndarray, *, n_components: int, standardize: bool
) -> np.ndarray:
    prepared = np.asarray(X[:, :n_components], dtype=np.float32)
    if standardize:
        prepared = StandardScaler().fit_transform(prepared).astype(np.float32)
    return prepared


def _suggest_parameters(
    trial: optuna.Trial, method: str, *, n_rows: int, n_features: int
) -> dict[str, Any]:
    parameters: dict[str, Any] = {
        "n_input_components": trial.suggest_categorical(
            "n_input_components", _candidate_components(n_features)
        ),
        "standardize_components": trial.suggest_categorical(
            "standardize_components", [False, True]
        ),
    }
    if method == "tsne":
        maximum_perplexity = max(5, min(100, (n_rows - 1) // 3))
        perplexities = [
            value
            for value in [5, 10, 15, 20, 30, 40, 60, 80, 100]
            if value <= maximum_perplexity
        ]
        parameters.update(
            {
                "perplexity": trial.suggest_categorical("perplexity", perplexities),
                "early_exaggeration": trial.suggest_float(
                    "early_exaggeration", 6.0, 24.0
                ),
                "learning_rate": trial.suggest_float(
                    "learning_rate", 20.0, 1000.0, log=True
                ),
                "max_iter": trial.suggest_categorical(
                    "max_iter", [1000, 1500, 2000, 2500, 3000]
                ),
                "metric": trial.suggest_categorical(
                    "metric", ["euclidean", "cosine", "manhattan"]
                ),
                "init": trial.suggest_categorical("init", ["pca", "random"]),
                "angle": trial.suggest_float("angle", 0.3, 0.8),
            }
        )
        return parameters
    if method == "umap":
        neighbor_values = [
            value
            for value in [5, 10, 15, 25, 40, 60, 100, 150]
            if value < n_rows
        ]
        parameters.update(
            {
                "n_neighbors": trial.suggest_categorical(
                    "n_neighbors", neighbor_values
                ),
                "min_dist": trial.suggest_float("min_dist", 0.0, 0.8),
                "metric": trial.suggest_categorical(
                    "metric", ["euclidean", "cosine", "manhattan", "correlation"]
                ),
                "spread": trial.suggest_float("spread", 0.8, 2.0),
                "repulsion_strength": trial.suggest_float(
                    "repulsion_strength", 0.5, 2.0
                ),
                "negative_sample_rate": trial.suggest_categorical(
                    "negative_sample_rate", [5, 10, 15]
                ),
            }
        )
        return parameters
    raise ValueError(f"Unsupported embedding method: {method}")


def _fit_projection(
    method: str,
    X: np.ndarray,
    parameters: dict[str, Any],
    *,
    random_seed: int,
    cpu_jobs: int,
) -> np.ndarray:
    n_components = int(parameters["n_input_components"])
    prepared = _prepare_input(
        X,
        n_components=n_components,
        standardize=bool(parameters["standardize_components"]),
    )
    if method == "tsne":
        estimator = TSNE(
            n_components=2,
            perplexity=float(parameters["perplexity"]),
            early_exaggeration=float(parameters["early_exaggeration"]),
            learning_rate=float(parameters["learning_rate"]),
            max_iter=int(parameters["max_iter"]),
            metric=str(parameters["metric"]),
            init=str(parameters["init"]),
            angle=float(parameters["angle"]),
            method="barnes_hut",
            random_state=random_seed,
            n_jobs=cpu_jobs,
        )
    elif method == "umap":
        try:
            import umap
        except ImportError as error:
            raise RuntimeError(
                "UMAP no está instalado. Ejecuta `uv sync --cache-dir .uv-cache`."
            ) from error
        estimator = umap.UMAP(
            n_components=2,
            n_neighbors=int(parameters["n_neighbors"]),
            min_dist=float(parameters["min_dist"]),
            metric=str(parameters["metric"]),
            spread=float(parameters["spread"]),
            repulsion_strength=float(parameters["repulsion_strength"]),
            negative_sample_rate=int(parameters["negative_sample_rate"]),
            random_state=random_seed,
            transform_seed=random_seed,
            n_jobs=1,
            low_memory=True,
        )
    else:
        raise ValueError(f"Unsupported embedding method: {method}")
    coordinates = np.asarray(estimator.fit_transform(prepared), dtype=np.float32)
    if coordinates.shape != (len(X), 2) or not np.isfinite(coordinates).all():
        raise RuntimeError(f"{method} produced invalid coordinates.")
    return coordinates


def _neighbor_indices(coordinates: np.ndarray, n_neighbors: int) -> np.ndarray:
    model = NearestNeighbors(n_neighbors=min(n_neighbors + 1, len(coordinates)))
    model.fit(coordinates)
    return model.kneighbors(return_distance=False)[:, 1:]


def _neighbor_jaccard(neighborhoods: Sequence[np.ndarray]) -> float:
    if len(neighborhoods) < 2:
        return float("nan")
    values: list[float] = []
    for left_index in range(len(neighborhoods) - 1):
        left = neighborhoods[left_index]
        for right in neighborhoods[left_index + 1 :]:
            for left_row, right_row in zip(left, right):
                left_set = set(left_row.tolist())
                right_set = set(right_row.tolist())
                union = left_set | right_set
                values.append(len(left_set & right_set) / max(len(union), 1))
    return float(np.mean(values))


def _embedding_metrics(
    X: np.ndarray,
    coordinates_by_seed: Sequence[np.ndarray],
    y: np.ndarray,
    groups: np.ndarray,
    spacing: np.ndarray,
    *,
    evaluation_indices: np.ndarray,
    n_neighbors: int,
) -> dict[str, float]:
    label_silhouettes: list[float] = []
    label_balanced_accuracies: list[float] = []
    trustworthiness_values: list[float] = []
    family_confounds: list[float] = []
    spacing_confounds: list[float] = []
    all_neighborhoods: list[np.ndarray] = []

    eval_y = y[evaluation_indices]
    eval_groups = groups[evaluation_indices]
    eval_spacing = spacing[evaluation_indices]
    eval_X = X[evaluation_indices]
    unique_groups, group_counts = np.unique(eval_groups, return_counts=True)
    del unique_groups
    expected_same_family = float(np.sum((group_counts / len(eval_groups)) ** 2))
    spacing_pair_differences = np.abs(
        eval_spacing[:, np.newaxis] - eval_spacing[np.newaxis, :]
    )
    upper = spacing_pair_differences[np.triu_indices_from(spacing_pair_differences, k=1)]
    global_spacing_difference = float(np.nanmedian(upper))

    for coordinates in coordinates_by_seed:
        eval_coordinates = coordinates[evaluation_indices]
        neighborhoods = _neighbor_indices(eval_coordinates, n_neighbors)
        all_neighborhoods.append(neighborhoods)
        if np.unique(eval_y).size == 2 and min(np.bincount(eval_y)) >= 2:
            label_silhouettes.append(float(silhouette_score(eval_coordinates, eval_y)))
            neighbor_prevalence = eval_y[neighborhoods].mean(axis=1)
            label_balanced_accuracies.append(
                float(balanced_accuracy_score(eval_y, neighbor_prevalence >= 0.5))
            )
        trustworthiness_values.append(
            float(
                trustworthiness(
                    eval_X,
                    eval_coordinates,
                    n_neighbors=min(n_neighbors, (len(eval_X) - 1) // 2),
                )
            )
        )
        same_family = (eval_groups[neighborhoods] == eval_groups[:, np.newaxis]).mean()
        family_confounds.append(
            float(
                np.clip(
                    (same_family - expected_same_family)
                    / max(1.0 - expected_same_family, 1e-8),
                    0.0,
                    1.0,
                )
            )
        )
        local_spacing_difference = float(
            np.nanmedian(np.abs(eval_spacing[neighborhoods] - eval_spacing[:, np.newaxis]))
        )
        spacing_confounds.append(
            0.0
            if global_spacing_difference <= 1e-8
            else float(
                np.clip(
                    1.0 - local_spacing_difference / global_spacing_difference,
                    0.0,
                    1.0,
                )
            )
        )

    label_silhouette = float(np.mean(label_silhouettes))
    label_knn_balanced_accuracy = float(np.mean(label_balanced_accuracies))
    neighborhood_trustworthiness = float(np.mean(trustworthiness_values))
    seed_neighborhood_stability = _neighbor_jaccard(all_neighborhoods)
    acquisition_family_confound = float(np.mean(family_confounds))
    spacing_confound = float(np.mean(spacing_confounds))
    normalized_silhouette = float(np.clip((label_silhouette + 1.0) / 2.0, 0.0, 1.0))
    label_score = 0.5 * normalized_silhouette + 0.5 * label_knn_balanced_accuracy
    structure_score = (
        0.50 * neighborhood_trustworthiness
        + 0.30 * seed_neighborhood_stability
        + 0.10 * (1.0 - acquisition_family_confound)
        + 0.10 * (1.0 - spacing_confound)
    )
    balanced_score = 2.0 * label_score * structure_score / max(
        label_score + structure_score, 1e-8
    )
    return {
        "label_silhouette": label_silhouette,
        "label_knn_balanced_accuracy": label_knn_balanced_accuracy,
        "neighborhood_trustworthiness": neighborhood_trustworthiness,
        "seed_neighborhood_stability": seed_neighborhood_stability,
        "acquisition_family_confound": acquisition_family_confound,
        "spacing_confound": spacing_confound,
        "label_score": label_score,
        "structure_score": structure_score,
        "balanced_score": balanced_score,
    }


def _trial_rows(study: optuna.Study, method: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for trial in study.trials:
        row: dict[str, Any] = {
            "method": method,
            "trial_number": trial.number,
            "state": trial.state.name,
            "label_score": trial.values[0] if trial.values else np.nan,
            "structure_score": trial.values[1] if trial.values else np.nan,
            "parameters_json": json.dumps(
                trial.params, sort_keys=True, default=_json_default
            ),
        }
        row.update(trial.user_attrs)
        rows.append(row)
    return rows


def _selected_trials(study: optuna.Study) -> dict[str, optuna.trial.FrozenTrial]:
    complete = [
        trial
        for trial in study.trials
        if trial.state == optuna.trial.TrialState.COMPLETE and trial.values is not None
    ]
    if not complete:
        raise RuntimeError(f"Study {study.study_name!r} has no complete trials.")
    structure = max(complete, key=lambda trial: float(trial.values[1]))
    balanced = max(
        study.best_trials,
        key=lambda trial: float(trial.user_attrs.get("balanced_score", -np.inf)),
    )
    return {"balanced": balanced, "structure": structure}


def _evaluate_baseline(
    X: np.ndarray,
    coordinates: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    spacing: np.ndarray,
    *,
    evaluation_indices: np.ndarray,
    n_neighbors: int,
) -> dict[str, float]:
    # A single saved run has no between-seed stability.  It is not assigned a
    # composite structure/balanced score, but all directly comparable metrics
    # remain available in the comparison table.
    metrics = _embedding_metrics(
        X,
        [coordinates, coordinates],
        y,
        groups,
        spacing,
        evaluation_indices=evaluation_indices,
        n_neighbors=n_neighbors,
    )
    metrics["seed_neighborhood_stability"] = np.nan
    metrics["structure_score"] = np.nan
    metrics["balanced_score"] = np.nan
    return metrics


def run_embedding_search(
    X: np.ndarray,
    *,
    y: Sequence[int],
    groups: Sequence[str],
    spacing: Sequence[float],
    uids: Sequence[str],
    output_dir: Path,
    experiment_config: dict[str, Any],
    n_trials_per_method: int = 30,
    methods: Sequence[str] = METHODS,
    seeds: Sequence[int] = (20260821, 20260822, 20260823),
    cpu_jobs: int = 1,
    evaluation_rows: int = 800,
    evaluation_neighbors: int = 15,
    baseline_coordinates: np.ndarray | None = None,
) -> EmbeddingSearchResult:
    """Tune t-SNE/UMAP without fitting either method on pathology labels.

    Optuna receives two objectives (label separation and structural fidelity).
    The returned ``balanced`` choice is selected from the Pareto frontier; the
    independent ``structure`` choice ignores label separation during selection.
    """

    X_array = np.asarray(X, dtype=np.float32)
    y_array = np.asarray(y, dtype=int)
    group_array = np.asarray([str(value) for value in groups])
    spacing_array = np.asarray(spacing, dtype=float)
    uid_array = np.asarray([str(value) for value in uids])
    if X_array.ndim != 2 or X_array.shape[1] < 2:
        raise ValueError("X must be a two-dimensional matrix with at least 2 columns.")
    if not np.isfinite(X_array).all():
        raise ValueError("X contains non-finite values.")
    if len({len(y_array), len(group_array), len(spacing_array), len(uid_array), len(X_array)}) != 1:
        raise ValueError("X, y, groups, spacing and uids must have equal length.")
    if len(np.unique(uid_array)) != len(uid_array):
        raise ValueError("uids must be unique.")
    if np.unique(y_array).size != 2:
        raise ValueError("The exploratory label metrics require two classes.")
    if not np.isfinite(spacing_array).all():
        raise ValueError("spacing contains non-finite values.")
    unknown = sorted(set(methods) - set(METHODS))
    if unknown:
        raise ValueError(f"Unsupported methods: {unknown}")
    if n_trials_per_method < 1:
        raise ValueError("n_trials_per_method must be >= 1.")
    if len(seeds) < 2:
        raise ValueError("At least two seeds are required to measure stability.")

    output_dir = Path(output_dir)
    studies_dir = output_dir / "studies"
    studies_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(int(seeds[0]))
    evaluation_size = min(max(100, evaluation_rows), len(X_array))
    if evaluation_size < len(X_array):
        # Preserve approximate class balance in the fixed evaluation sample.
        selected: list[int] = []
        for label in np.unique(y_array):
            label_indices = np.flatnonzero(y_array == label)
            label_target = round(evaluation_size * len(label_indices) / len(y_array))
            selected.extend(
                rng.choice(
                    label_indices,
                    size=min(label_target, len(label_indices)),
                    replace=False,
                ).tolist()
            )
        if len(selected) < evaluation_size:
            remaining = np.setdiff1d(np.arange(len(X_array)), np.asarray(selected))
            selected.extend(
                rng.choice(remaining, size=evaluation_size - len(selected), replace=False).tolist()
            )
        evaluation_indices = np.sort(np.asarray(selected[:evaluation_size], dtype=int))
    else:
        evaluation_indices = np.arange(len(X_array))

    data_digest = hashlib.sha256()
    data_digest.update(np.ascontiguousarray(X_array).tobytes())
    data_digest.update("|".join(uid_array).encode("utf-8"))
    canonical_config = {
        **experiment_config,
        "parameter_space_version": PARAMETER_SPACE_VERSION,
        "methods": list(methods),
        "seeds": [int(seed) for seed in seeds],
        "evaluation_rows": int(evaluation_size),
        "evaluation_neighbors": int(evaluation_neighbors),
        "data_hash": data_digest.hexdigest(),
        "selection_uses_pathology_labels": True,
        "labels_passed_to_embedding_fit": False,
        "supervised_umap": False,
        "valid_for_visualization_only": True,
    }
    experiment_hash = hashlib.sha256(
        json.dumps(canonical_config, sort_keys=True, default=_json_default).encode("utf-8")
    ).hexdigest()
    persisted_config = {
        **canonical_config,
        "experiment_hash": experiment_hash,
        "target_complete_trials_per_method": int(n_trials_per_method),
    }
    config_path = output_dir / "embedding_search_config.json"
    if config_path.exists():
        previous = json.loads(config_path.read_text(encoding="utf-8"))
        if previous.get("experiment_hash") != experiment_hash:
            # Older notebook materializations included extra provenance fields
            # in ``experiment_config``.  If the data, algorithm version, seeds
            # and evaluation contract are identical, migrate that metadata
            # instead of discarding valid completed trials.
            stable_keys = (
                "data_hash",
                "parameter_space_version",
                "methods",
                "seeds",
                "evaluation_rows",
                "evaluation_neighbors",
                "labels_passed_to_embedding_fit",
                "supervised_umap",
            )
            compatible = all(previous.get(key) == persisted_config.get(key) for key in stable_keys)
            if not compatible:
                raise RuntimeError(
                    "The embedding checkpoints belong to another configuration. "
                    "Use a new embedding run id."
                )
    _atomic_json(persisted_config, config_path)

    studies: dict[str, optuna.Study] = {}
    all_trial_rows: list[dict[str, Any]] = []
    for method_index, method in enumerate(methods):
        journal_path = str(studies_dir / f"{method}.journal")
        storage = JournalStorage(
            JournalFileBackend(journal_path, lock_obj=JournalFileOpenLock(journal_path))
        )
        study = optuna.create_study(
            study_name=f"{method}-{experiment_hash[:16]}",
            storage=storage,
            sampler=optuna.samplers.TPESampler(seed=int(seeds[0]) + method_index),
            directions=("maximize", "maximize"),
            load_if_exists=True,
        )
        previous_hash = study.user_attrs.get("experiment_hash")
        if previous_hash not in (None, experiment_hash):
            raise RuntimeError(f"Study {study.study_name!r} belongs to another run.")
        study.set_user_attr("experiment_hash", experiment_hash)
        study.set_user_attr("method", method)
        study.set_user_attr("objectives", ["label_score", "structure_score"])

        progress_path = studies_dir / f"{method}_progress.json"

        def persist_progress(
            current_study: optuna.Study,
            method_name: str = method,
            destination: Path = progress_path,
        ) -> None:
            complete_trials = [
                trial
                for trial in current_study.trials
                if trial.state == optuna.trial.TrialState.COMPLETE
            ]
            _atomic_json(
                {
                    "method": method_name,
                    "completed_trials": len(complete_trials),
                    "target_complete_trials": int(n_trials_per_method),
                },
                destination,
            )

        def objective(trial: optuna.Trial, method_name: str = method) -> tuple[float, float]:
            parameters = _suggest_parameters(
                trial,
                method_name,
                n_rows=len(X_array),
                n_features=X_array.shape[1],
            )
            coordinates_by_seed = [
                _fit_projection(
                    method_name,
                    X_array,
                    parameters,
                    random_seed=int(seed),
                    cpu_jobs=cpu_jobs,
                )
                for seed in seeds
            ]
            metric_values = _embedding_metrics(
                _prepare_input(
                    X_array,
                    n_components=int(parameters["n_input_components"]),
                    standardize=bool(parameters["standardize_components"]),
                ),
                coordinates_by_seed,
                y_array,
                group_array,
                spacing_array,
                evaluation_indices=evaluation_indices,
                n_neighbors=evaluation_neighbors,
            )
            for name, value in metric_values.items():
                trial.set_user_attr(name, float(value))
            return metric_values["label_score"], metric_values["structure_score"]

        completed = sum(
            trial.state == optuna.trial.TrialState.COMPLETE for trial in study.trials
        )
        if completed < n_trials_per_method:
            study.optimize(
                objective,
                n_trials=n_trials_per_method - completed,
                n_jobs=1,
                gc_after_trial=True,
                show_progress_bar=True,
                callbacks=[lambda current_study, _trial: persist_progress(current_study)],
            )
        completed = sum(
            trial.state == optuna.trial.TrialState.COMPLETE for trial in study.trials
        )
        if completed < n_trials_per_method:
            raise RuntimeError(
                f"{method}: only {completed}/{n_trials_per_method} trials completed."
            )
        persist_progress(study)
        studies[method] = study
        method_rows = _trial_rows(study, method)
        all_trial_rows.extend(method_rows)
        _atomic_csv(pd.DataFrame(method_rows), studies_dir / f"{method}_trials.csv")

    trials = pd.DataFrame(all_trial_rows)
    _atomic_csv(trials, output_dir / "embedding_trials.csv")
    comparison_rows: list[dict[str, Any]] = []
    best_parameters: dict[str, dict[str, dict[str, Any]]] = {}
    coordinates = pd.DataFrame(
        {
            "uid": uid_array,
            "is_pathologic": y_array,
            "acquisition_family": group_array,
            "spacing_x_mm": spacing_array,
        }
    )
    for method, study in studies.items():
        selected = _selected_trials(study)
        best_parameters[method] = {}
        for selection, trial in selected.items():
            parameters = dict(trial.params)
            best_parameters[method][selection] = parameters
            selected_coordinates = _fit_projection(
                method,
                X_array,
                parameters,
                random_seed=int(seeds[0]),
                cpu_jobs=cpu_jobs,
            )
            column_prefix = f"{method}_{selection}"
            coordinates[f"{column_prefix}_1"] = selected_coordinates[:, 0]
            coordinates[f"{column_prefix}_2"] = selected_coordinates[:, 1]
            comparison_rows.append(
                {
                    "method": method,
                    "selection": selection,
                    "trial_number": trial.number,
                    **{
                        name: trial.user_attrs.get(name, np.nan)
                        for name in [
                            "label_silhouette",
                            "label_knn_balanced_accuracy",
                            "neighborhood_trustworthiness",
                            "seed_neighborhood_stability",
                            "acquisition_family_confound",
                            "spacing_confound",
                            "label_score",
                            "structure_score",
                            "balanced_score",
                        ]
                    },
                    "parameters_json": json.dumps(
                        parameters, sort_keys=True, default=_json_default
                    ),
                    "labels_used_to_fit_embedding": False,
                    "labels_used_to_select_trial": selection == "balanced",
                }
            )

    if baseline_coordinates is not None:
        baseline = np.asarray(baseline_coordinates, dtype=np.float32)
        valid = np.isfinite(baseline).all(axis=1)
        if valid.all() and baseline.shape == (len(X_array), 2):
            baseline_metrics = _evaluate_baseline(
                X_array,
                baseline,
                y_array,
                group_array,
                spacing_array,
                evaluation_indices=evaluation_indices,
                n_neighbors=evaluation_neighbors,
            )
            comparison_rows.append(
                {
                    "method": "tsne_current",
                    "selection": "baseline",
                    "trial_number": np.nan,
                    **baseline_metrics,
                    "parameters_json": "{}",
                    "labels_used_to_fit_embedding": False,
                    "labels_used_to_select_trial": False,
                }
            )

    comparison = pd.DataFrame(comparison_rows).sort_values(
        ["selection", "balanced_score"], ascending=[True, False], na_position="last"
    )
    _atomic_csv(comparison, output_dir / "embedding_method_comparison.csv")
    _atomic_csv(coordinates, output_dir / "embedding_search_coordinates.csv")
    _atomic_json(best_parameters, output_dir / "embedding_best_parameters.json")
    return EmbeddingSearchResult(
        comparison=comparison,
        coordinates=coordinates,
        trials=trials,
        best_parameters=best_parameters,
        output_dir=output_dir,
    )


def rank_trial_metrics(trials: pd.DataFrame) -> pd.DataFrame:
    """Return a compact ranking helper useful for external diagnostics."""

    complete = trials.loc[trials["state"] == "COMPLETE"].copy()
    if complete.empty:
        return complete
    complete["label_rank"] = rankdata(-complete["label_score"], method="average")
    complete["structure_rank"] = rankdata(-complete["structure_score"], method="average")
    complete["mean_rank"] = (complete["label_rank"] + complete["structure_rank"]) / 2
    return complete.sort_values("mean_rank")
