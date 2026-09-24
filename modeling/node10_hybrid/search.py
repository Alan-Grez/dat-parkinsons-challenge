from __future__ import annotations

import json
import pickle
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import pandas as pd
from optuna.exceptions import ExperimentalWarning
from optuna.trial import TrialState
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.preprocessing import RobustScaler

from modeling.cnn.io import atomic_csv, atomic_json, replace_with_retry
from modeling.dat_spect_v2.config import stable_hash

from .config import ExperimentConfig, SearchConfig
from .diffusion import DiffusionMapTransformer
from .training import (
    FoldResult,
    load_graph_arrays,
    train_dual_fold,
    train_torch_tabular_fold,
)


@dataclass
class SearchResult:
    summary: pd.DataFrame
    finalists: list[dict[str, Any]]
    database_path: Path


def _storage(path: Path, config: SearchConfig) -> optuna.storages.RDBStorage:
    path.parent.mkdir(parents=True, exist_ok=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ExperimentalWarning)
        retry = optuna.storages.RetryHeartbeatStaleTrialCallback(max_retry=2)
        return optuna.storages.RDBStorage(
            f"sqlite:///{path.resolve().as_posix()}",
            engine_kwargs={"connect_args": {"timeout": 60}},
            heartbeat_interval=60,
            grace_period=config.stale_trial_grace_seconds,
            heartbeat_stale_trial_callback=retry,
        )


def completed_trials(study: optuna.Study) -> list[optuna.trial.FrozenTrial]:
    return [trial for trial in study.trials if trial.state == TrialState.COMPLETE]


def optimize_until_complete(
    study: optuna.Study,
    objective: Any,
    *,
    target: int,
    timeout_seconds: int | None,
) -> None:
    """Keep replacing PRUNED trials until the COMPLETE target is reached.

    A failed objective is intentionally not swallowed: a systematic error must
    stop the run instead of looping forever. With no timeout (the default), a
    normally functioning study can only return after reaching the target.
    """

    target = max(10, int(target))
    started = time.monotonic()
    while len(completed_trials(study)) < target:
        if timeout_seconds is not None and time.monotonic() - started >= timeout_seconds:
            raise TimeoutError(
                f"Optuna agoto el timeout con {len(completed_trials(study))}/{target} trials COMPLETE. "
                "Reanuda la misma celda para continuar."
            )
        study.optimize(objective, n_trials=1, gc_after_trial=True, show_progress_bar=False)
    if len(completed_trials(study)) < target:
        raise RuntimeError("Contrato imposible: el estudio termino sin alcanzar trials COMPLETE.")


def _suggest(trial: optuna.Trial, family: str, experiment: ExperimentConfig) -> dict[str, Any]:
    if family == "dual_stream_multitask":
        return {
            "base_channels": trial.suggest_categorical("base_channels", [8, 12, 16]),
            "embedding_dim": trial.suggest_categorical("embedding_dim", [64, 96, 128]),
            "dropout": trial.suggest_float("dropout", 0.15, 0.42),
            "learning_rate": trial.suggest_float("learning_rate", 5e-5, 4e-4, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
            "auxiliary_weight": trial.suggest_float("auxiliary_weight", 0.05, 0.30),
            "consistency_weight": trial.suggest_float("consistency_weight", 0.005, 0.10, log=True),
            "augmentation_probability": trial.suggest_float("augmentation_probability", 0.80, 0.97),
            "augmentation_magnitude": trial.suggest_float("augmentation_magnitude", 2.0, 3.0),
            "rotation_degrees": trial.suggest_float("rotation_degrees", 0.0, 3.0),
        }
    if family in {"regional_hgb", "topology_hgb"}:
        return {
            "learning_rate": trial.suggest_float("learning_rate", 0.015, 0.12, log=True),
            "max_iter": trial.suggest_int("max_iter", 80, 260),
            "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 7, 31),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 12, 48),
            "l2_regularization": trial.suggest_float("l2_regularization", 1e-4, 3.0, log=True),
        }
    if family == "graph_elasticnet":
        return {
            "C": trial.suggest_float("C", 0.01, 20.0, log=True),
            "l1_ratio": trial.suggest_float("l1_ratio", 0.0, 1.0),
        }
    if family == "graph_random_forest":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 300, 900, step=100),
            "max_depth": trial.suggest_int("max_depth", 4, 18),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 2, 20),
            "max_features": trial.suggest_float("max_features", 0.25, 0.9),
        }
    if family in {"graph_mlp", "graph_gcn"}:
        return {
            "hidden_dim": trial.suggest_categorical("hidden_dim", [32, 48, 64, 96, 128]),
            "dropout": trial.suggest_float("dropout", 0.15, 0.55),
            "learning_rate": trial.suggest_float("learning_rate", 3e-5, 8e-4, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-7, 3e-3, log=True),
        }
    if family == "diffusion_map":
        return {
            "pca_variance": trial.suggest_categorical("pca_variance", [0.90, 0.95, 0.98]),
            "n_components": trial.suggest_int("n_components", 4, 18),
            "epsilon_quantile": trial.suggest_float("epsilon_quantile", 0.25, 0.80),
            "diffusion_time": trial.suggest_float("diffusion_time", 0.5, 3.0),
            "C": trial.suggest_float("C", 0.02, 20.0, log=True),
        }
    if family == "subtype_mixture":
        return {
            "hidden_dim": trial.suggest_categorical("hidden_dim", [48, 64, 96, 128]),
            "n_subtypes": trial.suggest_int("n_subtypes", 3, 6),
            "dropout": trial.suggest_float("dropout", 0.15, 0.45),
            "learning_rate": trial.suggest_float("learning_rate", 3e-5, 6e-4, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-7, 2e-3, log=True),
            "balance_weight": trial.suggest_float("balance_weight", 0.005, 0.10, log=True),
            "diversity_weight": trial.suggest_float("diversity_weight", 0.002, 0.04, log=True),
            "pca_variance": trial.suggest_categorical("pca_variance", [0.90, 0.95, 0.98]),
        }
    raise ValueError(f"Familia desconocida: {family}")


def _feature_columns(cohort: pd.DataFrame, family: str) -> list[str]:
    topology = [column for column in cohort if column.startswith("topo_")]
    regional = [column for column in cohort if column.startswith("regional930_")]
    graph = [column for column in cohort if column.startswith("graph_n")]
    magnitude = [
        "intensity_l2_norm",
        "intensity_mean",
        "intensity_p90",
        "intensity_positive_voxels",
    ]
    if family == "regional_hgb":
        return regional
    if family == "topology_hgb":
        return topology + magnitude
    if family in {"graph_elasticnet", "graph_random_forest", "graph_mlp", "graph_gcn"}:
        return graph
    if family in {"diffusion_map", "subtype_mixture"}:
        return regional + topology + graph + magnitude
    return []


def _atomic_pickle(value: Any, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f"{destination.name}.tmp")
    with temporary.open("wb") as stream:
        pickle.dump(value, stream, protocol=pickle.HIGHEST_PROTOCOL)
    replace_with_retry(temporary, destination)


def _sklearn_fold(
    family: str,
    train_frame: pd.DataFrame,
    valid_frame: pd.DataFrame,
    *,
    fold: int,
    parameters: dict[str, Any],
    output_dir: Path,
    seed: int,
) -> FoldResult:
    prediction_path = output_dir / "validation_predictions.csv"
    model_path = output_dir / "model.pkl"
    if prediction_path.exists() and model_path.exists():
        predictions = pd.read_csv(prediction_path)
        score = float(
            log_loss(predictions["is_pathologic"], predictions["probability"], labels=[0, 1])
        )
        return FoldResult(fold, score, -1, predictions, model_path)
    columns = _feature_columns(train_frame, family)
    x_train = train_frame[columns].to_numpy(dtype=np.float64)
    x_valid = valid_frame[columns].to_numpy(dtype=np.float64)
    y_train = train_frame["is_pathologic"].to_numpy(dtype=int)
    y_valid = valid_frame["is_pathologic"].to_numpy(dtype=int)
    median = np.nanmedian(x_train, axis=0)
    median = np.where(np.isfinite(median), median, 0.0)
    x_train = np.where(np.isfinite(x_train), x_train, median)
    x_valid = np.where(np.isfinite(x_valid), x_valid, median)
    transformer: Any = None
    if family in {"regional_hgb", "topology_hgb"}:
        model = HistGradientBoostingClassifier(
            learning_rate=float(parameters["learning_rate"]),
            max_iter=int(parameters["max_iter"]),
            max_leaf_nodes=int(parameters["max_leaf_nodes"]),
            min_samples_leaf=int(parameters["min_samples_leaf"]),
            l2_regularization=float(parameters["l2_regularization"]),
            class_weight="balanced",
            early_stopping=False,
            random_state=seed,
        )
    elif family == "graph_random_forest":
        model = RandomForestClassifier(
            n_estimators=int(parameters["n_estimators"]),
            max_depth=int(parameters["max_depth"]),
            min_samples_leaf=int(parameters["min_samples_leaf"]),
            max_features=float(parameters["max_features"]),
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=seed,
        )
    elif family == "graph_elasticnet":
        transformer = RobustScaler().fit(x_train)
        x_train, x_valid = transformer.transform(x_train), transformer.transform(x_valid)
        model = LogisticRegression(
            C=float(parameters["C"]),
            l1_ratio=float(parameters["l1_ratio"]),
            penalty="elasticnet",
            solver="saga",
            class_weight="balanced",
            max_iter=4000,
            random_state=seed,
        )
    elif family == "diffusion_map":
        scaler = RobustScaler().fit(x_train)
        scaled_train, scaled_valid = scaler.transform(x_train), scaler.transform(x_valid)
        pca = PCA(
            n_components=float(parameters["pca_variance"]),
            svd_solver="full",
            random_state=seed,
        ).fit(scaled_train)
        scaled_train, scaled_valid = pca.transform(scaled_train), pca.transform(scaled_valid)
        diffusion = DiffusionMapTransformer(
            n_components=int(parameters["n_components"]),
            epsilon_quantile=float(parameters["epsilon_quantile"]),
            diffusion_time=float(parameters["diffusion_time"]),
        ).fit(scaled_train)
        x_train, x_valid = diffusion.transform(scaled_train), diffusion.transform(scaled_valid)
        transformer = {"scaler": scaler, "pca": pca, "diffusion": diffusion}
        model = LogisticRegression(
            C=float(parameters["C"]),
            penalty="l2",
            class_weight="balanced",
            max_iter=3000,
            random_state=seed,
        )
    else:
        raise ValueError(f"{family} no es una familia sklearn.")
    model.fit(x_train, y_train)
    probability = model.predict_proba(x_valid)[:, 1].clip(1e-6, 1 - 1e-6)
    predictions = pd.DataFrame(
        {
            "uid": valid_frame["uid"].astype(str),
            "is_pathologic": y_valid,
            "probability": probability,
            "fold": fold,
            "selected_epoch": -1,
        }
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_pickle(
        {"model": model, "transformer": transformer, "median": median, "feature_columns": columns},
        model_path,
    )
    # Predictions are the completion marker and are written only after the
    # deployable model plus preprocessing state reached disk atomically.
    atomic_csv(predictions, prediction_path)
    score = float(log_loss(y_valid, probability, labels=[0, 1]))
    return FoldResult(fold, score, -1, predictions, model_path)


def evaluate_family_fold(
    cohort: pd.DataFrame,
    folds: pd.DataFrame,
    *,
    family: str,
    fold: int,
    parameters: dict[str, Any],
    experiment: ExperimentConfig,
    output_dir: Path,
    fixed_epochs: int | None = None,
    device: str | None = None,
) -> FoldResult:
    if family == "dual_stream_multitask":
        return train_dual_fold(
            cohort,
            folds,
            fold=fold,
            train_config=experiment.train,
            parameters=parameters,
            output_dir=output_dir,
            fixed_epochs=fixed_epochs,
            device=device,
        )
    merged = cohort.merge(folds[["uid", "fold"]], on="uid", validate="one_to_one")
    train_frame = merged.loc[merged["fold"] != fold].reset_index(drop=True)
    valid_frame = merged.loc[merged["fold"] == fold].reset_index(drop=True)
    if family in {
        "regional_hgb",
        "topology_hgb",
        "graph_elasticnet",
        "graph_random_forest",
        "diffusion_map",
    }:
        return _sklearn_fold(
            family,
            train_frame,
            valid_frame,
            fold=fold,
            parameters=parameters,
            output_dir=output_dir,
            seed=experiment.train.seed + fold * 3011,
        )
    columns = _feature_columns(cohort, family)
    x_train = train_frame[columns].to_numpy(dtype=np.float64)
    x_valid = valid_frame[columns].to_numpy(dtype=np.float64)
    median = np.nanmedian(x_train, axis=0)
    median = np.where(np.isfinite(median), median, 0.0)
    x_train = np.where(np.isfinite(x_train), x_train, median)
    x_valid = np.where(np.isfinite(x_valid), x_valid, median)
    adjacency_train = adjacency_valid = None
    preprocessing_state: dict[str, Any]
    if family == "graph_gcn":
        nodes_train, adjacency_train = load_graph_arrays(train_frame)
        nodes_valid, adjacency_valid = load_graph_arrays(valid_frame)
        # The first eight features are non-negative raw-count moments. A
        # deterministic log transform prevents the squared-intensity feature
        # from dominating while preserving order and remaining fold-safe.
        nodes_train = nodes_train.astype(np.float64, copy=True)
        nodes_valid = nodes_valid.astype(np.float64, copy=True)
        nodes_train[..., :8] = np.log1p(np.maximum(nodes_train[..., :8], 0.0))
        nodes_valid[..., :8] = np.log1p(np.maximum(nodes_valid[..., :8], 0.0))
        center = np.median(nodes_train.reshape(-1, nodes_train.shape[-1]), axis=0)
        q25, q75 = np.percentile(nodes_train.reshape(-1, nodes_train.shape[-1]), [25, 75], axis=0)
        scale = np.where((q75 - q25) > 1e-6, q75 - q25, 1.0)
        clip_value = 12.0
        x_train = np.clip((nodes_train - center) / scale, -clip_value, clip_value).astype(
            np.float32
        )
        x_valid = np.clip((nodes_valid - center) / scale, -clip_value, clip_value).astype(
            np.float32
        )
        preprocessing_state = {
            "family": family,
            "node_center": center,
            "node_scale": scale,
            "node_shape": list(nodes_train.shape[1:]),
            "raw_moment_transform": "log1p_nonnegative_features_0_7",
            "robust_clip": clip_value,
        }
    else:
        scaler = RobustScaler().fit(x_train)
        scaled_train = scaler.transform(x_train)
        scaled_valid = scaler.transform(x_valid)
        pca = None
        if family == "subtype_mixture":
            pca = PCA(
                n_components=float(parameters["pca_variance"]),
                svd_solver="full",
                random_state=experiment.train.seed + fold * 3011,
            ).fit(scaled_train)
            scaled_train = pca.transform(scaled_train)
            scaled_valid = pca.transform(scaled_valid)
        x_train = scaled_train.astype(np.float32)
        x_valid = scaled_valid.astype(np.float32)
        preprocessing_state = {
            "family": family,
            "median": median,
            "scaler": scaler,
            "pca": pca,
            "feature_columns": columns,
        }
    return train_torch_tabular_fold(
        family,
        x_train,
        x_valid,
        train_frame["is_pathologic"].to_numpy(dtype=int),
        valid_frame["is_pathologic"].to_numpy(dtype=int),
        valid_frame["uid"].astype(str).to_numpy(),
        fold=fold,
        parameters=parameters,
        train_config=experiment.train,
        output_dir=output_dir,
        train_adjacency=adjacency_train,
        valid_adjacency=adjacency_valid,
        preprocessing_state=preprocessing_state,
        fixed_epochs=fixed_epochs,
        device=device,
    )


def _trial_table(study: optuna.Study) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "trial": trial.number,
                "state": trial.state.name,
                "value": trial.value,
                "duration_seconds": trial.duration.total_seconds() if trial.duration else np.nan,
                "parameters": json.dumps(trial.params, sort_keys=True),
                "candidate": json.dumps(trial.user_attrs.get("candidate", {}), sort_keys=True),
            }
            for trial in study.trials
        ]
    )


def run_hybrid_search(
    cohort: pd.DataFrame,
    folds: pd.DataFrame,
    *,
    experiment: ExperimentConfig,
    run_dir: Path,
    device: str | None = None,
) -> SearchResult:
    run_dir.mkdir(parents=True, exist_ok=True)
    database = run_dir / "optuna_node10.sqlite3"
    storage = _storage(database, experiment.search)
    studies_dir, trials_dir = run_dir / "studies", run_dir / "trials"
    studies_dir.mkdir(parents=True, exist_ok=True)
    finalists: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for family_index, family in enumerate(experiment.model_families):
        study_name = f"node10_{family}"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ExperimentalWarning)
            sampler = optuna.samplers.TPESampler(
                seed=experiment.train.seed + family_index,
                n_startup_trials=min(5, experiment.search.effective_completed_trials),
                multivariate=True,
                constant_liar=True,
            )
        study = optuna.create_study(
            study_name=study_name,
            storage=storage,
            load_if_exists=True,
            direction="minimize",
            sampler=sampler,
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=experiment.search.pruning_startup_trials,
                n_warmup_steps=experiment.search.pruning_warmup_folds,
            ),
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ExperimentalWarning)
            optuna.storages.fail_stale_trials(study)

        def objective(
            trial: optuna.Trial,
            family: str = family,
            study_name: str = study_name,
        ) -> float:
            parameters = _suggest(trial, family, experiment)
            parameter_hash = stable_hash({"family": family, "parameters": parameters})[:20]
            scores: list[float] = []
            epochs: list[int] = []
            for fold in range(experiment.search.n_splits_search):
                result = evaluate_family_fold(
                    cohort,
                    folds,
                    family=family,
                    fold=fold,
                    parameters=parameters,
                    experiment=experiment,
                    output_dir=trials_dir / study_name / parameter_hash / f"fold_{fold}",
                    device=device,
                )
                scores.append(float(result.validation_log_loss))
                if result.selected_epoch >= 0:
                    epochs.append(int(result.selected_epoch) + 1)
                trial.report(float(np.mean(scores)), step=fold)
                if trial.should_prune():
                    raise optuna.TrialPruned()
            candidate = {
                "family": family,
                "parameters": parameters,
                "fold_log_loss": scores,
                "fixed_epochs": int(max(4, round(float(np.median(epochs))))) if epochs else None,
            }
            trial.set_user_attr("candidate", candidate)
            return float(np.mean(scores))

        optimize_until_complete(
            study,
            objective,
            target=experiment.search.effective_completed_trials,
            timeout_seconds=experiment.search.optuna_timeout_seconds,
        )
        complete = completed_trials(study)
        if len(complete) < experiment.search.effective_completed_trials:
            raise RuntimeError(f"{study_name}: no alcanzo el minimo de trials COMPLETE.")
        best_trial = min(complete, key=lambda item: float(item.value))
        finalist = dict(best_trial.user_attrs["candidate"])
        finalist.update(
            {
                "study_name": study_name,
                "trial_number": int(best_trial.number),
                "search_log_loss": float(best_trial.value),
            }
        )
        finalist["candidate_id"] = stable_hash(finalist)[:16]
        finalists.append(finalist)
        atomic_csv(_trial_table(study), studies_dir / f"{study_name}_trials.csv")
        atomic_json(finalist, studies_dir / f"{study_name}_best.json")
        summary_rows.append(
            {
                "study_name": study_name,
                "family": family,
                "completed_trials": len(complete),
                "total_trials": len(study.trials),
                "search_log_loss": finalist["search_log_loss"],
                "fixed_epochs": finalist["fixed_epochs"],
                "candidate_id": finalist["candidate_id"],
            }
        )
    summary = pd.DataFrame(summary_rows).sort_values("search_log_loss").reset_index(drop=True)
    atomic_csv(summary, run_dir / "search_summary.csv")
    atomic_json({"finalists": finalists}, run_dir / "finalists.json")
    return SearchResult(summary, finalists, database)
