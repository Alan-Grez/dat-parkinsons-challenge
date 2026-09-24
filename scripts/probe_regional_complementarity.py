"""Small fixed ablations on derived features, using CNN06's group-held-out folds.

This is a development stress test, not nested HPO or an external evaluation.
No raw scans are read. Historical hyperparameters are frozen before these fits.
Each complete fold is resumable and input/configuration hashes are checked.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from modeling.cnn.io import atomic_csv, atomic_json
from scripts.audit_project_log_loss import (
    BASE,
    CNN,
    HGB,
    bootstrap_gain,
    crossfit,
    load_predictions,
    losses,
    metrics,
)


def read_features():
    paths = sorted((BASE / "node10_cache/v3").glob("*/regional930_features.csv"))
    if len(paths) != 1:
        raise ValueError("Specify the intended node10 cache before probing multiple versions")
    cache = paths[0].parent
    regional = pd.read_csv(paths[0]).set_index("uid").sort_index()
    magnitude = pd.read_csv(cache / "manifest.csv").set_index("uid").sort_index()
    graph = pd.read_csv(cache / "graph_features.csv").set_index("uid").sort_index()
    topo = pd.read_csv(cache / "topology_features.csv").set_index("uid").sort_index()
    magnitude = magnitude[[c for c in magnitude if c.startswith("intensity_")]]
    return regional.join(magnitude, validate="one_to_one").join(graph, validate="one_to_one").join(
        topo, validate="one_to_one"
    ), paths[0]


def run(out, experiments):
    out.mkdir(parents=True, exist_ok=True)
    frames, _ = load_predictions()
    f = frames[CNN]
    y = f.is_pathologic.to_numpy(int)
    folds = f.fold.to_numpy(int)
    groups = f.acquisition_family.to_numpy(str)
    assert pd.DataFrame({"group": groups, "fold": folds}).groupby("group").fold.nunique().max() == 1
    features, source = read_features()
    assert features.index.equals(f.index)
    params = json.loads(
        (
            BASE / "node10_runs/node10_hybrid_v1/search/studies/node10_regional_hgb_best.json"
        ).read_text()
    )["parameters"]
    contract = {
        "params": params,
        "source": str(source),
        "feature_hash": hashlib.sha256(
            pd.util.hash_pandas_object(features, index=True).values.tobytes()
        ).hexdigest(),
        "fold_hash": hashlib.sha256(
            pd.util.hash_pandas_object(f[["fold", "is_pathologic"]], index=True).values.tobytes()
        ).hexdigest(),
        "experiments": experiments,
        "seed": 20260905,
        "version": 1,
        "validation": "same group-held-out folds as CNN06; fixed historical HPO parameters; exploratory",
    }
    path = out / "contract.json"
    if path.exists() and json.loads(path.read_text()) != contract:
        raise ValueError("Probe inputs/configuration changed: use a new output directory")
    atomic_json(contract, path)
    stats = pd.DataFrame(
        {
            "feature": features.columns,
            "nonfinite_fraction": 1 - np.isfinite(features).mean(),
            "unique_values": features.nunique(),
            "std": features.std(),
        }
    ).reset_index(drop=True)
    atomic_csv(stats, out / "feature_audit.csv")
    region = [c for c in features if c.startswith("regional930_")]
    mag = [c for c in features if c.startswith("intensity_")]
    topo = [c for c in features if c.startswith("topo_")]
    results, fold_results, boots = [], [], []
    predictions = pd.DataFrame(
        {
            "uid": f.index,
            "is_pathologic": y,
            "fold": folds,
            "acquisition_family": groups,
            "cnn06": f.probability.to_numpy(),
        }
    )
    for name in experiments:
        cols = region.copy()
        if name == "regional_magnitude":
            cols += mag
        if name == "regional_topology":
            cols += topo
        p = np.full(len(y), np.nan)
        importances = []
        for fold in np.unique(folds):
            folder = out / name / f"fold_{fold}"
            folder.mkdir(parents=True, exist_ok=True)
            done = folder / "predictions.csv"
            mask = folds != fold
            if done.exists() and (folder / "model.pkl").exists():
                saved = pd.read_csv(done).set_index("uid").reindex(f.index[~mask])
                if saved.probability.isna().any():
                    raise ValueError("Incomplete fold marker")
                p[~mask] = saved.probability
            else:
                train = features.loc[mask, cols].to_numpy(float)
                test = features.loc[~mask, cols].to_numpy(float)
                # HGB handles missing values, infinities are converted to missing.
                train = np.where(np.isfinite(train), train, np.nan)
                test = np.where(np.isfinite(test), test, np.nan)
                if name == "regional_xgb":
                    model = XGBClassifier(
                        n_estimators=250,
                        max_depth=3,
                        learning_rate=0.03,
                        min_child_weight=20,
                        reg_lambda=10.0,
                        subsample=0.85,
                        colsample_bytree=0.8,
                        objective="binary:logistic",
                        eval_metric="logloss",
                        tree_method="hist",
                        n_jobs=4,
                        random_state=20260905 + int(fold),
                    )
                else:
                    model = HistGradientBoostingClassifier(
                        **params,
                        early_stopping=False,
                        class_weight="balanced" if name == "regional_balanced" else None,
                        random_state=20260905 + int(fold),
                    )
                model.fit(train, y[mask])
                p[~mask] = model.predict_proba(test)[:, 1]
                payload = {
                    "model": model,
                    "feature_columns": cols,
                    "contract": contract,
                    "fold": int(fold),
                }
                temporary = folder / "model.pkl.tmp"
                with temporary.open("wb") as stream:
                    pickle.dump(payload, stream)
                temporary.replace(folder / "model.pkl")
                atomic_csv(pd.DataFrame({"uid": f.index[~mask], "probability": p[~mask]}), done)
                if name == "regional_unweighted":
                    # Separate train-free diagnostic on held-out predictions; no feature selection.
                    rng = np.random.default_rng(73 + int(fold))
                    base = float(losses(y[~mask], p[~mask]).mean())
                    blocks = {
                        "intensity_selfnorm": [i for i, c in enumerate(cols) if "selfnorm" in c],
                        "intensity_robust": [i for i, c in enumerate(cols) if "robust" in c],
                        "rank": [i for i, c in enumerate(cols) if "rank" in c],
                        "texture": [i for i, c in enumerate(cols) if "glcm" in c or "texture" in c],
                        "morphology": [i for i, c in enumerate(cols) if "morph" in c],
                    }
                    for block, ids in blocks.items():
                        if not ids:
                            continue
                        shifted = test.copy()
                        shifted[:, ids] = test[rng.permutation(len(test))][:, ids]
                        score = float(losses(y[~mask], model.predict_proba(shifted)[:, 1]).mean())
                        importances.append(
                            {
                                "fold": int(fold),
                                "block": block,
                                "features": len(ids),
                                "log_loss_increase": score - base,
                            }
                        )
            score = float(losses(y[~mask], p[~mask]).mean())
            fold_results.append(
                {"experiment": name, "fold": int(fold), "n": int((~mask).sum()), "log_loss": score}
            )
            print(f"{name} fold={fold} loss={score:.5f}", flush=True)
        predictions[name] = p
        variants = {name: p, name + "_cnn_equal": (p + f.probability.to_numpy()) / 2}
        for variant, p1 in variants.items():
            temp = crossfit(p1, y, folds, "temperature")[0]
            results.append(
                {
                    "experiment": variant,
                    **metrics(y, p1),
                    "temperature_log_loss": metrics(y, temp)["log_loss"],
                }
            )
            predictions[variant] = p1
            predictions[variant + "_temperature"] = temp
            boots.append(
                {
                    "experiment": variant,
                    **bootstrap_gain(y, frames[HGB].probability.to_numpy(), p1, groups),
                }
            )
        if importances:
            atomic_csv(pd.DataFrame(importances), out / name / "block_importance.csv")
        atomic_csv(pd.DataFrame(results), out / "probe_metrics.csv")
        atomic_csv(pd.DataFrame(fold_results), out / "probe_fold_metrics.csv")
        atomic_csv(predictions, out / "probe_oof_private.csv")
        atomic_csv(pd.DataFrame(boots), out / "probe_bootstrap.csv")
    print(pd.DataFrame(results).sort_values("log_loss").to_string(index=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", type=Path, default=BASE / "project_audit_20260905/grouped_probe"
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        default=[
            "regional_balanced",
            "regional_unweighted",
            "regional_magnitude",
            "regional_topology",
            "regional_xgb",
        ],
    )
    args = parser.parse_args()
    with threadpool_limits(limits=4):
        run(args.output, args.experiments)
