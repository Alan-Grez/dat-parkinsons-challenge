"""Reproducible local audit of saved predictions; never reads raw competition files.

Recombination and calibration of historical OOF are exploratory: different base
folds and repeated model selection preclude an independent performance claim.
All patient-level artifacts remain under outputs/private_eda.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar
from scipy.special import expit, logit
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
BASE = ROOT / "outputs/private_eda"
HGB = "10_regional_hgb"
CNN = "06_43765c3f"
SLAB = "09_555b8137"


def losses(y, p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return -(y * np.log(p) + (1 - y) * np.log1p(-p))


def metrics(y, p):
    return {
        "log_loss": float(losses(y, p).mean()),
        "auc": float(roc_auc_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
    }


def fit_calibrator(p, y, method):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    z = logit(p)
    if method == "temperature":
        # NLL is convex in inverse temperature (not in log temperature).
        result = minimize_scalar(
            lambda b: np.mean(np.logaddexp(0, b * z) - y * b * z),
            bounds=(0.05, 20.0),
            method="bounded",
        )
        candidates = [0.05, 1.0, 20.0, float(result.x)]
        b = min(candidates, key=lambda v: np.mean(np.logaddexp(0, v * z) - y * v * z))
        return np.array([b])
    if method == "platt":
        x = np.column_stack([z, np.ones(len(z))])
        anchor, bounds = np.array([1.0, 0.0]), [(0.05, 20.0), (-3.0, 3.0)]
    elif method == "beta":
        x = np.column_stack([np.log(p), -np.log1p(-p), np.ones(len(p))])
        anchor, bounds = np.array([1.0, 1.0, 0.0]), [(0.05, 20.0), (0.05, 20.0), (-3.0, 3.0)]
    else:
        raise ValueError(method)

    def objective(w):
        z1 = x @ w
        delta = w - anchor
        return (
            np.mean(np.logaddexp(0, z1) - y * z1) + 0.002 * np.dot(delta, delta),
            x.T @ (expit(z1) - y) / len(y) + 0.004 * delta,
        )

    result = minimize(objective, anchor, jac=True, method="L-BFGS-B", bounds=bounds)
    if not result.success:
        raise RuntimeError(result.message)
    return result.x


def apply_calibrator(p, params, method):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    if method == "temperature":
        return expit(params[0] * logit(p))
    if method == "platt":
        return expit(params[0] * logit(p) + params[1])
    return expit(params[0] * np.log(p) - params[1] * np.log1p(-p) + params[2])


def crossfit(p, y, folds, method):
    out = np.full(len(y), np.nan)
    rows = []
    for fold in np.unique(folds):
        train = folds != fold
        params = fit_calibrator(p[train], y[train], method)
        out[~train] = apply_calibrator(p[~train], params, method)
        rows.append(
            {
                "fold": int(fold),
                "parameters": params.tolist(),
                "train_before": float(losses(y[train], p[train]).mean()),
                "train_after": float(
                    losses(y[train], apply_calibrator(p[train], params, method)).mean()
                ),
            }
        )
    return out, rows


def load_predictions():
    records = {}
    sources = {}

    def add(name, path, column="probability", frame=None):
        f = pd.read_csv(path) if frame is None else frame.copy()
        f["uid"] = f["uid"].astype(str)
        if f.uid.duplicated().any():
            raise ValueError(f"Duplicate UID: {name}")
        f = f.set_index("uid").sort_index()
        f["probability"] = f[column]
        if not np.isfinite(f.probability).all() or not f.probability.between(0, 1).all():
            raise ValueError(f"Invalid probabilities: {name}")
        records[name] = f
        sources[name] = {
            "path": str(path.relative_to(ROOT)),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }

    for path in (BASE / "cnn_runs/cnn_compact_v1/final/oof").glob("*/oof_predictions.csv"):
        add("06_" + path.parent.name[:8], path)
    for node, run in [("07", "dat_spect_slab_v4"), ("09", "node09_fine_v1")]:
        for path in (BASE / f"node{node}_runs/{run}/final").glob("oof_*.csv"):
            add(node + "_" + path.stem.removeprefix("oof_")[:8], path)
    for path in (BASE / "node10_runs/node10_hybrid_v1/final").glob("oof_*.csv"):
        add("10_" + path.stem.removeprefix("oof_"), path)
    path = BASE / "node5_runs/optuna_radiomics_v1/optuna_models/optuna_oof_predictions.csv"
    for name in ["logistic", "random_forest", "xgboost", "ensemble"]:
        add("05_" + name, path, "p_" + name + "_oof")
    # Early node 4/5 reference pipelines used acquisition-family held-out CV.
    path = BASE / "full_cohort_v3_1/model_ablation_oof_full.csv"
    old = pd.read_csv(path)
    for name, f in old.groupby("feature_set"):
        add("04_" + name, path, "p_mean_oof", f)
    canonical = records[HGB]
    for name, frame in records.items():
        if not frame.index.equals(canonical.index):
            raise ValueError(f"Cohort mismatch: {name}")
        if not np.array_equal(frame.is_pathologic, canonical.is_pathologic):
            raise ValueError(f"Label mismatch: {name}")
    return records, sources


def bootstrap_gain(y, first, second, groups, draws=2000):
    # Positive = second better. Cluster bootstrap protects within-family dependence.
    d = losses(y, first) - losses(y, second)
    rng = np.random.default_rng(20260905)
    patient = np.array([d[rng.integers(len(d), size=len(d))].mean() for _ in range(draws)])
    unique, ix = np.unique(groups, return_inverse=True)
    totals = np.bincount(ix, weights=d)
    counts = np.bincount(ix)
    sampled = rng.integers(len(unique), size=(draws, len(unique)))
    cluster = totals[sampled].sum(axis=1) / counts[sampled].sum(axis=1)
    return {
        "gain": float(d.mean()),
        "patient_ci_low": float(np.quantile(patient, 0.025)),
        "patient_ci_high": float(np.quantile(patient, 0.975)),
        "family_ci_low": float(np.quantile(cluster, 0.025)),
        "family_ci_high": float(np.quantile(cluster, 0.975)),
    }


def run(out):
    out.mkdir(parents=True, exist_ok=True)
    frames, sources = load_predictions()
    y = frames[HGB].is_pathologic.to_numpy(int)
    groups = frames[CNN].acquisition_family.to_numpy(str)
    node10fold = frames[HGB].fold.to_numpy(int)
    groupfold = frames[CNN].fold.to_numpy(int)
    rows = []
    calibration_rows = []
    for name, f in frames.items():
        p = f.probability.to_numpy(float)
        row = {
            "model": name,
            "n": len(f),
            **metrics(y, p),
            "fold_agreement_node10": float(np.mean(f.fold.to_numpy() == node10fold)),
            "high_confidence_wrong": int(np.sum(((p > 0.9) & (y == 0)) | ((p < 0.1) & (y == 1)))),
            "max_logit_probability_error": float(np.max(np.abs(expit(f.logit) - p)))
            if "logit" in f
            else None,
        }
        if "probability_cross_calibrated" in f:
            row["saved_calibrated_log_loss"] = metrics(y, f.probability_cross_calibrated)[
                "log_loss"
            ]
        for method in ["temperature", "platt", "beta"]:
            calibrated, params = crossfit(p, y, f.fold.to_numpy(int), method)
            row[method + "_crossfit_log_loss"] = metrics(y, calibrated)["log_loss"]
            for x in params:
                calibration_rows.append({"model": name, "method": method, **x})
        rows.append(row)
    inventory = pd.DataFrame(rows).sort_values("log_loss")
    inventory.to_csv(out / "model_inventory.csv", index=False)
    pd.DataFrame(calibration_rows).to_json(
        out / "calibration_parameters.json", orient="records", indent=2
    )
    pmat = pd.DataFrame({k: f.probability for k, f in frames.items()})
    pmat.corr().to_csv(out / "probability_correlations.csv")
    pmat.subtract(y, axis=0).corr().to_csv(out / "residual_correlations.csv")
    pairrows = []
    for a, b in combinations(pmat.columns, 2):
        pa, pb = pmat[a].to_numpy(), pmat[b].to_numpy()
        pairrows.append(
            {
                "first": a,
                "second": b,
                **metrics(y, (pa + pb) / 2),
                "correlation": float(np.corrcoef(pa, pb)[0, 1]),
                "residual_correlation": float(np.corrcoef(pa - y, pb - y)[0, 1]),
                "mean_abs_diff": float(np.abs(pa - pb).mean()),
                "decision_disagreement": float(((pa >= 0.5) != (pb >= 0.5)).mean()),
            }
        )
    pairs = pd.DataFrame(pairrows).sort_values("log_loss")
    pairs.to_csv(out / "all_pairs_exploratory.csv", index=False)
    experiments = {
        "hgb": pmat[HGB].to_numpy(),
        "hgb_cnn_equal": pmat[[HGB, CNN]].mean(axis=1).to_numpy(),
        "hgb_slab_equal": pmat[[HGB, SLAB]].mean(axis=1).to_numpy(),
        "hgb_cnn_slab_equal": pmat[[HGB, CNN, SLAB]].mean(axis=1).to_numpy(),
        "hgb_topology_equal": pmat[[HGB, "10_topology_hgb"]].mean(axis=1).to_numpy(),
    }
    exp_rows, fold_rows, group_rows, bootrows = [], [], [], []
    per_patient = pd.DataFrame(
        {"uid": pmat.index, "is_pathologic": y, "acquisition_family": groups}
    )
    for name, p in experiments.items():
        variants = {"raw": p}
        for splitname, split in [("node10", node10fold), ("group06", groupfold)]:
            for method in ["temperature", "platt", "beta"]:
                variants[splitname + "_" + method] = crossfit(p, y, split, method)[0]
        for method, probability in variants.items():
            key = name + "__" + method
            exp_rows.append({"experiment": key, **metrics(y, probability)})
            per_patient[key] = probability
            for fold in np.unique(groupfold):
                selected = groupfold == fold
                fold_rows.append(
                    {
                        "experiment": key,
                        "fold": int(fold),
                        "n": int(selected.sum()),
                        "log_loss": float(losses(y[selected], probability[selected]).mean()),
                    }
                )
            for group in np.unique(groups):
                selected = groups == group
                group_rows.append(
                    {
                        "experiment": key,
                        "acquisition_family": group,
                        "n": int(selected.sum()),
                        "log_loss": float(losses(y[selected], probability[selected]).mean()),
                    }
                )
            if method in ["raw", "group06_platt"]:
                bootrows.append(
                    {
                        "experiment": key,
                        **bootstrap_gain(y, experiments["hgb"], probability, groups),
                    }
                )
    pd.DataFrame(exp_rows).sort_values("log_loss").to_csv(
        out / "ensemble_calibration_experiments.csv", index=False
    )
    pd.DataFrame(fold_rows).to_csv(out / "ensemble_by_groupfold.csv", index=False)
    pd.DataFrame(group_rows).to_csv(out / "ensemble_by_family.csv", index=False)
    pd.DataFrame(bootrows).to_csv(out / "paired_bootstrap.csv", index=False)
    per_patient.to_csv(out / "exploratory_predictions_private.csv", index=False)
    errorrows = []
    for name, p in experiments.items():
        ll = losses(y, p)
        for frac in [0.01, 0.05, 0.10, 0.20]:
            n = int(np.ceil(len(y) * frac))
            errorrows.append(
                {
                    "experiment": name,
                    "worst_fraction": frac,
                    "loss_share": float(np.sort(ll)[-n:].sum() / ll.sum()),
                }
            )
    pd.DataFrame(errorrows).to_csv(out / "error_concentration.csv", index=False)
    disagreement = (pmat[HGB].to_numpy() >= 0.5) != (pmat[CNN].to_numpy() >= 0.5)
    disagreement_rows = []
    for value in [False, True]:
        s = disagreement == value
        row = {"disagree": value, "n": int(s.sum()), "prevalence": float(y[s].mean())}
        for name, p in experiments.items():
            row[name + "_loss"] = float(losses(y[s], p[s]).mean())
        disagreement_rows.append(row)
    pd.DataFrame(disagreement_rows).to_csv(out / "disagreement_groups.csv", index=False)
    hardest = per_patient[["uid", "is_pathologic", "acquisition_family"]].copy()
    hardest["hgb_loss"] = losses(y, experiments["hgb"])
    hardest["blend_loss"] = losses(y, experiments["hgb_cnn_equal"])
    hardest["hgb_probability"] = experiments["hgb"]
    hardest["cnn_probability"] = pmat[CNN].to_numpy()
    hardest["both_confident_wrong"] = (
        (pmat[HGB].to_numpy() > 0.8) & (pmat[CNN].to_numpy() > 0.8) & (y == 0)
    ) | ((pmat[HGB].to_numpy() < 0.2) & (pmat[CNN].to_numpy() < 0.2) & (y == 1))
    hardest.sort_values("blend_loss", ascending=False).to_csv(
        out / "case_review_private.csv", index=False
    )
    summary = {
        "models": len(frames),
        "n": len(y),
        "prevalence": float(y.mean()),
        "constant_prior_loss": float(log_loss(y, np.full(len(y), y.mean()))),
        "node11_final_available": (
            BASE / "node11_runs/node11_best_shot_v1/final/final_metrics.csv"
        ).exists(),
        "best_individual": inventory.iloc[0].to_dict(),
        "both_confident_wrong": int(hardest.both_confident_wrong.sum()),
        "scope": "exploratory reuse of historical OOF; no raw files, model weights or original outputs changed",
        "selection_warning": "all pairs inspected post hoc; cross-fitted calibration does not repair base-level selection dependence",
        "sources": sources,
    }
    (out / "audit_manifest.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(
        inventory[
            [
                "model",
                "log_loss",
                "saved_calibrated_log_loss",
                "temperature_crossfit_log_loss",
                "platt_crossfit_log_loss",
            ]
        ].to_string(index=False)
    )
    print("\nBest exploratory pairs:\n" + pairs.head(10).to_string(index=False))
    print(
        "\nPredefined recombinations:\n"
        + pd.DataFrame(exp_rows).sort_values("log_loss").head(15).to_string(index=False)
    )
    print("\nBootstrap:\n" + pd.DataFrame(bootrows).to_string(index=False))
    print("\nDisagreement:\n" + pd.DataFrame(disagreement_rows).to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=BASE / "project_audit_20260905")
    args = parser.parse_args()
    with threadpool_limits(limits=4):
        run(args.output)
