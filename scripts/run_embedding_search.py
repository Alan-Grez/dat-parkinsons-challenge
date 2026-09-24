"""Tune and compare t-SNE/UMAP from durable notebook-05 PCA coordinates."""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(PROJECT_ROOT))

from modeling import run_embedding_search

PRIVATE_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "private_eda"


def _pc_columns(frame: pd.DataFrame) -> list[str]:
    columns = [
        column
        for column in frame.columns
        if re.fullmatch(r"PC[1-9][0-9]*", str(column))
    ]
    return sorted(columns, key=lambda column: int(column[2:]))


def _plot_comparison(coordinates: pd.DataFrame, qc: pd.DataFrame, destination: Path) -> None:
    plot_frame = coordinates.merge(
        qc[["uid", "technical_outlier_score"]],
        on="uid",
        how="left",
        validate="one_to_one",
    )
    figure, axes = plt.subplots(2, 3, figsize=(18, 10))
    for row, method in enumerate(("tsne", "umap")):
        x_column = f"{method}_balanced_1"
        y_column = f"{method}_balanced_2"
        sns.scatterplot(
            data=plot_frame,
            x=x_column,
            y=y_column,
            hue="is_pathologic",
            alpha=0.72,
            s=25,
            ax=axes[row, 0],
        )
        axes[row, 0].set_title(f"{method.upper()} optimizado · etiqueta")
        sns.scatterplot(
            data=plot_frame,
            x=x_column,
            y=y_column,
            hue="spacing_x_mm",
            palette="viridis",
            alpha=0.72,
            s=25,
            ax=axes[row, 1],
        )
        axes[row, 1].set_title(f"{method.upper()} optimizado · spacing")
        sns.scatterplot(
            data=plot_frame,
            x=x_column,
            y=y_column,
            hue="technical_outlier_score",
            palette="magma",
            alpha=0.72,
            s=25,
            ax=axes[row, 2],
        )
        axes[row, 2].set_title(f"{method.upper()} optimizado · QC técnico")
    figure.tight_layout()
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=160, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--node5-run-id", default="optuna_radiomics_v1")
    parser.add_argument("--embedding-run-id", default="umap_tsne_v1")
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--evaluation-rows", type=int, default=800)
    parser.add_argument("--cpu-jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = parser.parse_args()
    for value, option in [
        (args.node5_run_id, "--node5-run-id"),
        (args.embedding_run_id, "--embedding-run-id"),
    ]:
        if not re.fullmatch(r"[A-Za-z0-9._-]+", value):
            raise ValueError(f"{option} only accepts letters, numbers, dot, _ and -.")
    if args.trials < 1 or args.seeds < 2:
        raise ValueError("--trials must be >= 1 and --seeds must be >= 2.")

    node5_dir = PRIVATE_OUTPUT_DIR / "node5_runs" / args.node5_run_id
    source_path = node5_dir / "embedding_coordinates_full.csv"
    qc_path = PRIVATE_OUTPUT_DIR / "image_qc_manifest.csv"
    if not source_path.exists():
        raise FileNotFoundError(source_path)
    if not qc_path.exists():
        raise FileNotFoundError(qc_path)
    source = pd.read_csv(source_path, dtype={"uid": "string"})
    qc = pd.read_csv(qc_path, dtype={"uid": "string"})
    pc_columns = _pc_columns(source)
    if len(pc_columns) < 2:
        raise RuntimeError(f"No PCA columns found in {source_path}.")
    merged = source.merge(
        qc[["uid", "spacing_x_mm", "technical_outlier_score"]],
        on="uid",
        how="left",
        validate="one_to_one",
    )
    if merged[["spacing_x_mm", "technical_outlier_score"]].isna().any().any():
        raise RuntimeError("QC merge produced missing spacing/outlier values.")
    baseline = (
        merged[["NL1", "NL2"]].to_numpy(dtype=float)
        if {"NL1", "NL2"} <= set(merged.columns)
        else None
    )
    output_dir = node5_dir / "embedding_runs" / args.embedding_run_id
    seed_values = tuple(20260821 + index for index in range(args.seeds))
    result = run_embedding_search(
        merged[pc_columns].to_numpy(dtype=float),
        y=merged["is_pathologic"].to_numpy(dtype=int),
        groups=merged["acquisition_family"].astype(str).to_numpy(),
        spacing=merged["spacing_x_mm"].to_numpy(dtype=float),
        uids=merged["uid"].astype(str).to_numpy(),
        output_dir=output_dir,
        experiment_config={
            "node5_run_id": args.node5_run_id,
            "embedding_run_id": args.embedding_run_id,
            "source": str(source_path),
            "pca_columns": pc_columns,
        },
        n_trials_per_method=args.trials,
        seeds=seed_values,
        cpu_jobs=args.cpu_jobs,
        evaluation_rows=args.evaluation_rows,
        baseline_coordinates=baseline,
    )
    figure_path = output_dir / "embedding_comparison.png"
    _plot_comparison(result.coordinates, qc, figure_path)
    columns = [
        "method",
        "selection",
        "label_silhouette",
        "label_knn_balanced_accuracy",
        "neighborhood_trustworthiness",
        "seed_neighborhood_stability",
        "acquisition_family_confound",
        "spacing_confound",
        "balanced_score",
    ]
    print(result.comparison[columns].to_string(index=False))
    print(f"\nCoordinates: {output_dir / 'embedding_search_coordinates.csv'}")
    print(f"Figure: {figure_path}")


if __name__ == "__main__":
    main()
