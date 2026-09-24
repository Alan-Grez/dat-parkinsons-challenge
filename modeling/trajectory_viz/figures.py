from __future__ import annotations

import html
import json
import os
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

from modeling.cnn.calibration import binary_metrics

from .core import TrajectorySnapshot

PLOTLY_CONFIG = {
    "displaylogo": False,
    "responsive": True,
    "toImageButtonOptions": {"format": "png", "scale": 2},
}


def _empty_figure(title: str, message: str) -> go.Figure:
    figure = go.Figure()
    figure.add_annotation(
        text=message,
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        showarrow=False,
        font={"size": 15},
    )
    figure.update_layout(title=title, template="plotly_white", height=360)
    return figure


def _base_layout(figure: go.Figure, *, title: str, height: int = 520) -> go.Figure:
    figure.update_layout(
        title=title,
        template="plotly_white",
        height=height,
        hovermode="closest",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
        margin={"l": 70, "r": 30, "t": 95, "b": 70},
    )
    return figure


def optimization_figure(trials: pd.DataFrame) -> go.Figure:
    if trials.empty:
        return _empty_figure(
            "Trayectoria de Optuna",
            "Todavía no hay trials persistidos en los runs seleccionados.",
        )
    complete = trials.loc[trials["completed"]].copy()
    if complete.empty:
        return _empty_figure(
            "Trayectoria de Optuna",
            "La base existe, pero todavía no hay trials completos con log loss.",
        )
    run_keys = complete["run_key"].drop_duplicates().tolist()
    figure = make_subplots(
        rows=len(run_keys),
        cols=1,
        shared_xaxes=False,
        subplot_titles=run_keys,
        vertical_spacing=min(0.16, 0.24 / max(len(run_keys), 1)),
    )
    palette = px.colors.qualitative.Safe + px.colors.qualitative.Bold
    studies = complete["study_label"].drop_duplicates().tolist()
    color_map = {study: palette[index % len(palette)] for index, study in enumerate(studies)}
    for row_index, run_key in enumerate(run_keys, start=1):
        subset = complete.loc[complete["run_key"].eq(run_key)]
        for study_label, group in subset.groupby("study_label", sort=False):
            group = group.sort_values("trial_number")
            custom = np.column_stack(
                [
                    group["state"].astype(str),
                    group["architecture"].astype(str),
                    group["feature_variant"].astype(str),
                    group["duration_seconds"].fillna(np.nan),
                ]
            )
            color = color_map[study_label]
            figure.add_trace(
                go.Scatter(
                    x=group["trial_number"],
                    y=group["objective_log_loss"],
                    mode="lines+markers",
                    name=study_label,
                    legendgroup=study_label,
                    showlegend=row_index == 1,
                    line={"color": color, "width": 1.3},
                    marker={"color": color, "size": 7, "symbol": "circle"},
                    customdata=custom,
                    hovertemplate=(
                        "trial=%{x}<br>log loss=%{y:.4f}<br>estado=%{customdata[0]}"
                        "<br>arquitectura=%{customdata[1]}<br>variante=%{customdata[2]}"
                        "<br>duración=%{customdata[3]:.0f} s<extra></extra>"
                    ),
                ),
                row=row_index,
                col=1,
            )
            figure.add_trace(
                go.Scatter(
                    x=group["trial_number"],
                    y=group["best_so_far"],
                    mode="lines",
                    name=f"{study_label} · mejor acumulado",
                    legendgroup=study_label,
                    showlegend=False,
                    line={"color": color, "width": 2.7, "dash": "dash"},
                    hovertemplate="trial=%{x}<br>mejor acumulado=%{y:.4f}<extra></extra>",
                ),
                row=row_index,
                col=1,
            )
        figure.update_yaxes(title_text="log loss CV3", row=row_index, col=1)
        figure.update_xaxes(title_text="número de trial", row=row_index, col=1)
    return _base_layout(
        figure,
        title="Optuna: resultado de cada trial y mejor valor acumulado",
        height=max(470, 360 * len(run_keys)),
    )


def network_comparison_figure(trials: pd.DataFrame) -> go.Figure:
    if trials.empty or "completed" not in trials:
        return _empty_figure("Comparación entre redes", "No hay trials para comparar.")
    complete = trials.loc[trials["completed"]].copy()
    if complete.empty:
        return _empty_figure("Comparación entre redes", "No hay trials completos para comparar.")
    order = (
        complete.groupby("network")["objective_log_loss"]
        .min()
        .sort_values()
        .index.tolist()
    )
    figure = px.box(
        complete,
        x="network",
        y="objective_log_loss",
        color="node",
        points="all",
        category_orders={"network": order},
        hover_data=["study_name", "trial_number", "state", "duration_seconds"],
        labels={
            "network": "red / rama",
            "objective_log_loss": "log loss CV3",
            "node": "nodo",
        },
    )
    figure.update_xaxes(tickangle=-22)
    return _base_layout(
        figure,
        title="Distribución del rendimiento por arquitectura y fuente de features",
        height=560,
    )


def efficiency_figure(trials: pd.DataFrame) -> go.Figure:
    if trials.empty or "completed" not in trials:
        return _empty_figure("Costo frente a rendimiento", "No hay trials para comparar.")
    usable = trials.loc[
        trials["completed"]
        & trials["duration_seconds"].notna()
        & trials["duration_seconds"].gt(0)
    ].copy()
    if usable.empty:
        return _empty_figure(
            "Costo frente a rendimiento",
            "Los trials completos todavía no tienen duración registrada.",
        )
    usable["duration_minutes"] = usable["duration_seconds"] / 60.0
    figure = px.scatter(
        usable,
        x="duration_minutes",
        y="objective_log_loss",
        color="network",
        symbol="node",
        hover_data=["study_name", "trial_number", "state", "fixed_epochs"],
        labels={
            "duration_minutes": "duración total del trial (min)",
            "objective_log_loss": "log loss CV3",
            "network": "red / rama",
            "node": "nodo",
        },
    )
    return _base_layout(
        figure,
        title="Costo de cómputo frente a rendimiento fuera de muestra",
        height=540,
    )


def best_fold_comparison_figure(trials: pd.DataFrame, fold_scores: pd.DataFrame) -> go.Figure:
    if trials.empty or fold_scores.empty:
        return _empty_figure(
            "Mejor trial por red y fold",
            "Los folds se mostrarán cuando haya candidatos completos con fold_log_loss.",
        )
    complete = trials.loc[trials["completed"]].copy()
    if complete.empty:
        return _empty_figure("Mejor trial por red y fold", "No hay trials completos.")
    best_index = complete.groupby(["run_key", "study_name"])["objective_log_loss"].idxmin()
    best = complete.loc[best_index, ["run_key", "study_name", "trial_number", "network"]]
    selected = fold_scores.merge(
        best,
        on=["run_key", "study_name", "trial_number"],
        how="inner",
        suffixes=("", "_best"),
    )
    if selected.empty:
        return _empty_figure(
            "Mejor trial por red y fold",
            "Los mejores trials actuales aún no tienen scores de sus tres folds.",
        )
    order = (
        selected.groupby("network")["fold_log_loss"].mean().sort_values().index.tolist()
    )
    figure = px.strip(
        selected,
        x="network",
        y="fold_log_loss",
        color="fold",
        category_orders={"network": order},
        hover_data=["study_name", "trial_number", "run_id"],
        labels={"network": "mejor trial de cada red", "fold_log_loss": "log loss del fold"},
    )
    means = selected.groupby("network", as_index=False)["fold_log_loss"].agg(["mean", "std"]).reset_index()
    figure.add_trace(
        go.Scatter(
            x=means["network"],
            y=means["mean"],
            error_y={"type": "data", "array": means["std"].fillna(0), "visible": True},
            mode="markers",
            name="media ± DE entre folds",
            marker={"size": 12, "symbol": "diamond"},
            hovertemplate="%{x}<br>media=%{y:.4f}<extra></extra>",
        )
    )
    figure.update_xaxes(tickangle=-22)
    return _base_layout(
        figure,
        title="Robustez entre folds del mejor trial de cada red",
        height=560,
    )


def _candidate_display(frame: pd.DataFrame) -> pd.Series:
    return (
        frame["architecture"].astype(str)
        + " · "
        + frame["candidate_id"].astype(str).str.slice(0, 8)
    )


def final_progress_figure(progress: pd.DataFrame) -> go.Figure:
    if progress.empty:
        return _empty_figure(
            "Evaluación final CV5",
            "La evaluación final todavía no ha creado folds persistidos.",
        )
    frame = progress.copy()
    frame["candidate_display"] = _candidate_display(frame)
    candidate_order = (
        frame[["candidate_display", "search_log_loss"]]
        .drop_duplicates()
        .sort_values("search_log_loss", na_position="last")["candidate_display"]
        .tolist()
    )
    folds = sorted(frame["fold"].dropna().astype(int).unique().tolist())
    z: list[list[float | None]] = []
    text: list[list[str]] = []
    hover: list[list[str]] = []
    for candidate in candidate_order:
        row_values: list[float | None] = []
        row_text: list[str] = []
        row_hover: list[str] = []
        subset = frame.loc[frame["candidate_display"].eq(candidate)].set_index("fold")
        for fold in folds:
            if fold not in subset.index:
                row_values.append(None)
                row_text.append("pendiente")
                row_hover.append("sin artefactos")
                continue
            row = subset.loc[fold]
            score = pd.to_numeric(row.get("fold_log_loss"), errors="coerce")
            status = str(row.get("status", "pending"))
            epochs = int(row.get("epochs_recorded", 0))
            planned = int(row.get("planned_epochs", 0))
            row_values.append(float(score) if pd.notna(score) else None)
            row_text.append(
                f"{float(score):.3f}"
                if pd.notna(score)
                else (f"{epochs}/{planned}" if status == "training" else "pendiente")
            )
            row_hover.append(
                f"estado={status}<br>épocas={epochs}/{planned}"
                + (f"<br>log loss={float(score):.4f}" if pd.notna(score) else "")
            )
        z.append(row_values)
        text.append(row_text)
        hover.append(row_hover)
    finite = np.asarray(
        [value for row in z for value in row if value is not None], dtype=float
    )
    zmin = float(finite.min()) if finite.size else 0.45
    zmax = float(finite.max()) if finite.size else 0.75
    if zmin == zmax:
        zmin, zmax = zmin - 0.02, zmax + 0.02
    figure = go.Figure(
        go.Heatmap(
            z=z,
            x=[f"fold {fold}" for fold in folds],
            y=candidate_order,
            customdata=hover,
            hovertemplate="%{y}<br>%{x}<br>%{customdata}<extra></extra>",
            colorscale="Viridis_r",
            zmin=zmin,
            zmax=zmax,
            colorbar={"title": "log loss"},
            xgap=3,
            ygap=3,
        )
    )
    figure.add_trace(
        go.Scatter(
            x=[f"fold {fold}" for _ in candidate_order for fold in folds],
            y=[candidate for candidate in candidate_order for _ in folds],
            text=[value for row in text for value in row],
            mode="text",
            textfont={"size": 12},
            hoverinfo="skip",
            showlegend=False,
        )
    )
    figure.update_xaxes(title="fold final congelado")
    figure.update_yaxes(title="finalista")
    return _base_layout(
        figure,
        title="Avance y log loss de la evaluación final CV5",
        height=max(420, 120 + 80 * len(candidate_order)),
    )


def final_fold_figure(progress: pd.DataFrame) -> go.Figure:
    if progress.empty:
        return _empty_figure(
            "Variación entre folds finales", "Todavía no hay folds finales completos."
        )
    complete = progress.loc[progress["fold_log_loss"].notna()].copy()
    if complete.empty:
        return _empty_figure(
            "Variación entre folds finales", "Todavía no hay log loss final por fold."
        )
    complete["candidate_display"] = _candidate_display(complete)
    order = (
        complete.groupby("candidate_display")["fold_log_loss"]
        .mean()
        .sort_values()
        .index.tolist()
    )
    figure = px.strip(
        complete,
        x="candidate_display",
        y="fold_log_loss",
        color="fold",
        category_orders={"candidate_display": order},
        hover_data=[
            "candidate_id",
            "architecture",
            "feature_variant",
            "lateral_strategy",
            "n_predictions",
        ],
        labels={
            "candidate_display": "finalista",
            "fold_log_loss": "log loss OOF del fold",
            "fold": "fold",
        },
    )
    means = (
        complete.groupby("candidate_display", as_index=False)["fold_log_loss"]
        .agg(["mean", "std"])
        .reset_index()
    )
    figure.add_trace(
        go.Scatter(
            x=means["candidate_display"],
            y=means["mean"],
            error_y={
                "type": "data",
                "array": means["std"].fillna(0),
                "visible": True,
            },
            mode="markers",
            name="media ± DE",
            marker={"size": 13, "symbol": "diamond"},
            hovertemplate="%{x}<br>media=%{y:.4f}<extra></extra>",
        )
    )
    return _base_layout(
        figure,
        title="Estabilidad entre los cinco folds finales",
        height=520,
    )


def final_metrics_figure(metrics: pd.DataFrame) -> go.Figure:
    if metrics.empty:
        return _empty_figure(
            "Métricas OOF finales",
            "Las métricas aparecerán a medida que terminen los candidatos.",
        )
    frame = metrics.copy()
    frame["candidate_display"] = _candidate_display(frame)
    specifications = [
        ("log_loss", "Log loss ↓"),
        ("auc", "AUROC ↑"),
        ("brier", "Brier ↓"),
        ("ece_10", "ECE-10 ↓"),
    ]
    figure = make_subplots(
        rows=1,
        cols=len(specifications),
        subplot_titles=[title for _, title in specifications],
    )
    methods = [
        ("raw", "OOF sin calibrar"),
        ("calibrated", "OOF cross-calibrado"),
    ]
    colors = px.colors.qualitative.Safe
    for column_index, (metric, _) in enumerate(specifications, start=1):
        for method_index, (prefix, label) in enumerate(methods):
            column = f"{prefix}_{metric}"
            if column not in frame or frame[column].notna().sum() == 0:
                continue
            usable = frame.loc[frame[column].notna()].copy()
            figure.add_trace(
                go.Bar(
                    x=usable["candidate_display"],
                    y=usable[column],
                    name=label,
                    legendgroup=prefix,
                    showlegend=column_index == 1,
                    marker_color=colors[method_index],
                    customdata=np.column_stack(
                        [
                            usable["folds_completed"],
                            usable["expected_folds"],
                            usable["metrics_source"].astype(str),
                        ]
                    ),
                    hovertemplate=(
                        "%{x}<br>valor=%{y:.4f}<br>folds=%{customdata[0]}/%{customdata[1]}"
                        "<br>fuente=%{customdata[2]}<extra></extra>"
                    ),
                ),
                row=1,
                col=column_index,
            )
    figure.update_layout(barmode="group")
    figure.update_xaxes(tickangle=-24)
    return _base_layout(
        figure,
        title="Rendimiento OOF de los finalistas (parcial o final)",
        height=560,
    )


def _select_candidate_predictions(
    predictions: pd.DataFrame,
    candidate_id: str | None,
) -> tuple[pd.DataFrame, str | None]:
    if predictions.empty:
        return pd.DataFrame(), None
    available = predictions["candidate_id"].astype(str)
    if candidate_id is None or str(candidate_id) not in set(available):
        ranking = (
            predictions.assign(candidate_id=available)
            .groupby("candidate_id")
            .agg(folds=("fold", "nunique"), rows=("uid", "size"))
            .sort_values(["folds", "rows"], ascending=False)
        )
        candidate_id = str(ranking.index[0])
    selected = predictions.loc[available.eq(str(candidate_id))].copy()
    return selected, str(candidate_id)


def probability_distribution_figure(
    predictions: pd.DataFrame,
    candidate_id: str | None = None,
) -> go.Figure:
    selected, candidate_id = _select_candidate_predictions(predictions, candidate_id)
    if selected.empty:
        return _empty_figure(
            "Distribución de probabilidades OOF", "Todavía no hay predicciones finales."
        )
    probability_columns = [("probability", "sin calibrar")]
    if (
        "probability_cross_calibrated" in selected
        and selected["probability_cross_calibrated"].notna().all()
    ):
        probability_columns.append(
            ("probability_cross_calibrated", "cross-calibrada")
        )
    figure = make_subplots(
        rows=1,
        cols=len(probability_columns),
        subplot_titles=[label for _, label in probability_columns],
        shared_yaxes=True,
    )
    colors = px.colors.qualitative.Safe
    for column_index, (probability_column, _) in enumerate(
        probability_columns, start=1
    ):
        for class_value, class_label in ((0, "control"), (1, "patológico")):
            values = selected.loc[
                selected["is_pathologic"].eq(class_value), probability_column
            ]
            figure.add_trace(
                go.Histogram(
                    x=values,
                    nbinsx=24,
                    histnorm="probability density",
                    name=class_label,
                    legendgroup=str(class_value),
                    showlegend=column_index == 1,
                    opacity=0.62,
                    marker_color=colors[class_value],
                    hovertemplate="P=%{x:.3f}<br>densidad=%{y:.3f}<extra></extra>",
                ),
                row=1,
                col=column_index,
            )
        figure.update_xaxes(title_text="P(is_pathologic)", row=1, col=column_index)
    figure.update_layout(barmode="overlay")
    figure.update_yaxes(title_text="densidad", row=1, col=1)
    label = str(selected.iloc[0].get("candidate_label", candidate_id))
    return _base_layout(
        figure,
        title=f"Separación OOF por etiqueta · {label}",
        height=500,
    )


def _reliability_points(
    frame: pd.DataFrame,
    probability_column: str,
    bins: int,
) -> pd.DataFrame:
    usable = frame.loc[frame[probability_column].notna()].copy()
    if usable.empty:
        return pd.DataFrame()
    unique = int(usable[probability_column].nunique())
    quantiles = min(max(unique, 1), bins)
    usable["calibration_bin"] = pd.qcut(
        usable[probability_column], q=quantiles, duplicates="drop"
    )
    return (
        usable.groupby("calibration_bin", observed=True)
        .agg(
            mean_probability=(probability_column, "mean"),
            observed_fraction=("is_pathologic", "mean"),
            n=("uid", "size"),
        )
        .reset_index(drop=True)
    )


def calibration_figure(
    predictions: pd.DataFrame,
    candidate_id: str | None = None,
    *,
    bins: int = 10,
) -> go.Figure:
    selected, candidate_id = _select_candidate_predictions(predictions, candidate_id)
    if selected.empty:
        return _empty_figure(
            "Calibración OOF", "Todavía no hay predicciones finales para calibrar."
        )
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="calibración ideal",
            line={"dash": "dash", "color": "gray"},
            hoverinfo="skip",
        )
    )
    columns = [("probability", "sin calibrar")]
    if (
        "probability_cross_calibrated" in selected
        and selected["probability_cross_calibrated"].notna().all()
    ):
        columns.append(("probability_cross_calibrated", "cross-calibrada"))
    for probability_column, label in columns:
        points = _reliability_points(selected, probability_column, bins)
        figure.add_trace(
            go.Scatter(
                x=points["mean_probability"],
                y=points["observed_fraction"],
                mode="lines+markers",
                name=label,
                marker={"size": np.clip(np.sqrt(points["n"]) * 1.5, 7, 18)},
                customdata=points["n"],
                hovertemplate=(
                    "probabilidad media=%{x:.3f}<br>frecuencia observada=%{y:.3f}"
                    "<br>n=%{customdata}<extra></extra>"
                ),
            )
        )
    figure.update_xaxes(title="probabilidad predicha", range=[0, 1])
    figure.update_yaxes(title="fracción patológica observada", range=[0, 1])
    label = str(selected.iloc[0].get("candidate_label", candidate_id))
    return _base_layout(
        figure,
        title=f"Curva de calibración OOF · {label}",
        height=520,
    )


def finalist_agreement_figure(predictions: pd.DataFrame) -> go.Figure:
    if predictions.empty or predictions["candidate_id"].nunique() < 2:
        return _empty_figure(
            "Acuerdo entre finalistas",
            "Se necesitan predicciones de al menos dos finalistas.",
        )
    labels = (
        predictions[["candidate_id", "architecture"]]
        .drop_duplicates()
        .assign(display=lambda frame: _candidate_display(frame))
        .set_index("candidate_id")["display"]
        .to_dict()
    )
    pivot = predictions.pivot_table(
        index="uid", columns="candidate_id", values="probability", aggfunc="first"
    )
    columns = [column for column in pivot.columns if pivot[column].notna().sum()]
    if len(columns) < 2:
        return _empty_figure(
            "Acuerdo entre finalistas", "Los finalistas todavía no comparten casos OOF."
        )
    difference = np.full((len(columns), len(columns)), np.nan, dtype=float)
    correlation = np.full_like(difference, np.nan)
    support = np.zeros_like(difference, dtype=int)
    for row_index, first in enumerate(columns):
        for column_index, second in enumerate(columns):
            paired = pd.concat(
                [
                    pivot[first].rename("first_probability"),
                    pivot[second].rename("second_probability"),
                ],
                axis=1,
            ).dropna()
            if paired.empty:
                continue
            support[row_index, column_index] = len(paired)
            difference[row_index, column_index] = float(
                np.mean(
                    np.abs(
                        paired["first_probability"]
                        - paired["second_probability"]
                    )
                )
            )
            correlation[row_index, column_index] = (
                float(
                    paired["first_probability"].corr(
                        paired["second_probability"]
                    )
                )
                if len(paired) > 1
                else np.nan
            )
    display = [labels.get(str(column), str(column)[:8]) for column in columns]
    figure = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=["diferencia absoluta media", "correlación de probabilidades"],
    )
    figure.add_trace(
        go.Heatmap(
            z=difference,
            x=display,
            y=display,
            colorscale="Viridis",
            zmin=0,
            zmax=max(float(np.nanmax(difference)), 0.01),
            text=np.round(difference, 3),
            texttemplate="%{text}",
            customdata=support,
            colorbar={"title": "|ΔP|", "x": 0.44},
            hovertemplate=(
                "%{y}<br>%{x}<br>|ΔP|=%{z:.3f}<br>casos compartidos=%{customdata}"
                "<extra></extra>"
            ),
        ),
        row=1,
        col=1,
    )
    figure.add_trace(
        go.Heatmap(
            z=correlation,
            x=display,
            y=display,
            colorscale="Viridis",
            zmin=0,
            zmax=1,
            text=np.round(correlation, 3),
            texttemplate="%{text}",
            customdata=support,
            colorbar={"title": "r"},
            hovertemplate=(
                "%{y}<br>%{x}<br>r=%{z:.3f}<br>casos compartidos=%{customdata}"
                "<extra></extra>"
            ),
        ),
        row=1,
        col=2,
    )
    figure.update_xaxes(tickangle=-24)
    return _base_layout(
        figure,
        title="Complementariedad potencial entre los finalistas",
        height=560,
    )


def acquisition_family_figure(
    predictions: pd.DataFrame,
    *,
    minimum_cases: int = 10,
) -> go.Figure:
    if predictions.empty or "acquisition_family" not in predictions:
        return _empty_figure(
            "Robustez por familia de adquisición",
            "Las predicciones todavía no incluyen familias de adquisición.",
        )
    records: list[dict[str, object]] = []
    for (candidate_id, family), group in predictions.groupby(
        ["candidate_id", "acquisition_family"], dropna=False
    ):
        if len(group) < minimum_cases:
            continue
        metrics = binary_metrics(group, "probability")
        records.append(
            {
                "candidate_id": candidate_id,
                "candidate_display": _candidate_display(group.iloc[[0]]).iloc[0],
                "acquisition_family": str(family),
                "n": len(group),
                "family_log_loss": metrics["log_loss"],
                "family_auc": metrics["auc"],
            }
        )
    frame = pd.DataFrame.from_records(records)
    if frame.empty:
        return _empty_figure(
            "Robustez por familia de adquisición",
            f"No hay familias con al menos {minimum_cases} casos.",
        )
    figure = px.box(
        frame,
        x="candidate_display",
        y="family_log_loss",
        color="candidate_display",
        points="all",
        hover_data=["acquisition_family", "n", "family_auc"],
        labels={
            "candidate_display": "finalista",
            "family_log_loss": "log loss por familia",
        },
    )
    figure.update_layout(showlegend=False)
    return _base_layout(
        figure,
        title=f"Variabilidad entre familias de adquisición (n ≥ {minimum_cases})",
        height=540,
    )


def _encoded_parallel_dimension(series: pd.Series, label: str) -> dict:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().all():
        minimum, maximum = float(numeric.min()), float(numeric.max())
        if minimum == maximum:
            minimum -= 0.5
            maximum += 0.5
        return {"label": label, "values": numeric, "range": [minimum, maximum]}
    categories = sorted(series.astype(str).fillna("NA").unique().tolist())
    mapping = {value: index for index, value in enumerate(categories)}
    encoded = series.astype(str).fillna("NA").map(mapping)
    return {
        "label": label,
        "values": encoded,
        "range": [-0.2, max(len(categories) - 0.8, 0.8)],
        "tickvals": list(mapping.values()),
        "ticktext": categories,
    }


def hyperparameter_figure(
    trials: pd.DataFrame,
    study_name: str | None = None,
    *,
    run_key: str | None = None,
) -> go.Figure:
    if trials.empty:
        return _empty_figure("Espacio de hiperparámetros", "No hay trials disponibles.")
    complete = trials.loc[trials["completed"]].copy()
    if run_key is not None:
        complete = complete.loc[complete["run_key"].eq(run_key)].copy()
    if complete.empty:
        return _empty_figure("Espacio de hiperparámetros", "No hay trials completos.")
    if study_name is None or study_name not in set(complete["study_name"]):
        study_name = str(
            complete.groupby("study_name")["objective_log_loss"].min().idxmin()
        )
    subset = complete.loc[complete["study_name"].eq(study_name)].copy()
    parameter_columns = [
        column
        for column in subset.columns
        if column.startswith("param__") and subset[column].nunique(dropna=False) > 1
    ]
    if not parameter_columns:
        return _empty_figure(
            f"Espacio de hiperparámetros · {study_name}",
            "Este estudio todavía no tiene suficientes variaciones de parámetros.",
        )
    dimensions = [
        _encoded_parallel_dimension(subset[column], column.removeprefix("param__"))
        for column in parameter_columns[:11]
    ]
    objective = subset["objective_log_loss"].to_numpy(dtype=float)
    dimensions.append(_encoded_parallel_dimension(pd.Series(objective), "log loss"))
    figure = go.Figure(
        data=go.Parcoords(
            line={
                "color": objective,
                "colorscale": "Viridis_r",
                "showscale": True,
                "colorbar": {"title": "log loss"},
                "cmin": float(np.nanmin(objective)),
                "cmax": float(np.nanmax(objective)),
            },
            dimensions=dimensions,
            labelfont={"size": 12},
            tickfont={"size": 10},
        )
    )
    return _base_layout(
        figure,
        title=(
            f"Espacio de hiperparámetros · {run_key + ' · ' if run_key else ''}"
            f"{study_name} (menor log loss es mejor)"
        ),
        height=590,
    )


def _trajectory_label(row: pd.Series) -> str:
    trial = row.get("trial_number")
    trial_text = f"trial {int(trial)}" if pd.notna(trial) else f"hash {row['trajectory_id'][:8]}"
    return f"{row['run_key']} · {row['study_name']} · {trial_text}"


def learning_curve_figure(
    histories: pd.DataFrame,
    trajectory_id: str | None = None,
    *,
    run_key: str | None = None,
    phase: str | None = None,
) -> go.Figure:
    if histories.empty:
        return _empty_figure(
            "Curvas de aprendizaje",
            "Todavía no hay history.csv persistidos en los runs seleccionados.",
        )
    candidates = histories.copy()
    if run_key is not None:
        candidates = candidates.loc[candidates["run_key"].eq(run_key)].copy()
    if phase is not None:
        candidates = candidates.loc[candidates["phase"].eq(phase)].copy()
    elif trajectory_id is None:
        search_candidates = candidates.loc[candidates["phase"].eq("search_cv3")]
        if not search_candidates.empty:
            candidates = search_candidates.copy()
    if candidates.empty:
        return _empty_figure(
            "Curvas de aprendizaje",
            "No hay historiales para el run o la fase seleccionados.",
        )
    if trajectory_id is None or trajectory_id not in set(candidates["trajectory_id"]):
        ranked = (
            candidates.groupby("trajectory_id", dropna=False)["objective_log_loss"]
            .first()
            .sort_values(na_position="last")
        )
        trajectory_id = str(ranked.index[0])
    selected = candidates.loc[candidates["trajectory_id"].astype(str).eq(str(trajectory_id))].copy()
    if selected.empty:
        return _empty_figure("Curvas de aprendizaje", "La trayectoria seleccionada no existe.")
    first = selected.iloc[0]
    figure = go.Figure()
    palette = px.colors.qualitative.Safe
    for fold, group in selected.groupby("fold", sort=True):
        ordered = group.sort_values("epoch_number")
        color = palette[int(fold) % len(palette)]
        figure.add_trace(
            go.Scatter(
                x=ordered["epoch_number"],
                y=ordered["train_loss"],
                mode="lines+markers",
                name=f"fold {fold} · objetivo train",
                legendgroup=f"fold-{fold}",
                line={"color": color, "dash": "dot", "width": 1.5},
                marker={"size": 5},
                hovertemplate="época=%{x}<br>objetivo train=%{y:.4f}<extra></extra>",
            )
        )
        validation = ordered.loc[np.isfinite(ordered["validation_log_loss"])]
        figure.add_trace(
            go.Scatter(
                x=validation["epoch_number"],
                y=validation["validation_log_loss"],
                mode="lines+markers",
                name=f"fold {fold} · val log loss",
                legendgroup=f"fold-{fold}",
                line={"color": color, "width": 2.7},
                marker={"size": 7, "symbol": "circle-open"},
                hovertemplate="época=%{x}<br>val log loss=%{y:.4f}<extra></extra>",
            )
        )
        if not validation.empty:
            best = validation.loc[validation["validation_log_loss"].idxmin()]
            figure.add_trace(
                go.Scatter(
                    x=[best["epoch_number"]],
                    y=[best["validation_log_loss"]],
                    mode="markers",
                    name=f"fold {fold} · mejor época",
                    legendgroup=f"fold-{fold}",
                    showlegend=False,
                    marker={"color": color, "size": 13, "symbol": "star"},
                    hovertemplate="mejor época=%{x}<br>val=%{y:.4f}<extra></extra>",
                )
            )
    figure.add_annotation(
        text=(
            "La curva train es el objetivo optimizado (incluye consistencia y, en nodos 07/09, auxiliares); "
            "no es exactamente la misma magnitud que val log loss."
        ),
        x=0,
        y=-0.22,
        xref="paper",
        yref="paper",
        showarrow=False,
        align="left",
        font={"size": 11},
    )
    figure.update_xaxes(title="época")
    figure.update_yaxes(title="pérdida")
    return _base_layout(
        figure,
        title=f"Curvas por fold · {_trajectory_label(first)}",
        height=610,
    )


def diagnostic_figure(diagnostics: pd.DataFrame) -> go.Figure:
    usable = diagnostics.loc[
        diagnostics.get("validation_points", pd.Series(dtype=float)).ge(4)
    ].copy()
    if usable.empty:
        return _empty_figure(
            "Diagnóstico de ajuste",
            "Se necesitan al menos cuatro evaluaciones de validación por fold.",
        )
    usable["trajectory_short"] = usable["trajectory_id"].astype(str).str.slice(0, 8)
    figure = px.scatter(
        usable,
        x="train_drop",
        y="validation_rebound",
        color="diagnosis",
        symbol="node",
        facet_col="phase",
        hover_data=[
            "run_id",
            "study_name",
            "trial_number",
            "trajectory_short",
            "fold",
            "validation_best",
            "best_epoch",
            "epochs_recorded",
            "validation_step_std",
        ],
        labels={
            "train_drop": "caída del objetivo de entrenamiento",
            "validation_rebound": "rebote final sobre el mejor val log loss",
            "diagnosis": "señal heurística",
            "phase": "fase",
        },
    )
    figure.add_hline(y=0.03, line_dash="dash", line_color="gray")
    return _base_layout(
        figure,
        title="Señales de sobreajuste, inestabilidad y aprendizaje incompleto",
        height=560,
    )


def _atomic_csv(frame: pd.DataFrame, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f"{destination.name}.tmp.{os.getpid()}")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, destination)


def write_dashboard(
    snapshot: TrajectorySnapshot,
    output_dir: str | Path,
    figures: Mapping[str, go.Figure] | None = None,
) -> Path:
    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    _atomic_csv(snapshot.catalog, destination / "run_catalog.csv")
    _atomic_csv(snapshot.trials, destination / "trials_snapshot.csv")
    _atomic_csv(snapshot.histories, destination / "learning_histories_snapshot.csv")
    _atomic_csv(snapshot.fold_scores, destination / "fold_scores_snapshot.csv")
    _atomic_csv(snapshot.diagnostics, destination / "learning_diagnostics.csv")
    _atomic_csv(snapshot.trajectory_summary, destination / "trajectory_summary.csv")
    _atomic_csv(snapshot.final_progress, destination / "final_cv5_progress.csv")
    _atomic_csv(snapshot.final_predictions, destination / "final_oof_predictions.csv")
    _atomic_csv(snapshot.final_metrics, destination / "final_oof_metrics.csv")
    if figures is None:
        figures = {
            "Avance final CV5": final_progress_figure(snapshot.final_progress),
            "Métricas OOF finales": final_metrics_figure(snapshot.final_metrics),
            "Estabilidad final por fold": final_fold_figure(snapshot.final_progress),
            "Distribución OOF": probability_distribution_figure(
                snapshot.final_predictions
            ),
            "Calibración OOF": calibration_figure(snapshot.final_predictions),
            "Acuerdo entre finalistas": finalist_agreement_figure(
                snapshot.final_predictions
            ),
            "Robustez por adquisición": acquisition_family_figure(
                snapshot.final_predictions
            ),
            "Trayectoria de Optuna": optimization_figure(snapshot.trials),
            "Comparación entre redes": network_comparison_figure(snapshot.trials),
            "Costo frente a rendimiento": efficiency_figure(snapshot.trials),
            "Mejores trials por fold": best_fold_comparison_figure(
                snapshot.trials, snapshot.fold_scores
            ),
            "Espacio de hiperparámetros": hyperparameter_figure(snapshot.trials),
            "Curvas de aprendizaje": learning_curve_figure(snapshot.histories),
            "Diagnóstico de ajuste": diagnostic_figure(snapshot.diagnostics),
        }
    sections: list[str] = []
    for index, (name, figure) in enumerate(figures.items()):
        sections.append(f"<section><h2>{html.escape(name)}</h2>")
        sections.append(
            pio.to_html(
                figure,
                full_html=False,
                include_plotlyjs=index == 0,
                config=PLOTLY_CONFIG,
            )
        )
        sections.append("</section>")
    warning_html = "".join(f"<li>{html.escape(message)}</li>" for message in snapshot.warnings)
    document = f"""<!doctype html>
<html lang="es">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Trayectorias CNN nodos 06, 07 y 09</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 24px auto; max-width: 1500px; color: #172033; }}
    h1, h2 {{ font-weight: 600; }}
    .meta {{ color: #596273; }}
    section {{ margin: 32px 0 56px; }}
    ul:empty {{ display: none; }}
  </style>
</head>
<body>
  <h1>Trayectorias CNN · nodos 06, 07 y 09</h1>
  <p class="meta">Snapshot UTC: {html.escape(snapshot.created_at)}. Menor log loss es mejor.</p>
  <ul>{warning_html}</ul>
  {''.join(sections)}
</body>
</html>
"""
    output_path = destination / "dashboard.html"
    temporary = output_path.with_name(f"{output_path.name}.tmp.{os.getpid()}")
    temporary.write_text(document, encoding="utf-8")
    os.replace(temporary, output_path)
    manifest = {
        "created_at": snapshot.created_at,
        "dashboard": str(output_path),
        "runs": snapshot.catalog["run_key"].tolist() if not snapshot.catalog.empty else [],
        "n_trials": len(snapshot.trials),
        "n_histories_rows": len(snapshot.histories),
        "n_final_fold_rows": len(snapshot.final_progress),
        "n_final_predictions": len(snapshot.final_predictions),
        "warnings": list(snapshot.warnings),
    }
    manifest_path = destination / "dashboard_manifest.json"
    temporary_manifest = manifest_path.with_name(
        f"{manifest_path.name}.tmp.{os.getpid()}"
    )
    temporary_manifest.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    os.replace(temporary_manifest, manifest_path)
    return output_path
