from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def _loss_color_bounds(values: pd.Series) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if finite.empty:
        return 0.0, 1.0
    low, high = float(finite.min()), float(finite.max())
    return (low, high + 1e-9) if high <= low else (low, high)


def _dimension(column: str, values: pd.Series) -> dict[str, Any]:
    name = column.removeprefix("param__")
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.notna().all():
        array = numeric.to_numpy(dtype=float)
        if name in {"learning_rate", "weight_decay", "l2_regularization"}:
            array = np.log10(np.clip(array, 1e-12, None))
            name = f"log10({name})"
        return {"label": name, "values": array}
    categories = sorted(values.astype(str).unique())
    mapping = {value: index for index, value in enumerate(categories)}
    legend = ", ".join(f"{index}={value}" for value, index in mapping.items())
    return {
        "label": f"{name} [{legend}]",
        "values": values.astype(str).map(mapping).to_numpy(dtype=float),
    }

def optuna_pairwise_scatter_figures(run_dir: Path) -> dict[str, go.Figure]:
    """One lower-triangle scatter matrix per Optuna study.

    Every off-diagonal panel is one searched hyperparameter pair; points are
    trials and colour is the mean cross-validated log loss.
    """

    figures: dict[str, go.Figure] = {}
    for path in sorted((run_dir / "search" / "studies").glob("*_trials.csv")):
        frame = pd.read_csv(path)
        frame = frame.loc[(frame["state"] == "COMPLETE") & frame["objective"].notna()].copy()
        parameter_columns = [column for column in frame if column.startswith("param__")]
        if frame.empty or not parameter_columns:
            continue
        dimensions = [_dimension(column, frame[column]) for column in parameter_columns]
        low, high = _loss_color_bounds(frame["objective"])
        hover = [
            f"trial={int(row.trial)}<br>log loss={float(row.objective):.5f}"
            for row in frame.itertuples()
        ]
        figure = go.Figure(
            data=go.Splom(
                dimensions=dimensions,
                text=hover,
                hovertemplate="%{text}<extra></extra>",
                diagonal_visible=False,
                showupperhalf=False,
                marker={
                    "color": frame["objective"],
                    "colorscale": "Viridis_r",
                    "cmin": low,
                    "cmax": high,
                    "size": 8,
                    "opacity": 0.85,
                    "line": {"width": 0.4, "color": "rgba(20,20,20,0.45)"},
                    "colorbar": {"title": "log loss<br>CV3"},
                },
            )
        )
        study_name = path.stem.removesuffix("_trials")
        figure.update_layout(
            title=(
                f"Optuna · {study_name}: pares de hiperparámetros "
                "(menor log loss es mejor)"
            ),
            height=max(650, 105 * len(dimensions)),
            margin={"l": 70, "r": 90, "t": 90, "b": 70},
            template="plotly_white",
        )
        figures[study_name] = figure
    return figures


def cnn_epoch_loss_figure(run_dir: Path) -> go.Figure:
    paths = sorted(
        (run_dir / "final" / "models" / "cnn_25d_image_only").glob(
            "*/fold_*/epoch_selection/inner_fold_*/training_history.csv"
        )
    )
    if not paths:
        raise FileNotFoundError("Aun no existen historias finales de epoch selection CNN.")
    rows, columns = 3, 2
    figure = make_subplots(
        rows=rows,
        cols=columns,
        subplot_titles=[f"outer fold {fold}" for fold in range(5)],
        shared_xaxes=False,
        vertical_spacing=0.10,
    )
    colours = ("#3b82f6", "#f97316")
    for path in paths:
        outer_fold = int(path.parents[2].name.removeprefix("fold_"))
        inner_fold = int(path.parent.name.removeprefix("inner_fold_"))
        history = pd.read_csv(path)
        history["epoch_display"] = history["epoch"].astype(int) + 1
        row, column = divmod(outer_fold, columns)
        row += 1
        column += 1
        colour = colours[inner_fold % len(colours)]
        showlegend = outer_fold == 0
        figure.add_trace(
            go.Scatter(
                x=history["epoch_display"],
                y=history["train_loss"],
                mode="lines",
                line={"color": colour, "dash": "dot", "width": 1.5},
                opacity=0.75,
                name=f"inner {inner_fold} · train",
                legendgroup=f"inner_{inner_fold}",
                showlegend=showlegend,
            ),
            row=row,
            col=column,
        )
        figure.add_trace(
            go.Scatter(
                x=history["epoch_display"],
                y=history["validation_log_loss"],
                mode="lines",
                line={"color": colour, "width": 2.1},
                name=f"inner {inner_fold} · validación",
                legendgroup=f"inner_{inner_fold}",
                showlegend=showlegend,
            ),
            row=row,
            col=column,
        )
        valid = history["validation_log_loss"].to_numpy(dtype=float)
        if np.isfinite(valid).any():
            best_index = int(np.nanargmin(valid))
            figure.add_trace(
                go.Scatter(
                    x=[int(history.iloc[best_index]["epoch_display"])],
                    y=[float(valid[best_index])],
                    mode="markers",
                    marker={"symbol": "star", "size": 12, "color": colour},
                    name=f"inner {inner_fold} · best_epoch",
                    legendgroup=f"inner_{inner_fold}",
                    showlegend=showlegend,
                    hovertemplate="best epoch=%{x}<br>val=%{y:.5f}<extra></extra>",
                ),
                row=row,
                col=column,
            )
    figure.update_xaxes(title_text="época")
    figure.update_yaxes(title_text="loss")
    figure.update_layout(
        title=(
            "CNN 2.5D · época versus loss en particiones internas "
            "(el fold externo permanece intacto)"
        ),
        height=980,
        template="plotly_white",
        hovermode="x unified",
        margin={"l": 70, "r": 30, "t": 110, "b": 60},
    )
    return figure


def hgb_iteration_loss_figure(run_dir: Path) -> go.Figure:
    paths = sorted(
        (run_dir / "final" / "models" / "regional_hgb").glob(
            "*/fold_*/iteration_history.csv"
        )
    )
    if not paths:
        raise FileNotFoundError("Aun no existen historias finales del HGB regional.")
    figure = go.Figure()
    palette = ("#2563eb", "#dc2626", "#059669", "#7c3aed", "#d97706")
    for path in paths:
        fold = int(path.parent.name.removeprefix("fold_"))
        history = pd.read_csv(path)
        colour = palette[fold % len(palette)]
        figure.add_trace(
            go.Scatter(
                x=history["iteration"],
                y=history["validation_loss"],
                mode="lines",
                line={"color": colour, "width": 2},
                name=f"fold {fold} · validación interna",
            )
        )
        selected = history.loc[history["is_best"].astype(bool)]
        if not selected.empty:
            figure.add_trace(
                go.Scatter(
                    x=selected["iteration"],
                    y=selected["validation_loss"],
                    mode="markers",
                    marker={"symbol": "star", "size": 12, "color": colour},
                    name=f"fold {fold} · best_iteration",
                    showlegend=False,
                )
            )
    figure.update_layout(
        title="Regional HGB · iteración versus loss de early stopping interno",
        xaxis_title="iteración de boosting",
        yaxis_title="log loss",
        template="plotly_white",
        hovermode="x unified",
        height=560,
    )
    return figure


def blend_weight_figure(run_dir: Path) -> go.Figure:
    path = run_dir / "final" / "exploratory_weight_sweep.csv"
    frame = pd.read_csv(path)
    best = frame.loc[frame["raw_log_loss"].idxmin()]
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=100 * frame["regional_weight"],
            y=frame["raw_log_loss"],
            mode="lines",
            line={"width": 2.5, "color": "#2563eb"},
            name="mezcla OOF raw",
        )
    )
    fixed = frame.iloc[(frame["regional_weight"] - 0.5).abs().argmin()]
    figure.add_trace(
        go.Scatter(
            x=[50, 100 * float(best["regional_weight"])],
            y=[float(fixed["raw_log_loss"]), float(best["raw_log_loss"])],
            mode="markers+text",
            marker={"size": 12, "symbol": ["diamond", "star"]},
            text=["primario 50/50", "mínimo exploratorio"],
            textposition="top center",
            name="referencias",
        )
    )
    figure.update_layout(
        title="Complementariedad OOF · peso del HGB regional",
        xaxis_title="peso regional HGB (%)",
        yaxis_title="log loss OOF raw",
        template="plotly_white",
        height=520,
    )
    return figure
