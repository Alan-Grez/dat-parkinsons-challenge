from __future__ import annotations

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = ROOT / "notebooks" / "08_trayectorias_optuna_cnn_plotly.ipynb"


def code(source: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(source.strip() + "\n")


def markdown(source: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(source.strip() + "\n")


def main() -> Path:
    notebook = nbf.v4.new_notebook()
    notebook["metadata"] = {
        "kernelspec": {
            "display_name": "dat-parkinsons-challenge",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.12"},
    }
    notebook["cells"] = [
        markdown(
            r"""
# 08 - Trayectorias de Optuna y aprendizaje CNN

Este nodo es **sólo de lectura**: reconstruye el historial persistido de los nodos 06, 07, 09 y 10 sin abrir los NIfTI ni interferir con un entrenamiento en curso. Lee:

- las bases SQLite de Optuna mediante una copia transaccional temporal;
- los `training_history.csv` / `history.csv` guardados después de cada época;
- los scores de cada fold incluidos en los trials completos;
- las evaluaciones finales de cinco folds cuando estén disponibles.

Por eso los resultados sobreviven a un apagado: están en disco, no únicamente en memoria RAM. Para ver avances nuevos basta con volver a ejecutar este notebook.
"""
        ),
        code(
            """
from __future__ import annotations

import os
import sys
from pathlib import Path

import ipywidgets as widgets
import pandas as pd
from IPython.display import FileLink, Markdown, clear_output, display

PROJECT_ROOT = Path.cwd().resolve()
if PROJECT_ROOT.name.lower() == "notebooks":
    PROJECT_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modeling.trajectory_viz import (
    acquisition_family_figure,
    best_fold_comparison_figure,
    calibration_figure,
    diagnostic_figure,
    discover_runs,
    efficiency_figure,
    final_fold_figure,
    final_metrics_figure,
    final_progress_figure,
    finalist_agreement_figure,
    hyperparameter_figure,
    learning_curve_figure,
    load_trajectory_snapshot,
    network_comparison_figure,
    optimization_figure,
    probability_distribution_figure,
    write_dashboard,
)
from modeling.trajectory_viz.figures import PLOTLY_CONFIG

# Opcional: "node06:cnn_compact_v1,node07:dat_spect_slab_v4,node09:node09_fine_v1,node10:node10_hybrid_v1".
RUN_KEYS_TEXT = os.environ.get("DAT_TRAJECTORY_RUNS", "").strip()
RUN_KEYS = [value.strip() for value in RUN_KEYS_TEXT.split(",") if value.strip()] or None
EXPORT_DIR = (
    PROJECT_ROOT / "outputs" / "private_eda" / "trajectory_dashboard" / "latest"
)
"""
        ),
        markdown(
            """
## 1. Descubrimiento y snapshot consistente

Si `RUN_KEYS` queda en `None`, se toma automáticamente el run con actividad más reciente de cada nodo. La copia temporal de SQLite incluye incluso trials `RUNNING`, `PRUNED` o `FAIL`; el archivo original permanece intacto.
"""
        ),
        code(
            """
catalog = discover_runs(PROJECT_ROOT)
display(catalog)

snapshot = load_trajectory_snapshot(PROJECT_ROOT, RUN_KEYS)
display(Markdown(f"**Snapshot UTC:** `{snapshot.created_at}`"))
display(snapshot.catalog)
if snapshot.warnings:
    display(Markdown("**Advertencias de lectura:**\\n\\n" + "\\n".join(f"- {item}" for item in snapshot.warnings)))

coverage = pd.DataFrame({
    "indicador": [
        "trials registrados",
        "trials completos",
        "trials en curso",
        "trials podados",
        "trials fallidos",
        "archivos de historial",
        "filas época-fold",
        "trayectorias diagnosticables",
        "finalistas CV5 observados",
        "modelos-fold finales completos",
        "modelos-fold finales en curso",
        "predicciones OOF finales disponibles",
    ],
    "valor": [
        len(snapshot.trials),
        int(snapshot.trials.get("state", pd.Series(dtype=str)).eq("COMPLETE").sum()),
        int(snapshot.trials.get("state", pd.Series(dtype=str)).eq("RUNNING").sum()),
        int(snapshot.trials.get("state", pd.Series(dtype=str)).eq("PRUNED").sum()),
        int(snapshot.trials.get("state", pd.Series(dtype=str)).eq("FAIL").sum()),
        int(snapshot.catalog.get("history_files", pd.Series(dtype=int)).sum()),
        len(snapshot.histories),
        int(snapshot.diagnostics.get("validation_points", pd.Series(dtype=int)).ge(4).sum()),
        int(snapshot.final_progress.get("candidate_id", pd.Series(dtype=str)).nunique()),
        int(snapshot.final_progress.get("status", pd.Series(dtype=str)).eq("complete").sum()),
        int(snapshot.final_progress.get("status", pd.Series(dtype=str)).eq("training").sum()),
        len(snapshot.final_predictions),
    ],
})
display(coverage)
"""
        ),
        markdown(
            """
## 2. Trayectoria global de la búsqueda

Los puntos son trials completos; la línea discontinua es el mejor log loss acumulado dentro de cada estudio. Una meseta indica que ampliar ciegamente el número de trials probablemente rinde poco; mejoras tardías indican que todavía vale la pena explorar alrededor de esa zona.
"""
        ),
        code(
            """
fig_optimization = optimization_figure(snapshot.trials)
fig_optimization.show(config=PLOTLY_CONFIG)
"""
        ),
        markdown(
            """
## 3. Comparación entre redes y estabilidad entre folds

La primera figura compara toda la distribución de trials. La segunda compara únicamente el mejor trial actual de cada red y conserva sus tres folds separados. El promedio aislado puede esconder un fold frágil; por eso ambos gráficos son necesarios.
"""
        ),
        code(
            """
fig_networks = network_comparison_figure(snapshot.trials)
fig_networks.show(config=PLOTLY_CONFIG)

fig_folds = best_fold_comparison_figure(snapshot.trials, snapshot.fold_scores)
fig_folds.show(config=PLOTLY_CONFIG)

fig_efficiency = efficiency_figure(snapshot.trials)
fig_efficiency.show(config=PLOTLY_CONFIG)
"""
        ),
        markdown(
            """
## 4. Evaluación final de cinco folds

Esta sección se actualiza mientras corre `run_final_stage`. Distingue resultados **parciales** de métricas finales: un candidato sólo recibe calibración cruzada cuando completó sus cinco folds. La matriz muestra el avance y el log loss de cada fold; la dispersión permite detectar si una media aparentemente buena depende de un único fold favorable.
"""
        ),
        code(
            """
fig_final_progress = final_progress_figure(snapshot.final_progress)
fig_final_progress.show(config=PLOTLY_CONFIG)

fig_final_folds = final_fold_figure(snapshot.final_progress)
fig_final_folds.show(config=PLOTLY_CONFIG)

fig_final_metrics = final_metrics_figure(snapshot.final_metrics)
fig_final_metrics.show(config=PLOTLY_CONFIG)

if not snapshot.final_metrics.empty:
    final_metric_columns = [
        column for column in [
            "candidate_id", "architecture", "feature_variant", "lateral_strategy",
            "folds_completed", "expected_folds", "n_predictions", "metrics_source",
            "raw_log_loss", "calibrated_log_loss", "raw_auc", "calibrated_auc",
            "raw_brier", "calibrated_brier", "raw_ece_10", "calibrated_ece_10",
        ] if column in snapshot.final_metrics
    ]
    display(snapshot.final_metrics[final_metric_columns].sort_values(
        ["folds_completed", "raw_log_loss"], ascending=[False, True]
    ))
"""
        ),
        markdown(
            """
### Separación, calibración y complementariedad

El selector usa las predicciones OOF disponibles del finalista. Para candidatos incompletos la lectura es provisional y sólo se muestra la probabilidad sin calibrar. El panel de acuerdo ayuda a decidir si un ensemble puede aportar: redes idénticas en sus probabilidades agregan poco, mientras diferencias razonables pueden ser complementarias.
"""
        ),
        code(
            """
final_candidate_rows = (
    snapshot.final_metrics.sort_values(
        ["folds_completed", "raw_log_loss"], ascending=[False, True]
    )
    if not snapshot.final_metrics.empty
    else pd.DataFrame()
)
final_candidate_options = []
for row in final_candidate_rows.itertuples(index=False):
    calibrated = getattr(row, "calibrated_log_loss", float("nan"))
    score = calibrated if pd.notna(calibrated) else row.raw_log_loss
    status = "final" if bool(row.evaluation_complete) else "parcial"
    final_candidate_options.append((
        f"{row.architecture} · {str(row.candidate_id)[:8]} · "
        f"{row.folds_completed}/{row.expected_folds} folds · {status} · log loss={score:.4f}",
        str(row.candidate_id),
    ))

final_candidate_selector = widgets.Dropdown(
    options=final_candidate_options,
    description="Finalista:",
    layout=widgets.Layout(width="95%"),
)
final_candidate_output = widgets.Output()

def render_final_candidate(change=None):
    with final_candidate_output:
        clear_output(wait=True)
        if not final_candidate_options:
            display(Markdown("Aún no hay predicciones OOF finales."))
            return
        candidate_id = final_candidate_selector.value
        probability_distribution_figure(
            snapshot.final_predictions, candidate_id
        ).show(config=PLOTLY_CONFIG)
        calibration_figure(
            snapshot.final_predictions, candidate_id
        ).show(config=PLOTLY_CONFIG)

final_candidate_selector.observe(render_final_candidate, names="value")
display(final_candidate_selector, final_candidate_output)
render_final_candidate()

fig_final_agreement = finalist_agreement_figure(snapshot.final_predictions)
fig_final_agreement.show(config=PLOTLY_CONFIG)

fig_final_families = acquisition_family_figure(snapshot.final_predictions)
fig_final_families.show(config=PLOTLY_CONFIG)
"""
        ),
        markdown(
            """
## 5. Espacio de hiperparámetros

Cada línea es un trial. El color representa log loss: permite ver interacciones que una tabla de números oculta. El selector separa los estudios para no comparar parámetros con significados incompatibles.
"""
        ),
        code(
            """
study_rows = (
    snapshot.trials[["run_key", "study_name"]]
    .dropna()
    .drop_duplicates()
    .sort_values(["run_key", "study_name"])
    if not snapshot.trials.empty
    else pd.DataFrame()
)
study_options = [
    (f"{row.run_key} · {row.study_name}", (row.run_key, row.study_name))
    for row in study_rows.itertuples(index=False)
]
study_selector = widgets.Dropdown(
    options=study_options,
    description="Estudio:",
    layout=widgets.Layout(width="95%"),
)
study_output = widgets.Output()

def render_hyperparameters(change=None):
    with study_output:
        clear_output(wait=True)
        if not study_options:
            display(Markdown("Aún no hay estudios Optuna para visualizar."))
            return
        run_key, study_name = study_selector.value
        hyperparameter_figure(
            snapshot.trials, study_name, run_key=run_key
        ).show(config=PLOTLY_CONFIG)

study_selector.observe(render_hyperparameters, names="value")
display(study_selector, study_output)
render_hyperparameters()
"""
        ),
        markdown(
            """
## 6. Curvas por época y fold

Se muestran por separado el objetivo de entrenamiento y el log loss de validación. No deben restarse literalmente: el objetivo `train` incluye regularización por consistencia y, en los nodos 07, 09 y 10, puede incluir objetivos auxiliares. Sí sirven sus tendencias: entrenamiento descendente junto con validación que rebota es señal de posible sobreajuste.
"""
        ),
        code(
            """
trajectory_rows = (
    snapshot.histories[
        ["run_key", "phase", "study_name", "trajectory_id", "trial_number", "objective_log_loss"]
    ]
    .drop_duplicates()
    .sort_values(["phase", "objective_log_loss", "run_key", "study_name"], na_position="last")
    if not snapshot.histories.empty
    else pd.DataFrame()
)
trajectory_options = []
for row in trajectory_rows.itertuples(index=False):
    trial = f"trial {int(row.trial_number)}" if pd.notna(row.trial_number) else f"hash {str(row.trajectory_id)[:8]}"
    score = f" · {row.objective_log_loss:.4f}" if pd.notna(row.objective_log_loss) else " · activo/no mapeado"
    trajectory_options.append((
        f"{row.run_key} · {row.phase} · {row.study_name} · {trial}{score}",
        (row.run_key, row.phase, str(row.trajectory_id)),
    ))

trajectory_selector = widgets.Dropdown(
    options=trajectory_options,
    description="Trayectoria:",
    layout=widgets.Layout(width="95%"),
)
trajectory_output = widgets.Output()

def render_learning_curve(change=None):
    with trajectory_output:
        clear_output(wait=True)
        if not trajectory_options:
            display(Markdown("Aún no hay historiales por época."))
            return
        run_key, phase, trajectory_id = trajectory_selector.value
        learning_curve_figure(
            snapshot.histories,
            trajectory_id,
            run_key=run_key,
            phase=phase,
        ).show(config=PLOTLY_CONFIG)

trajectory_selector.observe(render_learning_curve, names="value")
display(trajectory_selector, trajectory_output)
render_learning_curve()
"""
        ),
        markdown(
            """
## 7. Señales de sobreajuste y subajuste

El diagnóstico es deliberadamente heurístico. **Sobreajuste** requiere que el objetivo de entrenamiento siga bajando mientras validación se aleja al menos `0.03` de su mejor punto. **Posible subajuste/optimización débil** exige poco aprendizaje en train y validación cercana o peor que `0.67`. Los cinco folds finales sólo validan en la última época por diseño, por lo que no alcanzan para este diagnóstico temporal.
"""
        ),
        code(
            """
fig_diagnostics = diagnostic_figure(snapshot.diagnostics)
fig_diagnostics.show(config=PLOTLY_CONFIG)

if not snapshot.trajectory_summary.empty:
    display(
        snapshot.trajectory_summary[
            [
                "run_key", "phase", "study_name", "trial_number", "folds_observed",
                "mean_best_validation", "std_best_validation", "mean_rebound",
                "median_best_epoch", "overfit_fraction", "underfit_fraction",
                "unstable_fraction",
            ]
        ].head(40)
    )
"""
        ),
        markdown(
            """
## 8. Exportación persistente

Se guarda un HTML interactivo autocontenido y snapshots CSV. Volver a ejecutar esta celda actualiza `latest` de forma atómica; no borra ni altera artefactos de los nodos 06, 07, 09 y 10.
"""
        ),
        code(
            """
dashboard_path = write_dashboard(
    snapshot,
    EXPORT_DIR,
    {
        "Avance final CV5": fig_final_progress,
        "Métricas OOF finales": fig_final_metrics,
        "Estabilidad final por fold": fig_final_folds,
        "Distribución OOF": probability_distribution_figure(snapshot.final_predictions),
        "Calibración OOF": calibration_figure(snapshot.final_predictions),
        "Acuerdo entre finalistas": fig_final_agreement,
        "Robustez por adquisición": fig_final_families,
        "Trayectoria de Optuna": fig_optimization,
        "Comparación entre redes": fig_networks,
        "Costo frente a rendimiento": fig_efficiency,
        "Mejores trials por fold": fig_folds,
        "Espacio de hiperparámetros": hyperparameter_figure(snapshot.trials),
        "Curvas de aprendizaje": learning_curve_figure(snapshot.histories),
        "Diagnóstico de ajuste": fig_diagnostics,
    },
)
display(Markdown(f"**Dashboard actualizado:** `{dashboard_path}`"))
display(FileLink(dashboard_path))
"""
        ),
    ]
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(notebook, NOTEBOOK_PATH)
    print(NOTEBOOK_PATH)
    return NOTEBOOK_PATH


if __name__ == "__main__":
    main()
