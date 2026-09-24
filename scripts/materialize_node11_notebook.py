from __future__ import annotations

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
DESTINATION = ROOT / "notebooks" / "11_ensemble_confirmatorio_hgb_cnn25d.ipynb"


def code(source: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(source.strip() + "\n")


def markdown(source: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(source.strip() + "\n")


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
# 11 — Mejor tiro confirmatorio: regional HGB + CNN 2.5D

Este nodo prueba la complementariedad observada entre el experto regional del nodo 10 y la CNN **2.5D image-only** del nodo 6. Ambos usan los mismos pacientes, manifiestos de folds y semilla base.

La mezcla primaria queda congelada antes de mirar este CV5: **50% de cada probabilidad raw**. Sólo después se ajusta una temperatura de forma cross-fitted. La curva completa de pesos se presenta como diagnóstico exploratorio y no reemplaza el resultado primario.

La CNN puede recorrer hasta 500 épocas, con paciencia 80 y checkpoint por época. En cada fold externo, dos folds internos del train seleccionan `best_epoch`; posteriormente se reentrena sobre todo el train externo hasta la mediana de esas épocas. El fold externo se predice una sola vez y nunca controla el early stopping.

HGB aplica el contrato análogo: hasta 500 iteraciones, early stopping interno con paciencia 60, selección de `best_iteration` y refit sobre todo el train externo.
"""
    ),
    code(
        """
from __future__ import annotations

import os
import sys
from dataclasses import replace
from pathlib import Path

import pandas as pd
import torch
from IPython.display import Markdown, display

PROJECT_ROOT = Path.cwd().resolve()
if PROJECT_ROOT.name.lower() == "notebooks":
    PROJECT_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modeling.cnn.io import RunLock
from modeling.node11_confirmatory import (
    ExperimentConfig,
    blend_weight_figure,
    cnn_epoch_loss_figure,
    hgb_iteration_loss_figure,
    optuna_pairwise_scatter_figures,
    prepare_experiment,
    run_final_stage,
    run_search_stage,
)

RUN_ID = os.environ.get("DAT_NODE11_RUN_ID", "node11_best_shot_v1")
COMPLETED_TRIALS = max(10, int(os.environ.get("DAT_NODE11_COMPLETED_TRIALS", "10")))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

EXPERIMENT = ExperimentConfig(run_id=RUN_ID)
EXPERIMENT = replace(
    EXPERIMENT,
    search=replace(EXPERIMENT.search, completed_trials_per_expert=COMPLETED_TRIALS),
)
RUN_DIR = PROJECT_ROOT / "outputs" / "private_eda" / "node11_runs" / RUN_ID

display(Markdown(
    f"**Run:** `{RUN_ID}` · **device:** `{DEVICE}` · **épocas CNN máximas:** "
    f"`{EXPERIMENT.train.max_epochs}` · **patience CNN:** `{EXPERIMENT.train.cnn_patience}` · "
    f"**iteraciones HGB máximas:** `{EXPERIMENT.train.hgb_max_iter}` · "
    f"**trials COMPLETE por experto:** `{EXPERIMENT.search.effective_completed_trials}`"
))
"""
    ),
    markdown(
        """
## 1. Cohorte y folds comunes

Se reutilizan los caches físicos de los nodos 6 y 10. No se repite el registro. `acquisition_family` participa únicamente en el balance de los folds y en la auditoría posterior; nunca entra como predictor.
"""
    ),
    code(
        """
with RunLock(RUN_DIR / "prepare.lock"):
    prepared = prepare_experiment(EXPERIMENT, project_root=PROJECT_ROOT)

display(pd.DataFrame({
    "indicador": [
        "pacientes únicos", "variables regionales", "folds búsqueda comunes",
        "folds finales comunes", "trials completos por experto",
        "máximo de entrenamientos CNN en búsqueda",
    ],
    "valor": [
        prepared.cohort["uid"].nunique(),
        len([c for c in prepared.cohort if c.startswith("regional930_")]),
        EXPERIMENT.search.n_splits_search,
        EXPERIMENT.search.n_splits_final,
        EXPERIMENT.search.effective_completed_trials,
        EXPERIMENT.search.effective_completed_trials * EXPERIMENT.search.n_splits_search,
    ],
}))
display(pd.read_csv(RUN_DIR / "config" / "search_fold_summary.csv"))
display(pd.read_csv(RUN_DIR / "config" / "final_fold_summary.csv"))
"""
    ),
    markdown(
        """
## 2. Optuna prudente y reanudable

Hay sólo dos estudios. El espacio es local y se ancla en los ganadores anteriores: learning rate menor y regularización alrededor de las regiones ya prometedoras. Un trial podado no cuenta; cada estudio continúa hasta alcanzar el mínimo de trials `COMPLETE`.

SQLite conserva el estudio, y la CNN guarda `last.pt`, `last.prev.pt`, `best.pt` e historia por época para cada combinación y fold. Si la ejecución se corta, vuelve a ejecutar esta celda con el mismo `RUN_ID`.
"""
    ),
    code(
        """
with RunLock(RUN_DIR / "search.lock"):
    search_result = run_search_stage(prepared, EXPERIMENT, device=DEVICE)

display(search_result.summary)
assert (search_result.summary["completed_trials"] >= 10).all()
"""
    ),
    markdown(
        """
### Regiones exploradas por Optuna

Cada matriz contiene todos los pares de hiperparámetros del estudio. Cada punto es un trial completo y el color representa su log loss CV3. Para learning rate y regularización se muestra `log10`, evitando que los puntos queden visualmente comprimidos.
"""
    ),
    code(
        """
for study_name, figure in optuna_pairwise_scatter_figures(RUN_DIR).items():
    display(Markdown(f"#### `{study_name}`"))
    figure.show()
"""
    ),
    markdown(
        """
## 3. CV5 final sin usar el fold externo para early stopping

Para cada fold final, el HGB y la CNN reciben exactamente los mismos UIDs de train y validación. La selección de época/iteración ocurre sólo dentro del train. El resultado guarda los dos expertos, la mezcla primaria 50/50, calibración, bootstrap pareado, peor familia de adquisición y una curva exploratoria de pesos.

Esta etapa también es reanudable. La CNN puede tardar: por fold se ejecutan dos trayectorias internas de hasta 500 épocas y un refit hasta el `best_epoch` robusto.
"""
    ),
    code(
        """
with RunLock(RUN_DIR / "final.lock"):
    final_result = run_final_stage(prepared, EXPERIMENT, device=DEVICE)

display(final_result.metrics)
display(final_result.bootstrap_differences)
display(pd.read_csv(RUN_DIR / "final" / "cnn_epoch_selection.csv"))
display(pd.read_csv(RUN_DIR / "final" / "hgb_iteration_selection.csv"))
display(Markdown(
    f"**Primario preespecificado:** `{final_result.deployment_manifest['primary_family']}` · "
    f"**folds comunes:** `{final_result.deployment_manifest['n_outer_folds']}` · "
    f"**fold externo usado para early stopping:** "
    f"`{final_result.deployment_manifest['outer_fold_used_for_early_stopping']}`"
))
"""
    ),
    markdown(
        """
## 4. Época versus loss: diagnóstico de sobreajuste

Las líneas sólidas son log loss de validación **interna** y las punteadas son la pérdida de entrenamiento. Las estrellas indican los `best_epoch` usados para decidir cuántas épocas tendrá el refit de cada fold externo.

Una separación creciente entre train y validación, acompañada de una validación que empeora, indica sobreajuste. Si ambas siguen descendiendo al detenerse, la paciencia u horizonte podrían seguir siendo insuficientes.
"""
    ),
    code(
        """
cnn_epoch_loss_figure(RUN_DIR).show()
"""
    ),
    markdown("## 5. Iteración versus loss del regional HGB"),
    code(
        """
hgb_iteration_loss_figure(RUN_DIR).show()
"""
    ),
    markdown(
        """
## 6. Complementariedad final

El diamante 50/50 es el resultado primario porque su peso fue congelado antes del CV5. La estrella marca el mínimo retrospectivo de la curva y sirve sólo para formular el siguiente experimento; no debe reportarse como estimación imparcial del desempeño.
"""
    ),
    code(
        """
blend_weight_figure(RUN_DIR).show()
display(pd.read_json(RUN_DIR / "final" / "complementarity.json", typ="series"))
display(pd.read_csv(RUN_DIR / "final" / "subgroup_metrics_by_acquisition_family.csv"))
"""
    ),
    markdown(
        """
## Criterio de promoción

Promover la mezcla sólo si mejora el log loss OOF frente al HGB regional, el bootstrap pareado favorece la mezcla, no empeora sustancialmente la peor familia soportada y las curvas internas no muestran inestabilidad grave. El leaderboard sigue siendo validación externa adicional, no sustituto del CV.
"""
    ),
]

DESTINATION.parent.mkdir(parents=True, exist_ok=True)
nbf.write(notebook, DESTINATION)
print(DESTINATION)
