from __future__ import annotations

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
DESTINATION = ROOT / "notebooks" / "10_expertos_hibridos_intensidad_grafos.ipynb"


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
        """
# 10 - Expertos híbridos: intensidad, patrón, topología y grafos

Este nodo investiga la complementariedad observada entre los nodos 06 y 07. No crea una CNN 4D artificial. Construye dos vistas del **mismo crop físico**:

- `volume_intensity`: reconstruye captación registrada truncada como `volume_intensity01 × foreground_p99_registered` y la escala sólo con el train de cada fold;
- `volume_selfnorm`: proyecta esos mismos vóxeles mediante $\\sqrt{d}x/\\lVert x\rVert_2$.

La intensidad entra en la CNN 3D de dos maneras: como valores voxel a voxel de la primera rama y como cuatro escalares explícitos (`L2`, media, p90 y soporte positivo). La segunda rama aprende el patrón sin escala. La concatenación tiene tres compuertas sigmoid para impedir que una vista anule numéricamente a la otra.

Además se comparan un experto regional de 930 variables, topología multiumbral, tres baselines tabulares del mismo grafo, una GCN pequeña, Diffusion Maps con Nyström y una mezcla de subtipos. La etapa final genera OOF de cinco folds, cross-calibra cada experto, calcula una media preespecificada y optimiza un stacking regularizado exploratorio.
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
from modeling.node10_hybrid import (
    ExperimentConfig,
    prepare_experiment,
    run_final_stage,
    run_search_stage,
)

RUN_ID = os.environ.get("DAT_NODE10_RUN_ID", "node10_hybrid_v1")
NODE4_PROFILE = os.environ.get("DAT_NODE10_NODE4_PROFILE", "v3")
COMPLETED_TRIALS = max(10, int(os.environ.get("DAT_NODE10_COMPLETED_TRIALS", "10")))
STACK_TRIALS = max(10, int(os.environ.get("DAT_NODE10_STACK_TRIALS", "10")))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

EXPERIMENT = ExperimentConfig(run_id=RUN_ID)
EXPERIMENT = replace(
    EXPERIMENT,
    data=replace(EXPERIMENT.data, node4_profile=NODE4_PROFILE),
    search=replace(
        EXPERIMENT.search,
        completed_trials_per_model=COMPLETED_TRIALS,
        stack_completed_trials=STACK_TRIALS,
    ),
)
RUN_DIR = PROJECT_ROOT / "outputs" / "private_eda" / "node10_runs" / RUN_ID

display(Markdown(
    f"**Run:** `{RUN_ID}` · **device:** `{DEVICE}` · **familias base:** "
    f"`{len(EXPERIMENT.model_families)}` · **trials COMPLETE mínimos por familia:** "
    f"`{EXPERIMENT.search.effective_completed_trials}`"
))
"""
    ),
    markdown(
        """
## 1. Preparación reanudable y auditoría de contrato

Se reutilizan los 1362 pacientes únicos del nodo 04. El caché se persiste caso a caso; si se interrumpe, continúa con los UIDs pendientes. `acquisition_family` sólo participa en el balance de folds y las auditorías: nunca entra a los predictores.
"""
    ),
    code(
        """
with RunLock(RUN_DIR / "prepare.lock"):
    prepared = prepare_experiment(EXPERIMENT, project_root=PROJECT_ROOT)

display(pd.DataFrame({
    "indicador": [
        "casos", "UID únicos", "familias base", "folds por trial",
        "mínimo COMPLETE por modelo", "folds finales", "entrenamientos CV mínimos de búsqueda",
    ],
    "valor": [
        len(prepared.cohort), prepared.cohort["uid"].nunique(),
        len(EXPERIMENT.model_families), EXPERIMENT.search.n_splits_search,
        EXPERIMENT.search.effective_completed_trials, EXPERIMENT.search.n_splits_final,
        len(EXPERIMENT.model_families) * EXPERIMENT.search.effective_completed_trials * EXPERIMENT.search.n_splits_search,
    ],
}))
display(prepared.cohort[[
    "uid", "is_pathologic", "intensity_l2_norm", "intensity_mean",
    "intensity_p90", "intensity_positive_voxels",
]].head())
"""
    ),
    markdown(
        """
## 2. Nueve búsquedas base con Optuna

Cada estudio debe alcanzar al menos diez trials en estado `COMPLETE`. Un trial `PRUNED` no cuenta: se solicita otro automáticamente. Los modelos neuronales guardan `last.pt`, `last.prev.pt`, `best.pt`, historia por época y predicciones por fold. Los modelos sklearn persisten cada fold completo y Optuna conserva el estado en SQLite.

La ejecución puede ser larga: son como mínimo 270 evaluaciones de fold. Si se apaga el equipo, vuelve a ejecutar esta celda con el mismo `RUN_ID`.
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
## 3. CV5 congelado, calibración y stacking

El ganador de cada familia se reentrena en cinco folds congelados. Después se construye una matriz OOF común: una media igual preespecificada permanece elegible como modelo primario y un décimo estudio Optuna ajusta un meta-modelo Elastic Net sobre logits. Este stacking es exploratorio porque una estimación promocionable exigiría regenerar los expertos dentro de un segundo nivel estrictamente anidado. El meta-modelo también exige diez trials completos. La probabilidad reportada se cross-calibra sin usar el fold evaluado para ajustar su temperatura.

La evaluación guarda además intervalos bootstrap de log loss, peor familia de adquisición con soporte $n\\geq10$ y matrices de correlación/diferencia entre expertos. La familia de adquisición se une sólo después de predecir, exclusivamente para auditoría.
"""
    ),
    code(
        """
with RunLock(RUN_DIR / "final.lock"):
    final_result = run_final_stage(prepared, EXPERIMENT, device=DEVICE)
display(final_result.metrics)
display(Markdown(
    f"**Primario elegible:** `{final_result.deployment_manifest['primary_family']}` · "
    f"**expertos base:** `{final_result.deployment_manifest['n_base_experts']}` · "
    f"**folds por experto:** `{final_result.deployment_manifest['n_fold_models_per_expert']}`"
))
"""
    ),
    markdown(
        """
## 4. Lectura científica

No se promueve automáticamente el modelo más complejo. La comparación central es:

1. ¿La rama dual supera a las ramas previas en log loss CV5?
2. ¿GCN supera a Elastic Net, Random Forest y MLP usando exactamente los mismos nodos?
3. ¿Topología o Diffusion Maps reducen errores distintos?
4. ¿La media preespecificada mejora sin empeorar calibración ni peor familia?
5. ¿El stacking exploratorio justifica pagar el costo de una validación de segundo nivel anidada?

Las regiones son proxies geométricos reproducibles, no segmentaciones clínicas. Un resultado prometedor debe repetirse con semillas, bootstrap de diferencias y auditoría por familia antes de preparar un submission.
"""
    ),
]

DESTINATION.parent.mkdir(parents=True, exist_ok=True)
nbf.write(notebook, DESTINATION)
print(DESTINATION)
