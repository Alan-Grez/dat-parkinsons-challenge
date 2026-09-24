from __future__ import annotations

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
DESTINATION = ROOT / "notebooks" / "09_refinamiento_finalistas_optuna.ipynb"


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
# 09 - Refinamiento fino de los tres mejores candidatos

Este nodo no repite la búsqueda amplia. Espera a que el nodo 07 complete su evaluación OOF de cinco folds y selecciona exactamente tres puntos de partida:

1. los dos mejores candidatos por **log loss OOF cross-calibrado de cinco folds**;
2. el mejor candidato 3D;
3. si el 3D ya pertenece al top 2, completa el tercer cupo con el siguiente candidato del ranking.

Cada rama conserva arquitectura, pooling, variante y lateralidad. Optuna sólo explora un vecindario local informado por las mejores trayectorias del nodo 07: learning rate más bajo, regularización, dropout, intensidad de augmentation y configuración tabular. Cada trial usa tres folds; los tres ganadores se reentrenan después con cinco folds congelados.
"""
    ),
    code(
        """
from __future__ import annotations

import json
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
from modeling.node09_refinement import (
    RefinementExperiment,
    prepare_refinement,
    run_final_stage,
    run_search_stage,
)

RUN_ID = os.environ.get("DAT_NODE09_RUN_ID", "node09_fine_v1")
SOURCE_NODE07_RUN_ID = os.environ.get("DAT_NODE09_SOURCE_RUN_ID", "dat_spect_slab_v4")
NODE4_PROFILE = os.environ.get("DAT_NODE09_NODE4_PROFILE", "v3")
TRIALS_PER_MODEL = int(os.environ.get("DAT_NODE09_TRIALS_PER_MODEL", "18"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

EXPERIMENT = RefinementExperiment(
    run_id=RUN_ID,
    source_node07_run_id=SOURCE_NODE07_RUN_ID,
)
EXPERIMENT = replace(
    EXPERIMENT,
    data=replace(EXPERIMENT.data, node4_profile=NODE4_PROFILE),
    search=replace(EXPERIMENT.search, trials_per_model=TRIALS_PER_MODEL),
)
RUN_DIR = PROJECT_ROOT / "outputs" / "private_eda" / "node09_runs" / RUN_ID

display(Markdown(
    f"**Run nodo 09:** `{RUN_ID}` · **fuente nodo 07:** `{SOURCE_NODE07_RUN_ID}` · "
    f"**device:** `{DEVICE}` · **trials completos por rama:** `{TRIALS_PER_MODEL}` · "
    f"**máximo:** `{EXPERIMENT.search.max_epochs_search}` épocas"
))
"""
    ),
    markdown(
        """
## 1. Gate de comparabilidad

La selección no se habilita con OOF parciales ni con el score de los tres folds de Optuna. Debe existir `final_metrics.csv` del nodo 07, lo que implica que sus tres finalistas ya pasaron por el mismo CV5.
"""
    ),
    code(
        """
SOURCE_RUN_DIR = (
    PROJECT_ROOT / "outputs" / "private_eda" / "node07_runs" / SOURCE_NODE07_RUN_ID
)
source_metrics_path = SOURCE_RUN_DIR / "final" / "final_metrics.csv"
source_finalists_path = SOURCE_RUN_DIR / "search" / "finalists.json"
source_oof = sorted((SOURCE_RUN_DIR / "final").glob("oof_*.csv"))
display(pd.DataFrame({
    "artefacto": ["finalistas búsqueda", "OOF finalistas completos", "ranking CV5 final"],
    "estado": [
        source_finalists_path.exists(),
        len(source_oof),
        source_metrics_path.exists(),
    ],
}))
if not source_metrics_path.exists():
    raise RuntimeError(
        "El nodo 07 todavía no terminó los tres finalistas CV5. "
        "Reanuda su celda final y vuelve aquí cuando exista final_metrics.csv."
    )
display(pd.read_csv(source_metrics_path))
"""
    ),
    markdown(
        """
## 2. Preparación y selección de las tres referencias

Se reutilizan los mismos 1362 pacientes, caches y folds congelados. Esta etapa no vuelve a ejecutar registro ni extracción de características. La tabla permite comprobar qué modelo entró por top global, cuál por mejor 3D y cuál completó diversidad si hubo solapamiento.
"""
    ),
    code(
        """
with RunLock(RUN_DIR / "prepare.lock"):
    prepared = prepare_refinement(EXPERIMENT, project_root=PROJECT_ROOT)

references = pd.read_csv(RUN_DIR / "config" / "selected_references.csv")
display(references)
display(pd.DataFrame({
    "indicador": [
        "Pacientes únicos", "Ramas Optuna", "Folds por trial",
        "Trials completos por rama", "Entrenamientos CV de búsqueda objetivo",
        "Folds de evaluación final",
    ],
    "valor": [
        prepared.cohort["uid"].nunique(), 3,
        EXPERIMENT.search.n_splits_search,
        EXPERIMENT.search.trials_per_model,
        3 * EXPERIMENT.search.trials_per_model * EXPERIMENT.search.n_splits_search,
        EXPERIMENT.search.n_splits_final,
    ],
}))
"""
    ),
    code(
        """
spaces = json.loads(
    (RUN_DIR / "config" / "refinement_spaces.json").read_text(encoding="utf-8")
)["spaces"]
space_rows = []
for source_id, space in spaces.items():
    source = next(
        item for item in prepared.references if item["source_candidate_id"] == source_id
    )
    space_rows.append({
        "source_candidate_id": source_id,
        "architecture": source["architecture"],
        "source_learning_rate": source["train_config"]["learning_rate"],
        "node09_learning_rate_min": space["learning_rate"][0],
        "node09_learning_rate_max": space["learning_rate"][1],
        "dropout_range": tuple(space["dropout"]),
        "weight_decay_range": tuple(space["weight_decay"]),
        "tabular_embedding_dim": tuple(space["tabular_embedding_dim"]),
        "feature_top_k": tuple(space["feature_top_k"]),
        "pca_variance": tuple(space["pca_variance"]),
    })
display(pd.DataFrame(space_rows))
"""
    ),
    markdown(
        """
## 3. Tres búsquedas Optuna finas y reanudables

Son **tres estudios**, no tres trials. Por defecto cada estudio exige 18 trials completos y cada trial recorre tres folds. Los primeros ocho trials completos forman el calentamiento del pruner; después puede cortar configuraciones claramente inferiores. Un apagado conserva SQLite, checkpoints por época e historias.
"""
    ),
    code(
        """
with RunLock(RUN_DIR / "search.lock"):
    search_result = run_search_stage(prepared, EXPERIMENT, device=DEVICE)
display(search_result.summary)
display(Markdown(
    "Abre o vuelve a ejecutar el **nodo 08** y selecciona "
    f"`node09:{RUN_ID}` para comparar estas trayectorias con los nodos 06 y 07."
))
"""
    ),
    markdown(
        """
## 4. Evaluación final de los tres ganadores

Optuna entrega un ganador por cada rama. Sólo esos tres modelos se reentrenan en los cinco folds congelados; son 15 entrenamientos finales. El fold externo no controla early stopping: el número de épocas proviene de la mediana de la búsqueda, con un mínimo de ocho para evitar el entrenamiento demasiado corto observado antes.
"""
    ),
    code(
        """
with RunLock(RUN_DIR / "final.lock"):
    final_result = run_final_stage(prepared, EXPERIMENT, device=DEVICE)
display(final_result.metrics)
display(Markdown(
    f"**Modelo primario nodo 09:** `{final_result.deployment_manifest['primary_candidate_id']}` · "
    f"**modelos por fold del primario:** `{final_result.deployment_manifest['n_fold_models']}`"
))
"""
    ),
    markdown(
        """
## Lectura del resultado

El nodo 09 mejora la resolución de la optimización, pero no garantiza una mejora real. Se promueve un candidato sólo si el CV5 congelado supera al nodo 07 en log loss cross-calibrado y conserva estabilidad por fold y familia. Para revisar progreso parcial, curvas de aprendizaje, pruning, calibración y comparación entre redes, usa el nodo 08.
"""
    ),
]

DESTINATION.parent.mkdir(parents=True, exist_ok=True)
nbf.write(notebook, DESTINATION)
print(DESTINATION)

