from __future__ import annotations

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
DESTINATION = ROOT / "notebooks" / "07_dat_spect_slab_multitemplate.ipynb"


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
# 07 - Slab DaT-SPECT, self-normalization y registro multi-plantilla

Este nodo es independiente de los resultados del nodo 06 y reutiliza solamente los derivados privados del nodo 04. Implementa:

- crop físico a **2 mm isotrópicos** y slab axial de **12 mm** (seis muestras);
- ROI física estriatal independiente de la etiqueta, con recentrado de captación acotado a 10 mm;
- self-normalization $\hat{x}=\sqrt{d}\,x/\lVert x\rVert_2$ dentro de esa ROI, sin fondo occipital;
- comparación excluyente entre lateralidad canonizada y flip aleatorio;
- blur y ruido correlacionado fuertes expresados en milímetros (`M≈2.5`, `P≈0.9`);
- cuatro objetivos auxiliares: caudado/putamen derecho e izquierdo;
- **930 variables regionales**, con imputación, selector, scaler y PCA ajustados dentro de cada fold;
- registro multi-plantilla fold-safe con prototipos independientes de la etiqueta;
- búsqueda Optuna con tres folds y evaluación congelada de finalistas con cinco folds.

Fuentes principales: [Buddenkotte & Buchert 2024](https://doi.org/10.2967/jnumed.124.267570), [Zhou & Tagare 2021](https://arxiv.org/abs/2112.13637), [Apostolova et al. 2023](https://doi.org/10.1186/s40658-023-00544-9). El registro implementado es una aproximación acotada en PyTorch, no una reproducción exacta de SPM12.
"""
    ),
    code(
        """
from __future__ import annotations

import os
import sys
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from IPython.display import Markdown, display

PROJECT_ROOT = Path.cwd().resolve()
if PROJECT_ROOT.name.lower() == "notebooks":
    PROJECT_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modeling.cnn.io import RunLock
from modeling.dat_spect_v2 import (
    ExperimentConfig,
    ModelConfig,
    MultiTaskDaTClassifier,
    prepare_experiment,
    run_final_stage,
    run_search_stage,
)

RUN_ID = os.environ.get("DAT_NODE07_RUN_ID", "dat_spect_slab_v4")
NODE4_PROFILE = os.environ.get("DAT_NODE07_NODE4_PROFILE", "v3")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

EXPERIMENT = ExperimentConfig(run_id=RUN_ID)
EXPERIMENT = replace(
    EXPERIMENT,
    data=replace(EXPERIMENT.data, node4_profile=NODE4_PROFILE),
)

display(Markdown(
    f"**Run:** `{RUN_ID}` · **device:** `{DEVICE}` · "
    f"**slab:** `{EXPERIMENT.data.slab_slices}` cortes a "
    f"`{EXPERIMENT.data.output_spacing_mm:g} mm` · "
    f"**features regionales:** `{EXPERIMENT.data.regional_feature_budget}`"
))
"""
    ),
    markdown(
        """
## 1. Preparación reanudable

Materializa el cache físico base y congela dos manifiestos independientes: tres folds para búsqueda y cinco folds para evaluación. La unidad es el paciente único. `acquisition_family` sólo ayuda a equilibrar la distribución de protocolos y nunca entra al modelo. Además se conserva un split secundario con familias completamente no vistas como prueba de estrés; no sustituye la CV principal porque una familia contiene 460 casos e impide cinco folds de tamaño comparable.
"""
    ),
    code(
        """
RUN_DIR = PROJECT_ROOT / "outputs" / "private_eda" / "node07_runs" / RUN_ID
with RunLock(RUN_DIR / "prepare.lock"):
    prepared = prepare_experiment(EXPERIMENT, project_root=PROJECT_ROOT)

display(pd.DataFrame({
    "indicador": [
        "Casos",
        "Familias de adquisición",
        "Patológicos",
        "Fondos SBR válidos",
        "Crops base completos",
    ],
    "valor": [
        len(prepared.cohort),
        prepared.cohort["acquisition_family"].nunique(),
        int(prepared.cohort["is_pathologic"].sum()),
        int(prepared.cohort["background_qc_valid"].sum()),
        int(prepared.cohort["base_cache_path"].map(lambda value: Path(value).exists()).sum()),
    ],
}))
display(pd.read_csv(RUN_DIR / "config" / "search_fold_summary.csv"))
display(pd.read_csv(RUN_DIR / "config" / "final_fold_summary.csv"))
display(Markdown("**Prueba de estrés con familias completamente no vistas:**"))
display(pd.read_csv(RUN_DIR / "config" / "family_stress_fold_summary.csv"))
"""
    ),
    code(
        """
architecture_rows = []
for architecture in EXPERIMENT.architectures:
    model = MultiTaskDaTClassifier(ModelConfig(
        architecture=architecture,
        feature_variant="image_only",
        lateral_strategy="canonical",
        pooling="gmp" if architecture == "slab2d" else "gap_max",
    ))
    architecture_rows.append({
        "arquitectura": architecture,
        "parametros_entrenables": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        "embedding_imagen": model.config.image_embedding_dim,
        "objetivos_auxiliares_proxy": model.config.auxiliary_dim,
    })
display(pd.DataFrame(architecture_rows))
display(pd.DataFrame({
    "parametro_augmentation": [
        "magnitud M", "probabilidad P", "FWHM blur base (mm)",
        "FWHM ruido correlacionado (mm)", "rotacion maxima (grados)",
    ],
    "valor": [
        EXPERIMENT.train.augmentation.magnitude,
        EXPERIMENT.train.augmentation.probability,
        EXPERIMENT.train.augmentation.blur_base_fwhm_mm,
        EXPERIMENT.train.augmentation.correlated_noise_fwhm_mm,
        EXPERIMENT.train.augmentation.rotation_degrees,
    ],
}))
"""
    ),
    markdown(
        """
## 2. Auditoría visual de self-normalization, slab y lateralidad

El panel no selecciona casos por etiqueta; toma una muestra reproducible del manifiesto. La fila izquierda conserva la orientación y la derecha canoniza el putamen de menor captación al lado derecho del paciente.
"""
    ),
    code(
        """
sample = prepared.cohort.sample(min(6, len(prepared.cohort)), random_state=EXPERIMENT.train.seed)
fig, axes = plt.subplots(len(sample), 2, figsize=(8, 3 * len(sample)))
axes = np.atleast_2d(axes)
for row_index, row in enumerate(sample.itertuples(index=False)):
    with np.load(row.base_cache_path, allow_pickle=False) as payload:
        native = payload["slab_native"].astype(np.float32)
        canonical = payload["slab_canonical"].astype(np.float32)
    for column, (image, title) in enumerate(((native, "nativa"), (canonical, "canonizada"))):
        axes[row_index, column].imshow(image, cmap="inferno")
        axes[row_index, column].set_title(f"{row.uid} · {title}")
        axes[row_index, column].axis("off")
plt.tight_layout()
plt.show()
"""
    ),
    markdown(
        """
## 3. Búsqueda compacta: tres folds

Cada trial completa sus tres folds balanceados por paciente. Optuna, los checkpoints por época, los bancos de plantillas y las 930 variables quedan persistidos; tras un corte se retoma el trial/época pendiente. La rama slab recibe el mayor presupuesto; 2.5D y 3D actúan como miembros diferentes del ensemble.
"""
    ),
    code(
        """
with RunLock(RUN_DIR / "search.lock"):
    search_result = run_search_stage(prepared, EXPERIMENT, device=DEVICE)
display(search_result.summary)
"""
    ),
    markdown(
        """
## 4. Evaluación final: cinco folds congelados

Se reentrenan dos o tres finalistas durante el número de épocas decidido en la búsqueda. No se usa el fold externo para early stopping. La selección final emplea log loss OOF calibrado de forma cruzada.
"""
    ),
    code(
        """
with RunLock(RUN_DIR / "final.lock"):
    final_result = run_final_stage(prepared, EXPERIMENT, device=DEVICE)
display(final_result.metrics)
display(Markdown(
    f"**Modelo primario:** `{final_result.deployment_manifest['primary_candidate_id']}` · "
    f"**modelos por fold:** `{final_result.deployment_manifest['n_fold_models']}`"
))
"""
    ),
    markdown(
        """
## 5. Artefactos y lectura honesta

- `outputs/private_eda/node07_cache/`: derivados físicos, plantillas y features reanudables.
- `outputs/private_eda/node07_runs/<run_id>/search/`: SQLite de Optuna, trials y finalistas.
- `outputs/private_eda/node07_runs/<run_id>/final/`: OOF, métricas, calibración y cinco checkpoints del modelo primario.

La arquitectura sólo se promueve sobre el nodo 06 si mejora log loss OOF, calibración, estabilidad entre folds y peor familia de adquisición. La separación visual de embeddings no es criterio de selección.
"""
    ),
]

DESTINATION.parent.mkdir(parents=True, exist_ok=True)
nbf.write(notebook, DESTINATION)
print(DESTINATION)
