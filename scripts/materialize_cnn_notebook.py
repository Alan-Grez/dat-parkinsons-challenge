from __future__ import annotations

import textwrap
from pathlib import Path

import nbformat as nbf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "06_cnn_25d_3d_fusion.ipynb"


def markdown(source: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(textwrap.dedent(source).strip())


def code(source: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(textwrap.dedent(source).strip())


def main() -> None:
    cells = [
        markdown(
            """
            # 06 - CNN 2.5D/3D, radiomics y SBR condicionado

            Este nodo consume los crops y la tabla del nodo 04 en modo de solo lectura. Compara seis
            variantes: imagen, imagen+radiomics independientes del fondo e imagen+radiomics+SBR con
            compuerta de validez, para backbones 2.5D y 3D.

            La familia de adquisicion se usa unicamente para construir folds congelados y auditar
            generalizacion; nunca entra al modelo. Los crops v3 usan una plantilla global de la cohorte,
            por lo que sus metricas OOF son desarrollo de competencia, no validacion clinica externa.
            """
        ),
        code(
            """
            from __future__ import annotations

            import json
            import os
            import sys
            from contextlib import nullcontext
            from dataclasses import replace
            from pathlib import Path

            import matplotlib.pyplot as plt
            import numpy as np
            import pandas as pd
            import seaborn as sns
            import torch
            from IPython.display import Markdown, display

            PROJECT_ROOT = Path.cwd()
            if PROJECT_ROOT.name.lower() == 'notebooks':
                PROJECT_ROOT = PROJECT_ROOT.parent
            if str(PROJECT_ROOT) not in sys.path:
                sys.path.insert(0, str(PROJECT_ROOT))

            from modeling.cnn import ExperimentConfig
            from modeling.cnn.io import RunLock
            from modeling.cnn.pipeline import (
                prepare_experiment,
                run_final_stage,
                run_search_stage,
                validate_path_component,
            )
            from modeling.cnn.xai_runner import run_xai_audit

            RUN_ID = os.environ.get('DAT_CNN_RUN_ID', 'cnn_compact_v1')
            NODE4_PROFILE = os.environ.get('DAT_CNN_NODE4_PROFILE', 'v3')
            validate_path_component(RUN_ID, 'DAT_CNN_RUN_ID')
            validate_path_component(NODE4_PROFILE, 'DAT_CNN_NODE4_PROFILE')
            base = ExperimentConfig(run_id=RUN_ID)
            experiment = replace(
                base,
                data=replace(base.data, node4_profile=NODE4_PROFILE),
                train=replace(
                    base.train,
                    max_epochs_search=int(os.environ.get('DAT_CNN_EPOCHS_SEARCH', '12')),
                ),
                search=replace(
                    base.search,
                    trials_image=int(os.environ.get('DAT_CNN_TRIALS_IMAGE', '20')),
                    trials_fusion=int(os.environ.get('DAT_CNN_TRIALS_FUSION', '10')),
                    trials_sbr=int(os.environ.get('DAT_CNN_TRIALS_SBR', '8')),
                    finalists=int(os.environ.get('DAT_CNN_FINALISTS', '3')),
                ),
            )
            DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
            RUN_DIR = PROJECT_ROOT / 'outputs' / 'private_eda' / 'cnn_runs' / RUN_ID
            LOCK_PATH = RUN_DIR / '.run.lock'

            def stage_lock():
                if os.environ.get('DAT_CNN_LOCK_HELD') == '1':
                    return nullcontext()
                return RunLock(LOCK_PATH)

            sns.set_theme(style='whitegrid')
            display(Markdown(
                f'**Run:** `{RUN_ID}` - **device:** `{DEVICE}` - '
                f'**hash:** `{experiment.config_hash[:12]}`'
            ))
            """
        ),
        markdown(
            """
            ## Preparacion reanudable

            Normaliza robustamente cada crop sin usar el fondo, genera radiomics core independientes del
            fondo, congela folds agrupados 3/5 y persiste un cache compartido. Si se interrumpe, omite
            automaticamente los UIDs ya confirmados.
            """
        ),
        code(
            """
            with stage_lock():
                prepared = prepare_experiment(experiment, project_root=PROJECT_ROOT)
            display(pd.DataFrame({
                'indicador': ['casos', 'familias', 'fondo valido', 'features core'],
                'valor': [
                    len(prepared.cohort),
                    prepared.cohort['acquisition_family'].nunique(),
                    prepared.cohort['background_qc_valid'].mean(),
                    sum(column.startswith('core_') for column in prepared.cohort.columns),
                ],
            }))
            display(pd.read_csv(prepared.run_dir / 'config' / 'search_fold_summary.csv'))
            display(pd.read_csv(prepared.run_dir / 'config' / 'final_fold_summary.csv'))
            """
        ),
        markdown(
            """
            ## Busqueda compacta con tres folds

            Optuna busca primero el backbone de imagen y despues las ramas de fusion. Usa pruning,
            checkpoints por epoca y una base SQLite con heartbeat; un trial interrumpido reusa su
            checkpoint. La GPU ejecuta un solo trial a la vez y los DataLoaders paralelizan la lectura.
            """
        ),
        code(
            """
            with stage_lock():
                search_result = run_search_stage(prepared, experiment, device=DEVICE)
            display(search_result.summary.style.format({'search_log_loss': '{:.4f}'}))
            """
        ),
        markdown(
            """
            ## Evaluacion final con cinco folds congelados

            Los dos o tres finalistas se entrenan un numero fijo de epocas derivado de la busqueda. El
            fold externo no elige la epoca. Se guardan cinco modelos por configuracion, OOF completo,
            calibracion cruzada y un manifiesto del ensemble primario.
            """
        ),
        code(
            """
            with stage_lock():
                final_result = run_final_stage(prepared, experiment, device=DEVICE)
            display(final_result.metrics.style.format(precision=4))

            figure, axes = plt.subplots(1, 2, figsize=(13, 4))
            plot_metrics = final_result.metrics.copy()
            sns.barplot(
                data=plot_metrics,
                x='candidate_id', y='cross_calibrated_log_loss',
                hue='architecture', ax=axes[0],
            )
            axes[0].tick_params(axis='x', rotation=35)
            axes[0].set_title('Log loss OOF cross-calibrado')
            sns.histplot(
                data=final_result.oof_predictions,
                x='probability_cross_calibrated', hue='is_pathologic',
                bins=25, stat='density', common_norm=False, ax=axes[1],
            )
            axes[1].set_title('Probabilidades OOF')
            plt.tight_layout()
            """
        ),
        markdown(
            """
            ## Explicabilidad 3D cuantitativa

            Se audita el mejor finalista 3D con Grad-CAM, oclusion por bloques y contrafactuales. Cuando
            se contradicen, la decision de fidelidad prioriza oclusion y contrafactuales; la contradiccion
            queda marcada en vez de ocultarse.
            """
        ),
        code(
            """
            with stage_lock():
                xai_results = run_xai_audit(
                    prepared,
                    n_cases=int(os.environ.get('DAT_CNN_XAI_CASES', '24')),
                    device=DEVICE,
                )
            display(xai_results)
            display(
                xai_results[
                    ['evidence_priority', 'contradiction', 'counterfactual_invariance_failed']
                ].value_counts(dropna=False).rename('n').to_frame()
            )

            if not xai_results.empty:
                example = xai_results.iloc[0]
                patient = prepared.cohort.set_index('uid').loc[str(example['uid'])]
                with np.load(Path(patient['cache_path']), allow_pickle=False) as source:
                    volume = source['volume'].astype(np.float32)
                with np.load(Path(example['artifact_path']), allow_pickle=False) as maps:
                    gradcam_map = maps['gradcam'][0, 0].astype(np.float32)
                    occlusion_map = maps['occlusion_importance'][0, 0].astype(np.float32)
                    target_mask = maps['target_mask'][0, 0].astype(bool)
                image_mip = volume.max(axis=0)
                cam_mip = gradcam_map.max(axis=0)
                occlusion_mip = occlusion_map.max(axis=0)
                mask_mip = target_mask.max(axis=0)
                figure, axes = plt.subplots(1, 4, figsize=(15, 4))
                axes[0].imshow(image_mip, cmap='inferno')
                axes[0].set_title(f"Imagen MIP · {example['uid']}")
                axes[1].imshow(image_mip, cmap='gray')
                axes[1].imshow(cam_mip, cmap='jet', alpha=0.55)
                axes[1].set_title('Grad-CAM 3D')
                axes[2].imshow(image_mip, cmap='gray')
                axes[2].imshow(occlusion_mip, cmap='magma', alpha=0.60)
                axes[2].set_title('Oclusion 3D')
                axes[3].imshow(image_mip, cmap='gray')
                axes[3].contour(mask_mip, levels=[0.5], colors='cyan', linewidths=1.2)
                axes[3].set_title('Mascara target')
                for axis in axes:
                    axis.axis('off')
                plt.tight_layout()
            """
        ),
        markdown(
            """
            ## Artefactos principales

            - `search/finalists.json`: configuraciones finalistas.
            - `final/finalist_cv5_metrics.csv`: comparacion OOF de cinco folds.
            - `final/primary_ensemble_manifest.json`: cinco checkpoints y regla de promedio.
            - `final/xai/`: mapas y mediciones de Grad-CAM, oclusion y contrafactuales.
            """
        ),
    ]
    notebook = nbf.v4.new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.12"},
        },
    )
    nbf.validate(notebook)
    for cell_index, cell in enumerate(notebook.cells):
        if cell.cell_type == "code":
            compile(cell.source, f"{NOTEBOOK_PATH.name}:cell-{cell_index}", "exec")
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(notebook, NOTEBOOK_PATH)
    print(NOTEBOOK_PATH)


if __name__ == "__main__":
    main()
