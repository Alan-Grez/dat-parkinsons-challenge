"""Materialize the full-cohort 3D EDA notebooks without embedded outputs."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import nbformat as nbf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = PROJECT_ROOT / "notebooks"


def md(source: str):
    return nbf.v4.new_markdown_cell(dedent(source).strip())


def code(source: str):
    return nbf.v4.new_code_cell(dedent(source).strip())


def notebook(cells: list) -> nbf.NotebookNode:
    document = nbf.v4.new_notebook(cells=cells)
    document.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    document.metadata["language_info"] = {"name": "python", "version": "3.12"}
    return document


NOTEBOOK_03 = notebook(
    [
        md(
            """
            # 03 · EDA full-cohort y control de calidad físico

            Audita **todos los NIfTI de entrenamiento** sin remuestrear ni modificar las fuentes. El
            resultado es un manifiesto por examen, familias de adquisición reproducibles y scores de
            revisión técnica. La geometría se trata como QC/dominio, no como señal clínica.

            > Privacidad: ejecutar sólo localmente. El notebook se versiona sin outputs. Los artefactos
            > de `outputs/private_eda/` contienen derivados privados y no deben compartirse.
            """
        ),
        md(
            """
            ## 1. Configuración y contrato

            `MAX_SCANS=None` procesa la cohorte completa. La orientación RAS sólo homogeneiza ejes;
            todavía no constituye registro anatómico. `gradient_energy` se calcula por milímetro y
            dentro de voxeles positivos, corrigiendo la versión inicial por-voxel.
            """
        ),
        code(
            '''
            from __future__ import annotations

            import json
            import tempfile
            import zipfile
            from pathlib import Path

            import ipywidgets as widgets
            import matplotlib.pyplot as plt
            import nibabel as nib
            import numpy as np
            import pandas as pd
            import seaborn as sns
            from IPython.display import Markdown, display
            from scipy import ndimage
            from tqdm.auto import tqdm

            PROJECT_ROOT = Path.cwd()
            if PROJECT_ROOT.name.lower() == 'notebooks':
                PROJECT_ROOT = PROJECT_ROOT.parent

            NIFTI_ARCHIVE = PROJECT_ROOT / 'data' / 'raw' / 'niftis.zip'
            LABELS_PATH = PROJECT_ROOT / 'data' / 'raw' / 'train_labels.csv'
            PRIVATE_OUTPUT_DIR = PROJECT_ROOT / 'outputs' / 'private_eda'

            MAX_SCANS: int | None = None
            RANDOM_SEED = 20260821
            MIN_PROTOCOL_FAMILY_SIZE = 5
            SAVE_PRIVATE_OUTPUTS = True

            for required_path in [NIFTI_ARCHIVE, LABELS_PATH]:
                if not required_path.exists():
                    raise FileNotFoundError(required_path)

            sns.set_theme(style='whitegrid')
            display(Markdown(
                f'**Modo:** cohorte completa (`MAX_SCANS={MAX_SCANS}`) · '
                f'**salida privada:** `{PRIVATE_OUTPUT_DIR}`'
            ))
            '''
        ),
        code(
            '''
            def uid_from_member(member_name: str) -> str:
                name = Path(member_name).name
                return name[:-7] if name.lower().endswith('.nii.gz') else Path(name).stem


            def list_nifti_members(archive_path: Path) -> list[str]:
                with zipfile.ZipFile(archive_path) as archive:
                    return sorted(
                        name for name in archive.namelist()
                        if name.lower().endswith(('.nii', '.nii.gz')) and not name.endswith('/')
                    )


            def load_canonical_volume(
                member_name: str,
            ) -> tuple[np.ndarray, nib.Nifti1Image, dict[str, object]]:
                with tempfile.TemporaryDirectory(prefix='dat_qc_') as temporary_directory:
                    with zipfile.ZipFile(NIFTI_ARCHIVE) as archive:
                        extracted_path = Path(archive.extract(member_name, temporary_directory))
                    image = nib.load(extracted_path)
                    original_metadata = {
                        'original_orientation': ''.join(nib.aff2axcodes(image.affine)),
                        'original_affine_determinant': float(np.linalg.det(image.affine[:3, :3])),
                        'original_qform_code': int(image.header['qform_code']),
                        'original_sform_code': int(image.header['sform_code']),
                        'original_dtype': str(image.get_data_dtype()),
                    }
                    canonical = nib.as_closest_canonical(image)
                    volume = np.asarray(canonical.dataobj, dtype=np.float32).squeeze()
                    if volume.ndim != 3:
                        raise ValueError(f'{member_name}: se esperaba 3D y se obtuvo {volume.shape}.')
                    detached = nib.Nifti1Image(volume, canonical.affine, canonical.header.copy())
                return volume, detached, original_metadata


            members = list_nifti_members(NIFTI_ARCHIVE)
            if MAX_SCANS is not None and MAX_SCANS < len(members):
                rng = np.random.default_rng(RANDOM_SEED)
                indices = np.sort(rng.choice(len(members), size=MAX_SCANS, replace=False))
                members = [members[index] for index in indices]

            member_by_uid = {uid_from_member(member): member for member in members}
            display(Markdown(f'**NIfTI seleccionados:** `{len(members):,}`'))
            '''
        ),
        md(
            """
            ## 2. Manifiesto físico, intensidad y QC

            La asimetría se calcula sobre captación por encima de la mediana y no se asigna lateralidad
            anatómica antes del registro. `gradient_energy` es la magnitud media del gradiente físico
            (intensidad normalizada por milímetro) dentro del soporte positivo.
            """
        ),
        code(
            '''
            def safe_entropy(values: np.ndarray, bins: int = 64) -> float:
                if values.size < 2 or float(values.max()) <= float(values.min()):
                    return 0.0
                counts, _ = np.histogram(values, bins=bins)
                probabilities = counts[counts > 0] / counts.sum()
                return float(-(probabilities * np.log2(probabilities)).sum())


            def adjacent_slice_correlation(volume01: np.ndarray) -> float:
                correlations: list[float] = []
                for index in range(volume01.shape[2] - 1):
                    first = volume01[:, :, index].ravel()
                    second = volume01[:, :, index + 1].ravel()
                    if first.std() > 1e-6 and second.std() > 1e-6:
                        correlations.append(float(np.corrcoef(first, second)[0, 1]))
                return float(np.nanmedian(correlations)) if correlations else float('nan')


            def scan_record(member_name: str) -> dict[str, object]:
                volume, image, original_metadata = load_canonical_volume(member_name)
                finite_mask = np.isfinite(volume)
                finite = volume[finite_mask]
                if finite.size == 0:
                    raise ValueError(f'{member_name}: no contiene voxels finitos.')

                positive = finite[finite > 0]
                reference = positive if positive.size else finite
                p01, p05, p50, p90, p95, p995 = np.percentile(
                    reference, [1, 5, 50, 90, 95, 99.5]
                )
                scale = max(float(p995 - p01), np.finfo(np.float32).eps)
                normalized = np.clip(
                    (np.nan_to_num(volume, nan=float(p01)) - p01) / scale, 0, 1
                )
                spacing = tuple(float(value) for value in image.header.get_zooms()[:3])
                shape = tuple(int(value) for value in volume.shape)
                fov = tuple(shape[index] * spacing[index] for index in range(3))

                foreground = finite_mask & (volume > p50)
                high_uptake = finite_mask & (volume > p90)
                labels_cc, component_count = ndimage.label(high_uptake)
                component_sizes = (
                    np.bincount(labels_cc.ravel())[1:] if component_count else np.array([])
                )
                largest_fraction = (
                    float(component_sizes.max() / max(high_uptake.sum(), 1))
                    if component_sizes.size else 0.0
                )

                weights = np.where(foreground, normalized, 0.0)
                centroid_voxel = np.asarray(ndimage.center_of_mass(weights), dtype=float)
                if not np.all(np.isfinite(centroid_voxel)):
                    centroid_voxel = (np.asarray(volume.shape, dtype=float) - 1) / 2
                centroid_mm = nib.affines.apply_affine(image.affine, centroid_voxel)

                midpoint = volume.shape[0] // 2
                lower_x_signal = float(weights[:midpoint].sum())
                upper_x_signal = float(weights[-midpoint:].sum())
                denominator = max((lower_x_signal + upper_x_signal) / 2, 1e-8)
                asymmetry_abs = abs(upper_x_signal - lower_x_signal) / denominator
                asymmetry_signed = (upper_x_signal - lower_x_signal) / denominator

                gradients = np.gradient(normalized, *spacing, edge_order=1)
                gradient_magnitude = np.sqrt(sum(component ** 2 for component in gradients))
                gradient_support = finite_mask & (volume > p05)
                gradient_energy = float(
                    gradient_magnitude[gradient_support].mean()
                    if gradient_support.any() else gradient_magnitude.mean()
                )

                return {
                    'uid': uid_from_member(member_name),
                    'member': member_name,
                    'shape_x': shape[0], 'shape_y': shape[1], 'shape_z': shape[2],
                    'spacing_x_mm': spacing[0], 'spacing_y_mm': spacing[1],
                    'spacing_z_mm': spacing[2],
                    'spacing_anisotropy': max(spacing) / max(min(spacing), 1e-8),
                    'fov_x_mm': fov[0], 'fov_y_mm': fov[1], 'fov_z_mm': fov[2],
                    'fov_volume_l': float(np.prod(fov) / 1_000_000),
                    'voxel_volume_mm3': float(np.prod(spacing)),
                    **original_metadata,
                    'canonical_orientation': ''.join(nib.aff2axcodes(image.affine)),
                    'finite_fraction': float(finite_mask.mean()),
                    'zero_fraction': float(np.mean(finite == 0)),
                    'positive_fraction': float(np.mean(finite > 0)),
                    'intensity_min': float(finite.min()), 'intensity_max': float(finite.max()),
                    'p01': float(p01), 'p05': float(p05), 'p50': float(p50),
                    'p90': float(p90), 'p95': float(p95), 'p995': float(p995),
                    'positive_entropy_bits': safe_entropy(reference),
                    'foreground_volume_ml_proxy': float(
                        foreground.sum() * np.prod(spacing) / 1000
                    ),
                    'high_uptake_components_p90': int(component_count),
                    'largest_component_fraction_p90': largest_fraction,
                    'centroid_x_mm': float(centroid_mm[0]),
                    'centroid_y_mm': float(centroid_mm[1]),
                    'centroid_z_mm': float(centroid_mm[2]),
                    'lr_uptake_ai_proxy': float(asymmetry_abs),
                    'lr_uptake_signed_proxy': float(asymmetry_signed),
                    'lr_global_ai': float(asymmetry_abs),
                    'gradient_energy': gradient_energy,
                    'gradient_energy_per_mm': gradient_energy,
                    'slice_corr_z': adjacent_slice_correlation(normalized),
                }


            records: list[dict[str, object]] = []
            failures: list[dict[str, str]] = []
            for member in tqdm(members, desc='Auditando NIfTI'):
                try:
                    records.append(scan_record(member))
                except Exception as error:
                    failures.append({
                        'member': member,
                        'error': f'{type(error).__name__}: {error}',
                    })

            manifest = pd.DataFrame(records)
            labels = pd.read_csv(LABELS_PATH, dtype={'uid': 'string'})
            labels['uid'] = labels['uid'].astype('string')
            manifest['uid'] = manifest['uid'].astype('string')
            manifest = manifest.merge(
                labels[['uid', 'is_pathologic']],
                on='uid', how='left', validate='one_to_one',
            )
            display(Markdown(
                f'**Procesados:** `{len(manifest):,}` · **fallos:** `{len(failures):,}`'
            ))
            display(manifest.head().style.hide(axis='index'))
            '''
        ),
        md(
            """
            ## 3. Familias de adquisición y scores de revisión

            La familia se define sin etiqueta por `(shape, spacing)` redondeado. Se calculan por separado
            rareza geométrica global y desviación QC dentro de protocolo; así un protocolo válido pero
            poco frecuente no se confunde automáticamente con una imagen defectuosa.
            """
        ),
        code(
            '''
            def robust_z(frame: pd.DataFrame) -> pd.DataFrame:
                numeric = frame.replace([np.inf, -np.inf], np.nan).astype(float)
                numeric = numeric.fillna(numeric.median())
                median = numeric.median()
                mad = (numeric - median).abs().median().replace(0, np.nan)
                return (
                    (numeric - median) / (1.4826 * mad)
                ).replace([np.inf, -np.inf], np.nan).fillna(0)


            manifest['acquisition_signature'] = manifest.apply(
                lambda row: (
                    f"{int(row.shape_x)}x{int(row.shape_y)}x{int(row.shape_z)}|"
                    f"{row.spacing_x_mm:.3f},{row.spacing_y_mm:.3f},{row.spacing_z_mm:.3f}"
                ),
                axis=1,
            )
            signature_order = sorted(manifest['acquisition_signature'].unique())
            family_by_signature = {
                signature: f'AF{index + 1:03d}'
                for index, signature in enumerate(signature_order)
            }
            manifest['acquisition_family'] = manifest['acquisition_signature'].map(
                family_by_signature
            )
            family_sizes = manifest['acquisition_family'].value_counts()
            manifest['acquisition_family_size'] = manifest['acquisition_family'].map(
                family_sizes
            ).astype(int)
            manifest['is_rare_acquisition_family'] = (
                manifest['acquisition_family_size'] < MIN_PROTOCOL_FAMILY_SIZE
            )

            geometry_frame = pd.DataFrame({
                'log_voxel_volume': np.log1p(manifest['voxel_volume_mm3']),
                'log_fov_volume': np.log1p(manifest['fov_volume_l']),
                'spacing_anisotropy': manifest['spacing_anisotropy'],
                'grid_aspect_xy': manifest['shape_x'] / manifest['shape_y'].clip(lower=1),
                'grid_aspect_zx': manifest['shape_z'] / manifest['shape_x'].clip(lower=1),
            })
            geometry_z = robust_z(geometry_frame)
            manifest['global_geometry_outlier_score'] = np.sqrt(
                (geometry_z ** 2).mean(axis=1)
            )

            qc_frame = pd.DataFrame({
                'zero_fraction': manifest['zero_fraction'],
                'entropy': manifest['positive_entropy_bits'],
                'log_components': np.log1p(manifest['high_uptake_components_p90']),
                'largest_component_fraction': manifest['largest_component_fraction_p90'],
                'log_gradient_per_mm': np.log1p(manifest['gradient_energy_per_mm']),
                'slice_incoherence': 1 - manifest['slice_corr_z'],
            })
            global_qc_z = robust_z(qc_frame)
            within_qc_z = global_qc_z.copy()
            for family, indices in manifest.groupby('acquisition_family').groups.items():
                if len(indices) >= MIN_PROTOCOL_FAMILY_SIZE:
                    within_qc_z.loc[indices] = robust_z(qc_frame.loc[indices])

            manifest['within_protocol_qc_score'] = np.sqrt((within_qc_z ** 2).mean(axis=1))
            manifest['technical_outlier_score'] = np.sqrt(
                (
                    manifest['global_geometry_outlier_score'] ** 2
                    + manifest['within_protocol_qc_score'] ** 2
                ) / 2
            )

            protocol_summary = (
                manifest.groupby(['acquisition_family', 'acquisition_signature'], as_index=False)
                .agg(
                    n=('uid', 'size'),
                    n_labeled=('is_pathologic', 'count'),
                    pathologic_rate=('is_pathologic', 'mean'),
                    median_qc_score=('within_protocol_qc_score', 'median'),
                    median_gradient_per_mm=('gradient_energy_per_mm', 'median'),
                    median_slice_corr=('slice_corr_z', 'median'),
                )
                .sort_values('n', ascending=False)
            )

            review_columns = [
                'uid', 'is_pathologic', 'acquisition_family', 'acquisition_family_size',
                'global_geometry_outlier_score', 'within_protocol_qc_score',
                'technical_outlier_score', 'zero_fraction', 'gradient_energy_per_mm',
                'slice_corr_z',
            ]
            display(Markdown('### Familias de adquisición'))
            display(protocol_summary.head(25).style.hide(axis='index'))
            display(Markdown('### Casos priorizados para revisión técnica'))
            display(
                manifest.nlargest(20, 'technical_outlier_score')[review_columns]
                .style.format({
                    'global_geometry_outlier_score': '{:.2f}',
                    'within_protocol_qc_score': '{:.2f}',
                    'technical_outlier_score': '{:.2f}',
                })
                .hide(axis='index')
            )
            '''
        ),
        code(
            '''
            integrity = pd.DataFrame({
                'indicador': [
                    'Exámenes auditados', 'UID duplicados', 'Sin etiqueta',
                    'Fracción finita < 1', 'Spacing no positivo',
                    'Affine casi singular', 'Familias de adquisición', 'Familias raras',
                ],
                'valor': [
                    len(manifest), int(manifest['uid'].duplicated().sum()),
                    int(manifest['is_pathologic'].isna().sum()),
                    int((manifest['finite_fraction'] < 1).sum()),
                    int((manifest[['spacing_x_mm', 'spacing_y_mm', 'spacing_z_mm']] <= 0)
                        .any(axis=1).sum()),
                    int((manifest['original_affine_determinant'].abs() < 1e-8).sum()),
                    int(manifest['acquisition_family'].nunique()),
                    int(manifest.loc[manifest['is_rare_acquisition_family'],
                                     'acquisition_family'].nunique()),
                ],
            })
            display(integrity.style.hide(axis='index'))

            figure, axes = plt.subplots(2, 2, figsize=(14, 10))
            sns.histplot(
                data=manifest, x='spacing_x_mm', hue='is_pathologic',
                element='step', ax=axes[0, 0],
            )
            axes[0, 0].set_title('Spacing X por clase')
            sns.scatterplot(
                data=manifest, x='fov_x_mm', y='fov_z_mm',
                hue='acquisition_family', legend=False, ax=axes[0, 1],
            )
            axes[0, 1].set_title('Cobertura física por familia de adquisición')
            sns.boxplot(
                data=manifest, x='is_pathologic', y='lr_uptake_ai_proxy', ax=axes[1, 0]
            )
            axes[1, 0].set_title('Asimetría de captación global (proxy)')
            sns.scatterplot(
                data=manifest, x='gradient_energy_per_mm', y='slice_corr_z',
                hue='is_pathologic', ax=axes[1, 1],
            )
            axes[1, 1].set_title('Gradiente físico y coherencia entre cortes')
            plt.tight_layout()
            plt.show()
            '''
        ),
        md(
            """
            ## 4. Revisión visual triplanar y MIP

            Las MIP toman el máximo por eje; no son promedios. El visor abre un NIfTI a la vez y no
            persiste imágenes.
            """
        ),
        code(
            '''
            def plot_scan_qc(uid: str) -> None:
                member = member_by_uid[str(uid)]
                volume, image, _ = load_canonical_volume(member)
                positive = volume[np.isfinite(volume) & (volume > 0)]
                values = positive if positive.size else volume[np.isfinite(volume)]
                vmin, vmax = np.percentile(values, [1, 99.5])
                display_volume = np.clip(volume, vmin, vmax)
                weights = np.where(volume > np.percentile(values, 50), display_volume - vmin, 0)
                center = np.asarray(ndimage.center_of_mass(weights), dtype=float)
                if not np.all(np.isfinite(center)):
                    center = (np.asarray(volume.shape) - 1) / 2
                x, y, z = np.clip(
                    np.rint(center).astype(int), 0, np.asarray(volume.shape) - 1
                )
                views = [
                    ('Sagital', display_volume[x, :, :]),
                    ('Coronal', display_volume[:, y, :]),
                    ('Axial', display_volume[:, :, z]),
                    ('MIP X', display_volume.max(axis=0)),
                    ('MIP Y', display_volume.max(axis=1)),
                    ('MIP Z', display_volume.max(axis=2)),
                ]
                figure, axes = plt.subplots(2, 3, figsize=(14, 9))
                for axis, (title, plane) in zip(axes.ravel(), views):
                    axis.imshow(np.rot90(plane), cmap='hot', vmin=vmin, vmax=vmax)
                    axis.set_title(title)
                    axis.axis('off')
                spacing = tuple(round(float(v), 3) for v in image.header.get_zooms()[:3])
                figure.suptitle(
                    f'{uid} · shape={volume.shape} · spacing={spacing} mm', fontsize=14
                )
                plt.tight_layout()
                plt.show()


            default_uid = str(manifest.nlargest(1, 'technical_outlier_score').iloc[0]['uid'])
            uid_selector = widgets.Dropdown(
                options=sorted(member_by_uid), value=default_uid, description='UID:',
                layout=widgets.Layout(width='420px'),
                style={'description_width': '60px'},
            )
            viewer_output = widgets.interactive_output(plot_scan_qc, {'uid': uid_selector})
            display(widgets.VBox([uid_selector, viewer_output]))
            '''
        ),
        md(
            """
            ## 5. Persistencia y linaje

            `image_qc_manifest.csv` conserva todos los casos, incluidos los raros o dudosos. Ningún
            score elimina exámenes. `acquisition_family_summary.csv` alimenta la validación agrupada del
            notebook 05.
            """
        ),
        code(
            '''
            if SAVE_PRIVATE_OUTPUTS:
                PRIVATE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
                manifest.to_csv(PRIVATE_OUTPUT_DIR / 'image_qc_manifest.csv', index=False)
                pd.DataFrame(failures).to_csv(
                    PRIVATE_OUTPUT_DIR / 'image_qc_failures.csv', index=False
                )
                protocol_summary.to_csv(
                    PRIVATE_OUTPUT_DIR / 'acquisition_family_summary.csv', index=False
                )
                (PRIVATE_OUTPUT_DIR / 'image_qc_config.json').write_text(
                    json.dumps({
                        'archive_name': NIFTI_ARCHIVE.name,
                        'max_scans': MAX_SCANS,
                        'random_seed': RANDOM_SEED,
                        'canonical_orientation': 'RAS',
                        'gradient_contract': (
                            'mean physical gradient magnitude per mm inside voxels above p05'
                        ),
                        'acquisition_family_contract': 'exact shape plus spacing rounded to 0.001 mm',
                        'no_automatic_exclusions': True,
                    }, indent=2),
                    encoding='utf-8',
                )
                display(Markdown(f'Guardado localmente en `{PRIVATE_OUTPUT_DIR}`.'))
            else:
                display(Markdown('Persistencia desactivada.'))
            '''
        ),
    ]
)


NOTEBOOK_04 = notebook(
    [
        md(
            """
            # 04 · Registro full-cohort, biomarcadores y radiomics 3D

            Procesa **todos los exámenes etiquetados** mediante un flujo streaming y reanudable:
            cargar → orientar LPS → registrar → extraer features → actualizar mapas → liberar memoria.
            No se retienen los 1.362 volúmenes en RAM.

            Se prioriza un atlas validado si se configura `VALIDATED_MASK_PATH`. Sin atlas, se construye
            una máscara de consenso a partir de normales representativos; sus columnas se mantienen como
            `proxy` y no deben presentarse como biomarcadores clínicamente validados.
            """
        ),
        md(
            """
            ## 1. Configuración full-cohort

            Los artefactos nuevos viven en `outputs/private_eda/full_cohort/`, por lo que no se mezclan
            con los resultados del piloto anterior. Los checkpoints incluyen features, QC y acumuladores
            online de media/varianza por clase.
            """
        ),
        code(
            '''
            from __future__ import annotations

            import atexit
            import hashlib
            import json
            import math
            import tempfile
            import time
            import zipfile
            from pathlib import Path

            import matplotlib.pyplot as plt
            import numpy as np
            import pandas as pd
            import seaborn as sns
            import SimpleITK as sitk
            import torch
            import torch.nn.functional as torch_functional
            from IPython.display import Markdown, display
            from scipy import ndimage
            from skimage.measure import marching_cubes, mesh_surface_area
            from tqdm.auto import tqdm

            PROJECT_ROOT = Path.cwd()
            if PROJECT_ROOT.name.lower() == 'notebooks':
                PROJECT_ROOT = PROJECT_ROOT.parent

            NIFTI_ARCHIVE = PROJECT_ROOT / 'data' / 'raw' / 'niftis.zip'
            PRIVATE_OUTPUT_DIR = PROJECT_ROOT / 'outputs' / 'private_eda'
            RUN_PROFILE = 'v3'
            FULL_OUTPUT_DIR = PRIVATE_OUTPUT_DIR / f'full_cohort_{RUN_PROFILE}'
            MANIFEST_PATH = PRIVATE_OUTPUT_DIR / 'image_qc_manifest.csv'

            VALIDATED_MASK_PATH: Path | None = None
            ATLAS_LABELS = {
                'right_target': 1,
                'left_target': 2,
                'background': 3,
                # Opcionales: right_caudate, left_caudate, right_putamen, left_putamen.
            }
            MAX_SCANS: int | None = None
            REFERENCE_UID: str | None = None
            REGISTRATION_MODE = 'rigid'
            REGISTRATION_BACKEND = 'torch_cuda'
            ISOTROPIC_SPACING_MM = 2.5
            TORCH_REGISTRATION_STAGES = (
                (0.25, 28, 0.040),
                (0.50, 20, 0.025),
                (1.00, 12, 0.012),
            )
            TORCH_EARLY_STOPPING_PATIENCE = 7
            GPU_FALLBACK_TO_SITK = True
            SITK_METRIC_SAMPLING = 0.10
            SITK_ITERATIONS = 100
            MASK_TEMPLATE_SCANS = 48
            TARGET_BASE_PERCENTILE = 90.0
            TARGET_PERCENTILES = (88.0, 90.0, 92.0, 94.0)
            ACTIVE_BACKGROUND_SD = 2.0
            BACKGROUND_SCALE_FLOOR_FRACTION = 0.05
            MIN_BACKGROUND_VOXELS = 64
            MIN_BACKGROUND_SUPPORT_FRACTION = 0.20
            TEXTURE_LEVELS = 32
            CHECKPOINT_EVERY = 20
            RESUME = True
            RETRY_FAILURES = True
            SAVE_REGISTERED_CROPS = True
            CROP_MARGIN_MM = 20.0
            RANDOM_SEED = 20260821

            if REGISTRATION_BACKEND == 'torch_cuda' and not torch.cuda.is_available():
                display(Markdown(
                    '**CUDA no disponible:** se utilizarÃ¡ SimpleITK CPU como fallback.'
                ))
                RESOLVED_REGISTRATION_BACKEND = 'simpleitk_cpu'
            else:
                RESOLVED_REGISTRATION_BACKEND = REGISTRATION_BACKEND
            torch.manual_seed(RANDOM_SEED)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(RANDOM_SEED)
                torch.backends.cudnn.benchmark = True

            FEATURES_PATH = FULL_OUTPUT_DIR / 'dat_radiomics_features_full.csv'
            REGISTRATION_PATH = FULL_OUTPUT_DIR / 'registration_qc_full.csv'
            FAILURES_PATH = FULL_OUTPUT_DIR / 'registration_failures_full.csv'
            STATE_PATH = FULL_OUTPUT_DIR / 'streaming_state_full.npz'
            MAPS_PATH = FULL_OUTPUT_DIR / 'cohort_maps_full.npz'
            MASKS_PATH = FULL_OUTPUT_DIR / 'analysis_masks_full.npz'
            CONFIG_PATH = FULL_OUTPUT_DIR / 'registration_radiomics_full_config.json'
            CROPS_DIR = FULL_OUTPUT_DIR / 'registered_crops'

            if REGISTRATION_MODE not in {'rigid', 'affine'}:
                raise ValueError("REGISTRATION_MODE debe ser 'rigid' o 'affine'.")
            if REGISTRATION_BACKEND not in {'torch_cuda', 'simpleitk_cpu'}:
                raise ValueError(
                    "REGISTRATION_BACKEND debe ser 'torch_cuda' o 'simpleitk_cpu'."
                )
            for required_path in [NIFTI_ARCHIVE, MANIFEST_PATH]:
                if not required_path.exists():
                    raise FileNotFoundError(required_path)

            FULL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            if SAVE_REGISTERED_CROPS:
                CROPS_DIR.mkdir(parents=True, exist_ok=True)
            sns.set_theme(style='whitegrid')
            '''
        ),
        md(
            """
            ## 2. Cohorte y referencia congelable

            No se descarta el 5% técnicamente extremo. Todos los casos etiquetados se intentan procesar
            y cualquier problema queda como bandera QC. Si `REFERENCE_UID=None`, la referencia se elige
            determinísticamente entre normales de la familia de adquisición más frecuente; para una
            evaluación definitiva conviene fijar explícitamente ese UID o usar un template externo.
            """
        ),
        code(
            '''
            manifest = pd.read_csv(MANIFEST_PATH, dtype={'uid': 'string'})
            required_columns = {
                'uid', 'member', 'is_pathologic', 'acquisition_family',
                'technical_outlier_score', 'within_protocol_qc_score',
                'shape_x', 'shape_y', 'shape_z',
                'spacing_x_mm', 'spacing_y_mm', 'spacing_z_mm',
                'fov_x_mm', 'fov_y_mm', 'fov_z_mm',
            }
            missing_columns = sorted(required_columns - set(manifest.columns))
            if missing_columns:
                raise ValueError(
                    'Ejecuta nuevamente el notebook 03 full-cohort. Faltan: '
                    + ', '.join(missing_columns)
                )

            cohort = manifest.loc[manifest['is_pathologic'].notna()].copy()
            cohort['is_pathologic'] = cohort['is_pathologic'].astype(int)
            cohort = cohort.sort_values('uid').reset_index(drop=True)
            if MAX_SCANS is not None and MAX_SCANS < len(cohort):
                rng = np.random.default_rng(RANDOM_SEED)
                selected: list[int] = []
                per_class = max(1, MAX_SCANS // cohort['is_pathologic'].nunique())
                for _, group in cohort.groupby('is_pathologic'):
                    selected.extend(
                        rng.choice(group.index, size=min(per_class, len(group)), replace=False).tolist()
                    )
                remaining = cohort.index.difference(selected)
                if len(selected) < MAX_SCANS:
                    selected.extend(
                        rng.choice(
                            remaining,
                            size=min(MAX_SCANS - len(selected), len(remaining)),
                            replace=False,
                        ).tolist()
                    )
                cohort = cohort.loc[sorted(set(selected))].reset_index(drop=True)

            member_by_uid = dict(zip(manifest['uid'].astype(str), manifest['member'].astype(str)))
            largest_family = str(cohort['acquisition_family'].value_counts().idxmax())
            if REFERENCE_UID is None:
                reference_candidates = cohort.loc[
                    (cohort['is_pathologic'] == 0)
                    & (cohort['acquisition_family'].astype(str) == largest_family)
                ].copy()
                if reference_candidates.empty:
                    reference_candidates = cohort.loc[cohort['is_pathologic'] == 0].copy()
                if reference_candidates.empty:
                    reference_candidates = cohort.copy()
                geometry_columns = [
                    'spacing_x_mm', 'spacing_y_mm', 'spacing_z_mm',
                    'fov_x_mm', 'fov_y_mm', 'fov_z_mm',
                ]
                geometry = reference_candidates[geometry_columns].astype(float)
                scale = geometry.std(ddof=0).replace(0, 1)
                reference_candidates['geometry_distance'] = np.sqrt(
                    ((((geometry - geometry.median()) / scale) ** 2).mean(axis=1))
                )
                reference_candidates['reference_score'] = (
                    reference_candidates['geometry_distance']
                    + 0.20 * reference_candidates['within_protocol_qc_score'].fillna(0)
                )
                REFERENCE_UID = str(
                    reference_candidates.nsmallest(1, 'reference_score').iloc[0]['uid']
                )
            if REFERENCE_UID not in member_by_uid:
                raise ValueError(f'REFERENCE_UID={REFERENCE_UID!r} no existe en el manifiesto.')

            display(Markdown(
                f'**Cohorte:** `{len(cohort):,}` · **referencia:** `{REFERENCE_UID}` · '
                f'**familia principal:** `{largest_family}`'
            ))
            display(cohort.groupby('is_pathologic').size().rename('n').to_frame())
            '''
        ),
        md(
            """
            ## 3. Registro físico LPS e isotropía

            SimpleITK conserva origen, dirección y spacing y realiza la alineación geométrica inicial.
            La referencia se remuestrea a 2,5 mm isotrópicos; el ajuste rígido residual se optimiza en
            CUDA por correlación normalizada. Intensidades usan interpolación lineal; máscaras, vecino
            más cercano. Los casos no fiables tienen fallback automático a SimpleITK/CPU.
            """
        ),
        code(
            '''
            previous_archive = globals().get('_REGISTRATION_ARCHIVE')
            if isinstance(previous_archive, zipfile.ZipFile):
                previous_archive.close()
            previous_cache = globals().get('_REGISTRATION_CACHE')
            if isinstance(previous_cache, tempfile.TemporaryDirectory):
                previous_cache.cleanup()

            _REGISTRATION_CACHE = tempfile.TemporaryDirectory(
                prefix='dat_registration_cache_'
            )
            _REGISTRATION_CACHE_ROOT = Path(_REGISTRATION_CACHE.name)
            _REGISTRATION_ARCHIVE = zipfile.ZipFile(NIFTI_ARCHIVE)


            def cleanup_registration_cache() -> None:
                archive = globals().get('_REGISTRATION_ARCHIVE')
                if isinstance(archive, zipfile.ZipFile) and archive.fp is not None:
                    archive.close()
                cache = globals().get('_REGISTRATION_CACHE')
                if isinstance(cache, tempfile.TemporaryDirectory):
                    cache.cleanup()


            atexit.register(cleanup_registration_cache)


            def read_sitk_from_zip(
                member_name: str,
                pixel_type: int = sitk.sitkFloat32,
                keep_cached: bool = False,
            ) -> sitk.Image:
                extracted_path = _REGISTRATION_CACHE_ROOT / member_name
                if not extracted_path.exists():
                    _REGISTRATION_ARCHIVE.extract(
                        member_name, _REGISTRATION_CACHE_ROOT
                    )
                try:
                    image = sitk.ReadImage(str(extracted_path), pixel_type)
                    oriented = sitk.Image(sitk.DICOMOrient(image, 'LPS'))
                finally:
                    if not keep_cached:
                        extracted_path.unlink(missing_ok=True)
                return oriented


            def make_isotropic_reference(image: sitk.Image, spacing_mm: float) -> sitk.Image:
                old_size = np.asarray(image.GetSize(), dtype=float)
                old_spacing = np.asarray(image.GetSpacing(), dtype=float)
                new_spacing = np.repeat(float(spacing_mm), 3)
                new_size = np.maximum(1, np.rint(old_size * old_spacing / new_spacing)).astype(int)
                return sitk.Resample(
                    image,
                    [int(value) for value in new_size],
                    sitk.Transform(3, sitk.sitkIdentity),
                    sitk.sitkLinear,
                    image.GetOrigin(),
                    tuple(float(value) for value in new_spacing),
                    image.GetDirection(),
                    0.0,
                    sitk.sitkFloat32,
                )


            def rescale_for_registration(image: sitk.Image) -> sitk.Image:
                return sitk.RescaleIntensity(sitk.Cast(image, sitk.sitkFloat32), 0.0, 1.0)


            def normalized_correlation(first: np.ndarray, second: np.ndarray) -> float:
                mask = np.isfinite(first) & np.isfinite(second) & ((first > 0) | (second > 0))
                if mask.sum() < 10:
                    return float('nan')
                a, b = first[mask].astype(float), second[mask].astype(float)
                if a.std() <= 1e-8 or b.std() <= 1e-8:
                    return float('nan')
                return float(np.corrcoef(a, b)[0, 1])


            def configured_registration(
                fixed: sitk.Image, moving: sitk.Image, initial: sitk.Transform
            ) -> tuple[sitk.Transform, float]:
                method = sitk.ImageRegistrationMethod()
                method.SetMetricAsCorrelation()
                method.SetMetricSamplingStrategy(method.RANDOM)
                method.SetMetricSamplingPercentage(SITK_METRIC_SAMPLING, RANDOM_SEED)
                method.SetInterpolator(sitk.sitkLinear)
                method.SetOptimizerAsRegularStepGradientDescent(
                    learningRate=1.0,
                    minStep=1e-4,
                    numberOfIterations=SITK_ITERATIONS,
                    gradientMagnitudeTolerance=1e-8,
                )
                method.SetOptimizerScalesFromPhysicalShift()
                method.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
                method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])
                method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
                method.SetInitialTransform(initial, inPlace=False)
                transform = method.Execute(
                    rescale_for_registration(fixed), rescale_for_registration(moving)
                )
                return transform, float(method.GetMetricValue())


            def register_to_reference_sitk(
                fixed: sitk.Image, moving: sitk.Image, mode: str
            ) -> tuple[sitk.Image, sitk.Transform, dict[str, object]]:
                started_at = time.perf_counter()
                rigid_initial = sitk.CenteredTransformInitializer(
                    fixed,
                    moving,
                    sitk.Euler3DTransform(),
                    sitk.CenteredTransformInitializerFilter.GEOMETRY,
                )
                rigid, rigid_metric = configured_registration(fixed, moving, rigid_initial)
                final_transform: sitk.Transform = rigid
                final_metric = rigid_metric
                if mode == 'affine':
                    affine_initial = sitk.AffineTransform(3)
                    affine_initial.SetCenter(
                        tuple((np.asarray(fixed.GetSize()) - 1) * np.asarray(fixed.GetSpacing()) / 2)
                    )
                    final_transform, final_metric = configured_registration(
                        fixed, moving, sitk.CompositeTransform([rigid, affine_initial])
                    )

                before = sitk.Resample(
                    moving, fixed, sitk.Transform(3, sitk.sitkIdentity),
                    sitk.sitkLinear, 0.0, sitk.sitkFloat32,
                )
                after = sitk.Resample(
                    moving, fixed, final_transform, sitk.sitkLinear, 0.0, sitk.sitkFloat32
                )
                fixed_array = sitk.GetArrayViewFromImage(fixed)
                before_array = sitk.GetArrayViewFromImage(before)
                after_array = sitk.GetArrayViewFromImage(after)
                return after, final_transform, {
                    'metric_final_correlation_objective': final_metric,
                    'correlation_before': normalized_correlation(fixed_array, before_array),
                    'correlation_after': normalized_correlation(fixed_array, after_array),
                    'registration_backend': 'simpleitk_cpu',
                    'gpu_fallback_used': False,
                    'registration_seconds': time.perf_counter() - started_at,
                }


            def torch_rotation_matrix(angles: torch.Tensor) -> torch.Tensor:
                ax, ay, az = angles
                one = torch.ones((), device=angles.device)
                zero = torch.zeros((), device=angles.device)
                cx, sx = torch.cos(ax), torch.sin(ax)
                cy, sy = torch.cos(ay), torch.sin(ay)
                cz, sz = torch.cos(az), torch.sin(az)
                rotation_x = torch.stack([
                    one, zero, zero,
                    zero, cx, -sx,
                    zero, sx, cx,
                ]).reshape(3, 3)
                rotation_y = torch.stack([
                    cy, zero, sy,
                    zero, one, zero,
                    -sy, zero, cy,
                ]).reshape(3, 3)
                rotation_z = torch.stack([
                    cz, -sz, zero,
                    sz, cz, zero,
                    zero, zero, one,
                ]).reshape(3, 3)
                return rotation_z @ rotation_y @ rotation_x


            def torch_warp(
                volume: torch.Tensor, parameters: torch.Tensor
            ) -> torch.Tensor:
                theta = torch.cat([
                    torch_rotation_matrix(parameters[:3]),
                    parameters[3:, None],
                ], dim=1)[None]
                grid = torch_functional.affine_grid(
                    theta, volume.shape, align_corners=False
                )
                return torch_functional.grid_sample(
                    volume, grid, mode='bilinear', padding_mode='zeros',
                    align_corners=False,
                )


            def torch_ncc_loss(
                fixed: torch.Tensor, moving: torch.Tensor
            ) -> torch.Tensor:
                valid = (fixed > 0.01) | (moving.detach() > 0.01)
                first = fixed[valid]
                second = moving[valid]
                if first.numel() < 10:
                    return moving.sum() * 0 + 1
                first = first - first.mean()
                second = second - second.mean()
                denominator = torch.sqrt(
                    first.square().mean() * second.square().mean()
                ).clamp_min(1e-6)
                return -(first * second).mean() / denominator


            def register_to_reference_torch(
                fixed: sitk.Image, moving: sitk.Image
            ) -> tuple[sitk.Image, sitk.Transform, dict[str, object]]:
                started_at = time.perf_counter()
                initial = sitk.CenteredTransformInitializer(
                    fixed,
                    moving,
                    sitk.Euler3DTransform(),
                    sitk.CenteredTransformInitializerFilter.GEOMETRY,
                )
                centered = sitk.Resample(
                    moving, fixed, initial, sitk.sitkLinear, 0.0, sitk.sitkFloat32
                )
                fixed_array = sitk.GetArrayFromImage(
                    rescale_for_registration(fixed)
                ).astype(np.float32)
                moving_original_array = sitk.GetArrayFromImage(centered).astype(np.float32)
                moving_array = sitk.GetArrayFromImage(
                    rescale_for_registration(centered)
                ).astype(np.float32)
                correlation_before = normalized_correlation(
                    fixed_array, moving_array
                )

                device = torch.device('cuda')
                fixed_tensor = torch.from_numpy(fixed_array)[None, None].to(device)
                moving_tensor = torch.from_numpy(moving_array)[None, None].to(device)
                parameters = torch.zeros(6, device=device, requires_grad=True)

                for scale, iterations, learning_rate in TORCH_REGISTRATION_STAGES:
                    level_size = [
                        max(12, int(round(size * scale)))
                        for size in fixed_tensor.shape[2:]
                    ]
                    fixed_level = torch_functional.interpolate(
                        fixed_tensor, size=level_size, mode='trilinear',
                        align_corners=False,
                    )
                    moving_level = torch_functional.interpolate(
                        moving_tensor, size=level_size, mode='trilinear',
                        align_corners=False,
                    )
                    optimizer = torch.optim.Adam([parameters], lr=learning_rate)
                    best_loss = float('inf')
                    best_parameters = parameters.detach().clone()
                    stale_iterations = 0
                    for _ in range(iterations):
                        optimizer.zero_grad(set_to_none=True)
                        warped = torch_warp(moving_level, parameters)
                        loss = torch_ncc_loss(fixed_level, warped)
                        if not torch.isfinite(loss):
                            raise RuntimeError('PÃ©rdida CUDA no finita.')
                        loss.backward()
                        optimizer.step()
                        with torch.no_grad():
                            parameters[:3].clamp_(-0.35, 0.35)
                            parameters[3:].clamp_(-0.35, 0.35)
                        current_loss = float(loss.detach().cpu())
                        if current_loss < best_loss - 1e-5:
                            best_loss = current_loss
                            best_parameters = parameters.detach().clone()
                            stale_iterations = 0
                        else:
                            stale_iterations += 1
                        if stale_iterations >= TORCH_EARLY_STOPPING_PATIENCE:
                            break
                    with torch.no_grad():
                        parameters.copy_(best_parameters)

                with torch.no_grad():
                    normalized_registered_array = (
                        torch_warp(moving_tensor, parameters)
                        .squeeze().cpu().numpy().astype(np.float32)
                    )
                    original_tensor = torch.from_numpy(
                        moving_original_array
                    )[None, None].to(device)
                    registered_array = (
                        torch_warp(original_tensor, parameters)
                        .squeeze().cpu().numpy().astype(np.float32)
                    )
                registered = sitk.GetImageFromArray(registered_array)
                registered.CopyInformation(fixed)
                correlation_after = normalized_correlation(
                    fixed_array, normalized_registered_array
                )
                return registered, initial, {
                    'metric_final_correlation_objective': -best_loss,
                    'correlation_before': correlation_before,
                    'correlation_after': correlation_after,
                    'registration_backend': 'torch_cuda',
                    'gpu_fallback_used': False,
                    'registration_seconds': time.perf_counter() - started_at,
                }


            def register_to_reference(
                fixed: sitk.Image, moving: sitk.Image, mode: str
            ) -> tuple[sitk.Image, sitk.Transform, dict[str, object]]:
                if RESOLVED_REGISTRATION_BACKEND != 'torch_cuda' or mode != 'rigid':
                    return register_to_reference_sitk(fixed, moving, mode)
                try:
                    registered, transform, metrics = register_to_reference_torch(
                        fixed, moving
                    )
                    before = float(metrics['correlation_before'])
                    after = float(metrics['correlation_after'])
                    unreliable = (
                        not np.isfinite(after)
                        or after < 0.35
                        or (np.isfinite(before) and after < before - 0.03)
                    )
                    if unreliable and GPU_FALLBACK_TO_SITK:
                        fallback_image, fallback_transform, fallback_metrics = (
                            register_to_reference_sitk(fixed, moving, mode)
                        )
                        fallback_metrics['gpu_fallback_used'] = True
                        return fallback_image, fallback_transform, fallback_metrics
                    return registered, transform, metrics
                except RuntimeError:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    if not GPU_FALLBACK_TO_SITK:
                        raise
                    fallback_image, fallback_transform, fallback_metrics = (
                        register_to_reference_sitk(fixed, moving, mode)
                    )
                    fallback_metrics['gpu_fallback_used'] = True
                    return fallback_image, fallback_transform, fallback_metrics


            native_reference = read_sitk_from_zip(member_by_uid[REFERENCE_UID])
            reference_image = make_isotropic_reference(
                native_reference, ISOTROPIC_SPACING_MM
            )
            reference_array = sitk.GetArrayFromImage(reference_image).astype(np.float32)
            display(Markdown(
                f'**Grid común (x,y,z):** `{reference_image.GetSize()}` · '
                f'**spacing:** `{reference_image.GetSpacing()}` mm · **orientación:** LPS · '
                f'**backend:** `{RESOLVED_REGISTRATION_BACKEND}`'
            ))
            '''
        ),
        md(
            """
            ## 4. Máscaras: atlas validado o consenso de normales

            El fallback registra hasta 48 normales representativos y promedia intensidades normalizadas
            online. Conserva el componente dominante en cada hemisferio y define un fondo posterior de
            baja captación. Sigue siendo una aproximación: `mask_mode` permite impedir que se confunda con
            un atlas anatómico.
            """
        ),
        code(
            '''
            def normalize_display(array: np.ndarray) -> np.ndarray:
                positive = array[np.isfinite(array) & (array > 0)]
                values = positive if positive.size else array[np.isfinite(array)]
                low, high = np.percentile(values, [1, 99.5])
                return np.clip((array - low) / max(high - low, 1e-8), 0, 1)


            def representative_template_rows(frame: pd.DataFrame, total: int) -> pd.DataFrame:
                normals = frame.loc[frame['is_pathologic'] == 0].copy()
                pool = normals if len(normals) else frame.copy()
                pool = pool.sort_values(['within_protocol_qc_score', 'uid'])
                pieces = []
                family_count = max(1, pool['acquisition_family'].nunique())
                per_family = max(1, math.ceil(total / family_count))
                for _, group in pool.groupby('acquisition_family', sort=True):
                    pieces.append(group.head(per_family))
                return (
                    pd.concat(pieces, ignore_index=True)
                    .sort_values(['within_protocol_qc_score', 'uid'])
                    .head(total)
                )


            template_rows = representative_template_rows(cohort, MASK_TEMPLATE_SCANS)
            if REFERENCE_UID not in set(template_rows['uid'].astype(str)):
                template_rows = pd.concat([
                    cohort.loc[cohort['uid'].astype(str) == REFERENCE_UID],
                    template_rows.iloc[:-1] if len(template_rows) else template_rows,
                ]).drop_duplicates('uid')

            consensus = np.zeros(reference_array.shape, dtype=np.float64)
            consensus_count = 0
            template_failures: list[dict[str, str]] = []
            for row in tqdm(
                template_rows.itertuples(index=False),
                total=len(template_rows),
                desc='Construyendo consenso',
            ):
                uid = str(row.uid)
                try:
                    if uid == REFERENCE_UID:
                        registered_image = sitk.Image(reference_image)
                    else:
                        moving = read_sitk_from_zip(
                            member_by_uid[uid], keep_cached=True
                        )
                        registered_image, _, _ = register_to_reference(
                            reference_image, moving, REGISTRATION_MODE
                        )
                    candidate = normalize_display(
                        sitk.GetArrayViewFromImage(registered_image)
                    )
                    consensus_count += 1
                    consensus += (candidate - consensus) / consensus_count
                except Exception as error:
                    template_failures.append({
                        'uid': uid, 'error': f'{type(error).__name__}: {error}'
                    })
            if consensus_count < 2:
                raise RuntimeError('No se pudo construir un consenso con al menos dos exámenes.')
            consensus = consensus.astype(np.float32)


            def central_box(shape_zyx: tuple[int, int, int]) -> np.ndarray:
                z_size, y_size, x_size = shape_zyx
                mask = np.zeros(shape_zyx, dtype=bool)
                mask[
                    int(0.20 * z_size):int(0.80 * z_size),
                    int(0.20 * y_size):int(0.80 * y_size),
                    int(0.20 * x_size):int(0.80 * x_size),
                ] = True
                return mask


            def central_ellipsoid(shape_zyx: tuple[int, int, int]) -> np.ndarray:
                coordinates = np.indices(shape_zyx, dtype=np.float32)
                center = (np.asarray(shape_zyx, dtype=np.float32) - 1) / 2
                radii = np.maximum(np.asarray(shape_zyx, dtype=np.float32) * 0.42, 1)
                distance = sum(
                    ((coordinates[axis] - center[axis]) / radii[axis]) ** 2
                    for axis in range(3)
                )
                return distance <= 1


            def largest_component(mask: np.ndarray) -> np.ndarray:
                labels, count = ndimage.label(mask)
                if count == 0:
                    return np.zeros_like(mask, dtype=bool)
                sizes = np.bincount(labels.ravel())
                sizes[0] = 0
                return labels == int(np.argmax(sizes))


            def bilateral_target(array: np.ndarray, percentile: float) -> np.ndarray:
                positive = np.isfinite(array) & (array > 0)
                candidates = central_box(array.shape) & positive
                values = array[candidates]
                if values.size < 20:
                    raise ValueError('Consenso insuficiente para máscara objetivo.')
                raw = candidates & (array >= np.percentile(values, percentile))
                x_mid = array.shape[2] // 2
                target = np.zeros_like(raw)
                target[:, :, :x_mid] = largest_component(raw[:, :, :x_mid])
                target[:, :, x_mid:] = largest_component(raw[:, :, x_mid:])
                target = ndimage.binary_closing(target, iterations=1)
                return ndimage.binary_fill_holes(target)


            def proxy_masks_from_consensus(
                array: np.ndarray, percentile: float
            ) -> dict[str, np.ndarray]:
                target = bilateral_target(array, percentile)
                positive = np.isfinite(array) & (array > 0)
                support_values = array[positive]
                support = positive & (array >= np.percentile(support_values, 10))
                support = ndimage.binary_closing(support, iterations=2)
                brain = largest_component(support & central_ellipsoid(array.shape))
                brain = ndimage.binary_fill_holes(
                    ndimage.binary_closing(brain, iterations=2)
                )
                if brain.sum() < max(100, target.sum() * 2):
                    brain = support & central_box(array.shape)
                posterior_band = np.zeros_like(target)
                posterior_band[
                    int(0.15 * target.shape[0]):int(0.85 * target.shape[0]),
                    int(0.55 * target.shape[1]):int(0.95 * target.shape[1]),
                    int(0.10 * target.shape[2]):int(0.90 * target.shape[2]),
                ] = True
                exclusion = ndimage.binary_dilation(target, iterations=3)
                brain_values = array[brain & positive]
                low_cut = np.percentile(brain_values, 70)
                background = posterior_band & brain & (~exclusion) & (array <= low_cut)
                if background.sum() < max(20, target.sum() // 2):
                    background = brain & (~exclusion) & (array <= low_cut)

                x_mid = target.shape[2] // 2
                y_mid = target.shape[1] // 2
                right = target.copy(); right[:, :, x_mid:] = False
                left = target.copy(); left[:, :, :x_mid] = False
                anterior = target.copy(); anterior[:, y_mid:, :] = False
                posterior = target.copy(); posterior[:, :y_mid, :] = False
                if min(target.sum(), background.sum(), right.sum(), left.sum()) == 0:
                    raise ValueError('La máscara de consenso quedó vacía en una región requerida.')
                return {
                    'target': target, 'background': background, 'brain': brain,
                    'right': right, 'left': left,
                    'anterior': anterior, 'posterior': posterior,
                }


            def load_validated_masks(path: Path) -> dict[str, np.ndarray]:
                atlas = sitk.DICOMOrient(sitk.ReadImage(str(path), sitk.sitkUInt16), 'LPS')
                atlas = sitk.Resample(
                    atlas, reference_image, sitk.Transform(3, sitk.sitkIdentity),
                    sitk.sitkNearestNeighbor, 0,
                )
                labels = sitk.GetArrayFromImage(atlas)
                right = labels == ATLAS_LABELS['right_target']
                left = labels == ATLAS_LABELS['left_target']
                background = labels == ATLAS_LABELS['background']
                target = right | left
                brain = labels > 0
                y_mid = labels.shape[1] // 2
                result = {
                    'target': target, 'background': background, 'brain': brain,
                    'right': right, 'left': left,
                    'anterior': target & (np.indices(target.shape)[1] < y_mid),
                    'posterior': target & (np.indices(target.shape)[1] >= y_mid),
                }
                optional_labels = [
                    'right_caudate', 'left_caudate', 'right_putamen', 'left_putamen'
                ]
                for name in optional_labels:
                    if name in ATLAS_LABELS:
                        result[name] = labels == ATLAS_LABELS[name]
                if min(target.sum(), background.sum(), right.sum(), left.sum()) == 0:
                    raise ValueError('El atlas no contiene todas las regiones requeridas.')
                return result


            if VALIDATED_MASK_PATH is not None:
                if not VALIDATED_MASK_PATH.exists():
                    raise FileNotFoundError(VALIDATED_MASK_PATH)
                masks = load_validated_masks(VALIDATED_MASK_PATH)
                sensitivity_targets = {'atlas_base': masks['target']}
                mask_mode = 'validated_atlas'
            else:
                masks = proxy_masks_from_consensus(consensus, TARGET_BASE_PERCENTILE)
                sensitivity_targets = {
                    f'p{int(percentile)}': proxy_masks_from_consensus(consensus, percentile)['target']
                    for percentile in TARGET_PERCENTILES
                }
                mask_mode = 'consensus_uptake_proxy'

            with MASKS_PATH.open('wb') as stream:
                np.savez_compressed(
                    stream,
                    consensus=consensus,
                    **{name: value.astype(np.uint8) for name, value in masks.items()},
                )
            display(Markdown(
                f'**Modo de máscara:** `{mask_mode}` · **template:** `{consensus_count}` · '
                f'**target:** `{masks["target"].sum():,}` voxels · '
                f'**background:** `{masks["background"].sum():,}` voxels'
            ))
            '''
        ),
        md(
            """
            ## 5. Features semicuantitativas, forma, textura 3D y sensibilidad

            El fondo se intersecta con el soporte positivo de cada estudio y dispone de fallbacks y un
            piso relativo al percentil 90 del foreground. Se conservan tanto la razón robusta como flags
            de validez; ningún valor se presenta como SBR clínicamente validado. La textura usa una GLCM
            3D de 32 niveles y 13 direcciones y todavía requiere verificación IBSI.
            """
        ),
        code(
            '''
            def masked_values(array: np.ndarray, mask: np.ndarray) -> np.ndarray:
                return array[mask & np.isfinite(array)]


            def safe_mean(array: np.ndarray, mask: np.ndarray) -> float:
                values = masked_values(array, mask)
                return float(values.mean()) if values.size else float('nan')


            def sbr_like(target_mean: float, background_mean: float) -> float:
                if not np.isfinite(background_mean) or background_mean <= 1e-8:
                    return float('nan')
                return float((target_mean - background_mean) / background_mean)


            def robust_background_context(
                array: np.ndarray, base_masks: dict[str, np.ndarray]
            ) -> dict[str, object]:
                positive = np.isfinite(array) & (array > 0)
                brain_mask = base_masks.get('brain', central_ellipsoid(array.shape))
                foreground_mask = brain_mask & positive
                if foreground_mask.sum() < MIN_BACKGROUND_VOXELS:
                    foreground_mask = central_ellipsoid(array.shape) & positive
                if foreground_mask.sum() < MIN_BACKGROUND_VOXELS:
                    foreground_mask = positive
                foreground_values = array[foreground_mask]
                if foreground_values.size < MIN_BACKGROUND_VOXELS:
                    raise ValueError('Foreground positivo insuficiente después del registro.')

                foreground_p50, foreground_p90, foreground_p99 = np.percentile(
                    foreground_values, [50, 90, 99]
                )
                scale_floor = max(
                    1e-8,
                    BACKGROUND_SCALE_FLOOR_FRACTION * float(foreground_p90),
                )
                nominal_background = base_masks['background']
                candidate = nominal_background & positive
                support_fraction = float(candidate.sum() / max(nominal_background.sum(), 1))
                source = 'consensus_brain_background'
                enough_support = (
                    candidate.sum() >= MIN_BACKGROUND_VOXELS
                    and support_fraction >= MIN_BACKGROUND_SUPPORT_FRACTION
                )
                if not enough_support:
                    exclusion = ndimage.binary_dilation(
                        base_masks['target'], iterations=5
                    )
                    candidate = foreground_mask & (~exclusion)
                    if candidate.any():
                        candidate &= array <= np.percentile(array[candidate], 70)
                    source = 'scan_brain_fallback'

                values = array[candidate & positive]
                if values.size < MIN_BACKGROUND_VOXELS:
                    values = foreground_values[
                        foreground_values <= np.percentile(foreground_values, 50)
                    ]
                    source = 'foreground_lower_half_fallback'
                if values.size < 10:
                    raise ValueError('Fondo robusto insuficiente después del registro.')

                low, high = np.percentile(values, [5, 95])
                trimmed = values[(values >= low) & (values <= high)]
                if trimmed.size < 10:
                    trimmed = values
                raw_mean = float(trimmed.mean())
                raw_std = float(trimmed.std(ddof=1)) if trimmed.size > 1 else 0.0
                floor_applied = (not np.isfinite(raw_mean)) or raw_mean < scale_floor
                scale = max(raw_mean if np.isfinite(raw_mean) else 0.0, scale_floor)
                valid = bool(
                    enough_support
                    and source == 'consensus_brain_background'
                    and not floor_applied
                    and np.isfinite(raw_std)
                )
                return {
                    'values': trimmed,
                    'source': source,
                    'support_voxels': int(values.size),
                    'support_fraction': support_fraction,
                    'mean_raw': raw_mean,
                    'std_raw': raw_std,
                    'scale_floor': float(scale_floor),
                    'scale': float(scale),
                    'floor_applied': bool(floor_applied),
                    'qc_valid': valid,
                    'foreground_p50': float(foreground_p50),
                    'foreground_p90': float(foreground_p90),
                    'foreground_p99': float(foreground_p99),
                }


            def largest_component_shape(
                mask: np.ndarray, spacing_xyz: tuple[float, float, float]
            ) -> dict[str, float]:
                labels, components = ndimage.label(mask)
                if components == 0:
                    return {
                        'volume_ml': 0.0, 'surface_mm2': float('nan'),
                        'elongation': float('nan'), 'sphericity': float('nan'),
                        'extent': float('nan'), 'components': 0,
                        'largest_component_fraction': 0.0,
                    }
                sizes = np.bincount(labels.ravel())[1:]
                largest = labels == (int(np.argmax(sizes)) + 1)
                coords = np.argwhere(largest)
                spacing_zyx = np.asarray(spacing_xyz[::-1], dtype=float)
                physical = coords * spacing_zyx
                eigenvalues = np.linalg.eigvalsh(np.cov(physical, rowvar=False))
                elongation = math.sqrt(
                    max(float(eigenvalues[-1]), 0) / max(float(eigenvalues[0]), 1e-8)
                )
                volume_mm3 = float(largest.sum() * np.prod(spacing_xyz))
                bbox_size = (coords.max(axis=0) - coords.min(axis=0) + 1) * spacing_zyx
                bbox_volume = float(np.prod(bbox_size))
                try:
                    vertices, faces, _, _ = marching_cubes(
                        largest.astype(np.uint8), level=0.5, spacing=tuple(spacing_zyx)
                    )
                    surface = float(mesh_surface_area(vertices, faces))
                except (ValueError, RuntimeError):
                    surface = float('nan')
                sphericity = (
                    float((math.pi ** (1 / 3)) * ((6 * volume_mm3) ** (2 / 3)) / surface)
                    if np.isfinite(surface) and surface > 0 else float('nan')
                )
                return {
                    'volume_ml': volume_mm3 / 1000,
                    'surface_mm2': surface,
                    'elongation': float(elongation),
                    'sphericity': sphericity,
                    'extent': volume_mm3 / max(bbox_volume, 1e-8),
                    'components': int(components),
                    'largest_component_fraction': float(sizes.max() / max(mask.sum(), 1)),
                }


            OFFSETS_3D = [
                (1, 0, 0), (0, 1, 0), (0, 0, 1),
                (1, 1, 0), (1, -1, 0), (1, 0, 1), (1, 0, -1),
                (0, 1, 1), (0, 1, -1),
                (1, 1, 1), (1, 1, -1), (1, -1, 1), (1, -1, -1),
            ]


            def paired_slices(size: int, offset: int) -> tuple[slice, slice]:
                if offset > 0:
                    return slice(0, size - offset), slice(offset, size)
                if offset < 0:
                    return slice(-offset, size), slice(0, size + offset)
                return slice(0, size), slice(0, size)


            def texture_glcm_3d(
                ratio: np.ndarray, mask: np.ndarray, levels: int = 32
            ) -> dict[str, float]:
                values = masked_values(ratio, mask)
                names = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'entropy', 'correlation']
                if values.size < 30 or float(values.max()) <= float(values.min()):
                    return {name: float('nan') for name in names}
                low, high = np.percentile(values, [1, 99])
                quantized = np.rint(
                    np.clip((ratio - low) / max(high - low, 1e-8), 0, 1) * (levels - 1)
                ).astype(np.uint8)
                matrix = np.zeros((levels, levels), dtype=np.float64)
                for dz, dy, dx in OFFSETS_3D:
                    z0, z1 = paired_slices(mask.shape[0], dz)
                    y0, y1 = paired_slices(mask.shape[1], dy)
                    x0, x1 = paired_slices(mask.shape[2], dx)
                    valid = mask[z0, y0, x0] & mask[z1, y1, x1]
                    if not valid.any():
                        continue
                    first = quantized[z0, y0, x0][valid]
                    second = quantized[z1, y1, x1][valid]
                    np.add.at(matrix, (first, second), 1)
                    np.add.at(matrix, (second, first), 1)
                if matrix.sum() == 0:
                    return {name: float('nan') for name in names}
                probability = matrix / matrix.sum()
                i, j = np.indices(probability.shape)
                contrast = float((probability * (i - j) ** 2).sum())
                dissimilarity = float((probability * np.abs(i - j)).sum())
                homogeneity = float((probability / (1 + (i - j) ** 2)).sum())
                energy = float(np.sqrt((probability ** 2).sum()))
                nonzero = probability[probability > 0]
                texture_entropy = float(-(nonzero * np.log2(nonzero)).sum())
                row = probability.sum(axis=1)
                column = probability.sum(axis=0)
                mean_i = float((np.arange(levels) * row).sum())
                mean_j = float((np.arange(levels) * column).sum())
                std_i = math.sqrt(float((((np.arange(levels) - mean_i) ** 2) * row).sum()))
                std_j = math.sqrt(float((((np.arange(levels) - mean_j) ** 2) * column).sum()))
                correlation = float(
                    (probability * (i - mean_i) * (j - mean_j)).sum()
                    / max(std_i * std_j, 1e-8)
                )
                return {
                    'contrast': contrast, 'dissimilarity': dissimilarity,
                    'homogeneity': homogeneity, 'energy': energy,
                    'entropy': texture_entropy, 'correlation': correlation,
                }


            def shifted_mask(mask: np.ndarray, shift_zyx: tuple[int, int, int]) -> np.ndarray:
                return ndimage.shift(
                    mask.astype(np.uint8), shift=shift_zyx,
                    order=0, mode='constant', cval=0,
                ).astype(bool)


            def feature_record(
                uid: str, array: np.ndarray, base_masks: dict[str, np.ndarray]
            ) -> tuple[dict[str, object], np.ndarray, np.ndarray]:
                background = robust_background_context(array, base_masks)
                background_mean = float(background['scale'])
                background_std = float(background['std_raw'])
                ratio = array / background_mean
                intensity01 = np.clip(
                    array / max(float(background['foreground_p99']), 1e-8), 0, 1
                )

                target_mean = safe_mean(array, base_masks['target'])
                right_mean = safe_mean(array, base_masks['right'])
                left_mean = safe_mean(array, base_masks['left'])
                anterior_mean = safe_mean(array, base_masks['anterior'])
                posterior_mean = safe_mean(array, base_masks['posterior'])
                side_mean = max((left_mean + right_mean) / 2, 1e-8)
                target_ratio = masked_values(ratio, base_masks['target'])

                active_threshold = background_mean + ACTIVE_BACKGROUND_SD * background_std
                active_mask = base_masks['target'] & np.isfinite(array) & (array >= active_threshold)
                active_mask = ndimage.binary_closing(active_mask, iterations=1)
                shape = largest_component_shape(active_mask, reference_image.GetSpacing())
                texture = texture_glcm_3d(ratio, base_masks['target'], TEXTURE_LEVELS)

                threshold_sbr = [
                    sbr_like(safe_mean(array, candidate), background_mean)
                    for candidate in sensitivity_targets.values()
                ]
                morphology_masks = [
                    ndimage.binary_erosion(base_masks['target'], iterations=1),
                    base_masks['target'],
                    ndimage.binary_dilation(base_masks['target'], iterations=1),
                ]
                morphology_sbr = [
                    sbr_like(safe_mean(array, candidate), background_mean)
                    for candidate in morphology_masks
                ]
                translation_sbr = [
                    sbr_like(safe_mean(array, shifted_mask(base_masks['target'], shift)), background_mean)
                    for shift in [(0, 0, 0), (0, 0, 1), (0, 0, -1), (0, 1, 0), (0, -1, 0)]
                ]
                background_means = [
                    safe_mean(array, candidate)
                    for candidate in [
                        ndimage.binary_erosion(base_masks['background'], iterations=1),
                        base_masks['background'],
                        ndimage.binary_dilation(base_masks['background'], iterations=1),
                    ]
                ]
                background_sbr = [
                    sbr_like(target_mean, max(value, float(background['scale_floor'])))
                    if np.isfinite(value) else float('nan')
                    for value in background_means
                ]

                record: dict[str, object] = {
                    'uid': uid,
                    'mask_mode': mask_mode,
                    'background_source': str(background['source']),
                    'background_qc_valid': bool(background['qc_valid']),
                    'background_floor_applied': bool(background['floor_applied']),
                    'background_support_voxels': int(background['support_voxels']),
                    'background_support_fraction': float(background['support_fraction']),
                    'background_mean_raw': float(background['mean_raw']),
                    'background_std_raw': background_std,
                    'background_scale_floor': float(background['scale_floor']),
                    'background_scale_used': background_mean,
                    'foreground_p50_registered': float(background['foreground_p50']),
                    'foreground_p90_registered': float(background['foreground_p90']),
                    'foreground_p99_registered': float(background['foreground_p99']),
                    'semiquant_sbr': sbr_like(target_mean, background_mean),
                    'semiquant_right_sbr': sbr_like(right_mean, background_mean),
                    'semiquant_left_sbr': sbr_like(left_mean, background_mean),
                    'semiquant_min_side_sbr': min(
                        sbr_like(right_mean, background_mean),
                        sbr_like(left_mean, background_mean),
                    ),
                    'semiquant_lr_asymmetry_abs': abs(left_mean - right_mean) / side_mean,
                    'semiquant_lr_asymmetry_signed': (left_mean - right_mean) / side_mean,
                    'semiquant_posterior_anterior_ratio': posterior_mean / max(anterior_mean, 1e-8),
                    'semiquant_log_target_background': float(
                        np.log1p(max(target_mean, 0) / background_mean)
                    ),
                    'firstorder_ratio_mean': float(target_ratio.mean()),
                    'firstorder_ratio_std': float(target_ratio.std(ddof=1)),
                    'firstorder_ratio_p10': float(np.percentile(target_ratio, 10)),
                    'firstorder_ratio_p50': float(np.percentile(target_ratio, 50)),
                    'firstorder_ratio_p90': float(np.percentile(target_ratio, 90)),
                    'active_threshold_sbr': float(
                        (active_threshold - background_mean) / background_mean
                    ),
                    **{f'shape_active_{name}': value for name, value in shape.items()},
                    **{f'texture3d_{name}': value for name, value in texture.items()},
                    'stability_threshold_span': float(np.nanmax(threshold_sbr) - np.nanmin(threshold_sbr)),
                    'stability_mask_span': float(np.nanmax(morphology_sbr) - np.nanmin(morphology_sbr)),
                    'stability_translation_span': float(np.nanmax(translation_sbr) - np.nanmin(translation_sbr)),
                    'stability_background_span': float(np.nanmax(background_sbr) - np.nanmin(background_sbr)),
                }
                if all(name in base_masks for name in [
                    'right_caudate', 'left_caudate', 'right_putamen', 'left_putamen'
                ]):
                    caudate = safe_mean(
                        array, base_masks['right_caudate'] | base_masks['left_caudate']
                    )
                    putamen = safe_mean(
                        array, base_masks['right_putamen'] | base_masks['left_putamen']
                    )
                    record['semiquant_putamen_caudate_ratio'] = putamen / max(caudate, 1e-8)
                return record, ratio.astype(np.float32), intensity01.astype(np.float32)
            '''
        ),
        md(
            """
            ## 6. Ejecución streaming, checkpoints y crops

            Los crops guardan tanto la razón robusta contra fondo como una intensidad acotada por el
            percentil 99 del foreground. Comparten el grid de referencia y se almacenan en `float16`.
            Un fondo dudoso queda marcado, pero ya no elimina silenciosamente el estudio.
            """
        ),
        code(
            '''
            config_payload = {
                'algorithm_version': 'full_cohort_v3',
                'run_profile': RUN_PROFILE,
                'reference_uid': REFERENCE_UID,
                'registration_mode': REGISTRATION_MODE,
                'registration_backend_requested': REGISTRATION_BACKEND,
                'registration_backend_resolved': RESOLVED_REGISTRATION_BACKEND,
                'torch_registration_stages': [list(stage) for stage in TORCH_REGISTRATION_STAGES],
                'torch_early_stopping_patience': TORCH_EARLY_STOPPING_PATIENCE,
                'gpu_fallback_to_sitk': GPU_FALLBACK_TO_SITK,
                'sitk_metric_sampling': SITK_METRIC_SAMPLING,
                'sitk_iterations': SITK_ITERATIONS,
                'checkpoint_every': CHECKPOINT_EVERY,
                'retry_failures': RETRY_FAILURES,
                'cuda_device': (
                    torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
                ),
                'isotropic_spacing_mm': ISOTROPIC_SPACING_MM,
                'mask_mode': mask_mode,
                'validated_mask_path': str(VALIDATED_MASK_PATH) if VALIDATED_MASK_PATH else None,
                'atlas_labels': ATLAS_LABELS if VALIDATED_MASK_PATH else None,
                'target_base_percentile': TARGET_BASE_PERCENTILE,
                'target_percentiles': list(TARGET_PERCENTILES),
                'active_background_sd': ACTIVE_BACKGROUND_SD,
                'background_scale_floor_fraction': BACKGROUND_SCALE_FLOOR_FRACTION,
                'min_background_voxels': MIN_BACKGROUND_VOXELS,
                'min_background_support_fraction': MIN_BACKGROUND_SUPPORT_FRACTION,
                'texture_levels': TEXTURE_LEVELS,
                'max_scans': MAX_SCANS,
                'cohort_uid_hash': hashlib.sha256(
                    '|'.join(cohort['uid'].astype(str)).encode('utf-8')
                ).hexdigest(),
            }
            config_hash = hashlib.sha256(
                json.dumps(config_payload, sort_keys=True).encode('utf-8')
            ).hexdigest()
            config_payload['config_hash'] = config_hash

            if CONFIG_PATH.exists() and RESUME:
                previous_config = json.loads(CONFIG_PATH.read_text(encoding='utf-8'))
                if previous_config.get('config_hash') != config_hash:
                    raise RuntimeError(
                        'Existe un checkpoint full-cohort con otra configuración. '
                        f'Archiva {FULL_OUTPUT_DIR} o restaura sus parámetros antes de continuar.'
                    )
            else:
                CONFIG_PATH.write_text(
                    json.dumps(config_payload, indent=2), encoding='utf-8'
                )

            feature_rows = (
                pd.read_csv(FEATURES_PATH).to_dict('records')
                if RESUME and FEATURES_PATH.exists() else []
            )
            registration_rows = (
                pd.read_csv(REGISTRATION_PATH).to_dict('records')
                if RESUME and REGISTRATION_PATH.exists() else []
            )
            failure_rows = (
                pd.read_csv(FAILURES_PATH).to_dict('records')
                if RESUME and FAILURES_PATH.exists() else []
            )

            map_state = {
                0: {'n': 0, 'mean': np.zeros(reference_array.shape, dtype=np.float64),
                    'm2': np.zeros(reference_array.shape, dtype=np.float64)},
                1: {'n': 0, 'mean': np.zeros(reference_array.shape, dtype=np.float64),
                    'm2': np.zeros(reference_array.shape, dtype=np.float64)},
            }
            mapped_uids: set[str] = set()
            legacy_saved = globals().get('saved')
            if isinstance(legacy_saved, np.lib.npyio.NpzFile):
                legacy_saved.close()
            if RESUME and STATE_PATH.exists():
                with np.load(STATE_PATH, allow_pickle=False) as saved_state:
                    for label in [0, 1]:
                        map_state[label]['n'] = int(saved_state[f'n{label}'])
                        map_state[label]['mean'] = saved_state[f'mean{label}'].astype(np.float64)
                        map_state[label]['m2'] = saved_state[f'm2_{label}'].astype(np.float64)
                    if 'mapped_uids' in saved_state.files:
                        mapped_uids = {
                            str(uid) for uid in saved_state['mapped_uids'].astype(str)
                        }

                if not mapped_uids and sum(map_state[label]['n'] for label in [0, 1]):
                    feature_frame_resume = pd.DataFrame(feature_rows)
                    for label in [0, 1]:
                        available = set(
                            feature_frame_resume.loc[
                                feature_frame_resume['is_pathologic'].astype(int) == label,
                                'uid',
                            ].astype(str)
                        )
                        ordered = [
                            str(uid)
                            for uid in cohort.loc[
                                cohort['is_pathologic'] == label, 'uid'
                            ].astype(str)
                            if str(uid) in available
                        ]
                        expected = int(map_state[label]['n'])
                        if expected > len(ordered):
                            raise RuntimeError(
                                'El estado online contiene mÃ¡s casos que la tabla de features.'
                            )
                        mapped_uids.update(ordered[:expected])
            elif feature_rows:
                raise RuntimeError('Hay features reanudables, pero falta streaming_state_full.npz.')

            label_by_uid = dict(zip(
                cohort['uid'].astype(str), cohort['is_pathologic'].astype(int)
            ))
            mapped_counts = {
                label: sum(label_by_uid.get(uid) == label for uid in mapped_uids)
                for label in [0, 1]
            }
            if any(mapped_counts[label] != map_state[label]['n'] for label in [0, 1]):
                raise RuntimeError(
                    'Los UID del estado online no coinciden con sus conteos por clase.'
                )

            def online_update(label: int, uid: str, array: np.ndarray) -> None:
                if uid in mapped_uids:
                    return
                state = map_state[int(label)]
                state['n'] += 1
                delta = array - state['mean']
                state['mean'] += delta / state['n']
                state['m2'] += delta * (array - state['mean'])
                mapped_uids.add(uid)


            crop_indices = np.argwhere(masks['target'])
            spacing_zyx = np.asarray(reference_image.GetSpacing()[::-1])
            margin = np.ceil(CROP_MARGIN_MM / spacing_zyx).astype(int)
            crop_start = np.maximum(crop_indices.min(axis=0) - margin, 0)
            crop_stop = np.minimum(crop_indices.max(axis=0) + margin + 1, masks['target'].shape)
            crop_slices = tuple(
                slice(int(start), int(stop)) for start, stop in zip(crop_start, crop_stop)
            )


            def save_crop(
                uid: str, ratio: np.ndarray, intensity01: np.ndarray
            ) -> None:
                destination = CROPS_DIR / f'{uid}.npz'
                if destination.exists() and RESUME:
                    try:
                        with np.load(destination, allow_pickle=False) as saved_crop:
                            if {'volume_ratio', 'volume_intensity01'} <= set(saved_crop.files):
                                return
                    except (OSError, ValueError, EOFError, zipfile.BadZipFile):
                        pass
                temporary = destination.with_suffix('.npz.tmp')
                with temporary.open('wb') as stream:
                    np.savez_compressed(
                        stream,
                        volume_ratio=np.clip(ratio[crop_slices], 0, 20).astype(np.float16),
                        volume_intensity01=intensity01[crop_slices].astype(np.float16),
                        target_mask=masks['target'][crop_slices].astype(np.uint8),
                        crop_start_zyx=crop_start.astype(np.int16),
                        spacing_xyz=np.asarray(reference_image.GetSpacing(), dtype=np.float32),
                    )
                replace_with_retry(temporary, destination)


            def replace_with_retry(
                temporary: Path, destination: Path, attempts: int = 10
            ) -> None:
                delay_seconds = 0.25
                for attempt in range(attempts):
                    try:
                        temporary.replace(destination)
                        return
                    except PermissionError as error:
                        if attempt == attempts - 1:
                            raise PermissionError(
                                f'No se pudo reemplazar {destination}. Cierra visores/Excel que '
                                'tengan abierto el archivo y pausa temporalmente OneDrive.'
                            ) from error
                        time.sleep(delay_seconds)
                        delay_seconds = min(delay_seconds * 2, 5.0)


            def atomic_csv(rows: list[dict[str, object]], path: Path, columns=None) -> None:
                temporary = path.with_suffix(path.suffix + '.tmp')
                pd.DataFrame(rows, columns=columns).to_csv(temporary, index=False)
                replace_with_retry(temporary, path)


            def checkpoint() -> None:
                features_frame = pd.DataFrame(feature_rows).drop_duplicates('uid', keep='last')
                registration_frame = pd.DataFrame(registration_rows).drop_duplicates('uid', keep='last')
                temporary_features = FEATURES_PATH.with_suffix('.csv.tmp')
                temporary_registration = REGISTRATION_PATH.with_suffix('.csv.tmp')
                features_frame.sort_values('uid').to_csv(temporary_features, index=False)
                registration_frame.sort_values('uid').to_csv(temporary_registration, index=False)
                replace_with_retry(temporary_features, FEATURES_PATH)
                replace_with_retry(temporary_registration, REGISTRATION_PATH)
                atomic_csv(failure_rows, FAILURES_PATH, columns=['uid', 'error'])
                temporary_state = STATE_PATH.with_suffix('.npz.tmp')
                uid_width = max((len(uid) for uid in mapped_uids), default=1)
                with temporary_state.open('wb') as stream:
                    np.savez_compressed(
                        stream,
                        n0=np.asarray(map_state[0]['n']), mean0=map_state[0]['mean'],
                        m2_0=map_state[0]['m2'],
                        n1=np.asarray(map_state[1]['n']), mean1=map_state[1]['mean'],
                        m2_1=map_state[1]['m2'],
                        mapped_uids=np.asarray(
                            sorted(mapped_uids), dtype=f'<U{uid_width}'
                        ),
                    )
                replace_with_retry(temporary_state, STATE_PATH)


            completed = set(mapped_uids)
            if not RETRY_FAILURES:
                completed |= {str(row['uid']) for row in failure_rows}
            processed_since_checkpoint = 0

            for row in tqdm(
                cohort.itertuples(index=False), total=len(cohort), desc='Full-cohort streaming'
            ):
                uid = str(row.uid)
                if uid in completed:
                    continue
                try:
                    if uid == REFERENCE_UID:
                        registered_image = sitk.Image(reference_image)
                        metrics = {
                            'metric_final_correlation_objective': float('nan'),
                            'correlation_before': 1.0, 'correlation_after': 1.0,
                            'registration_backend': RESOLVED_REGISTRATION_BACKEND,
                            'gpu_fallback_used': False,
                            'registration_seconds': 0.0,
                        }
                    else:
                        moving = read_sitk_from_zip(member_by_uid[uid])
                        registered_image, _, metrics = register_to_reference(
                            reference_image, moving, REGISTRATION_MODE
                        )
                    registered_array = sitk.GetArrayFromImage(registered_image).astype(np.float32)
                    features, ratio, map_volume = feature_record(
                        uid, registered_array, masks
                    )
                    features['is_pathologic'] = int(row.is_pathologic)
                    features['acquisition_family'] = str(row.acquisition_family)
                    feature_rows.append(features)
                    failure_rows = [
                        failure for failure in failure_rows
                        if str(failure.get('uid')) != uid
                    ]

                    correlation_after = float(metrics['correlation_after'])
                    correlation_before = float(metrics['correlation_before'])
                    registration_rows.append({
                        'uid': uid,
                        'is_pathologic': int(row.is_pathologic),
                        'acquisition_family': str(row.acquisition_family),
                        **metrics,
                        'correlation_gain': correlation_after - correlation_before,
                        'registration_qc_flag': bool(
                            (not np.isfinite(correlation_after))
                            or (correlation_after < 0.55)
                            or (correlation_after - correlation_before < -0.05)
                        ),
                    })
                    online_update(int(row.is_pathologic), uid, map_volume)
                    if SAVE_REGISTERED_CROPS:
                        save_crop(uid, ratio, map_volume)
                except (KeyboardInterrupt, SystemExit):
                    checkpoint()
                    raise
                except Exception as error:
                    failure_rows = [
                        failure for failure in failure_rows
                        if str(failure.get('uid')) != uid
                    ]
                    failure_rows.append({
                        'uid': uid, 'error': f'{type(error).__name__}: {error}'
                    })

                processed_since_checkpoint += 1
                if processed_since_checkpoint >= CHECKPOINT_EVERY:
                    checkpoint()
                    processed_since_checkpoint = 0

            checkpoint()
            features = pd.read_csv(FEATURES_PATH, dtype={'uid': 'string'})
            registration_qc = pd.read_csv(REGISTRATION_PATH, dtype={'uid': 'string'})
            failures = pd.read_csv(FAILURES_PATH, dtype={'uid': 'string'})
            mean_registration_seconds = pd.to_numeric(
                registration_qc.get('registration_seconds'), errors='coerce'
            ).mean()
            gpu_fallbacks = int(
                registration_qc.get(
                    'gpu_fallback_used', pd.Series(False, index=registration_qc.index)
                ).fillna(False).astype(bool).sum()
            )
            background_valid = int(features['background_qc_valid'].astype(bool).sum())
            background_floored = int(features['background_floor_applied'].astype(bool).sum())
            display(Markdown(
                f'**Features:** `{len(features):,}/{len(cohort):,}` · '
                f'**fallos:** `{len(failures):,}` · '
                f'**crops:** `{len(list(CROPS_DIR.glob("*.npz"))) if SAVE_REGISTERED_CROPS else 0:,}` · '
                f'**backend:** `{RESOLVED_REGISTRATION_BACKEND}` · '
                f'**tiempo medio/registro:** `{mean_registration_seconds:.2f} s` · '
                f'**fallbacks GPU→CPU:** `{gpu_fallbacks}`'
            ))
            '''
        ),
        md(
            """
            ## 7. Mapas online, efecto estandarizado y control visual

            Las medias son voxel-a-voxel de volúmenes registrados y normalizados por el percentil 99 del
            foreground; no son MIP ni dependen del denominador SBR.
            Cohen d usa varianza agrupada ponderada por tamaño muestral. También se guarda Hedges g, que
            corrige el sesgo de muestra pequeña.
            """
        ),
        code(
            '''
            n0, n1 = map_state[0]['n'], map_state[1]['n']
            normal_mean = map_state[0]['mean'].astype(np.float32)
            pathologic_mean = map_state[1]['mean'].astype(np.float32)
            normal_var = (
                map_state[0]['m2'] / max(n0 - 1, 1)
            ).astype(np.float32)
            pathologic_var = (
                map_state[1]['m2'] / max(n1 - 1, 1)
            ).astype(np.float32)
            difference = pathologic_mean - normal_mean
            pooled_variance = (
                ((n0 - 1) * normal_var + (n1 - 1) * pathologic_var)
                / max(n0 + n1 - 2, 1)
            )
            pooled_std = np.sqrt(np.maximum(pooled_variance, 0))
            cohen_d = np.divide(
                difference, pooled_std,
                out=np.zeros_like(difference), where=pooled_std > 1e-6,
            )
            correction = 1 - 3 / max(4 * (n0 + n1) - 9, 1)
            hedges_g = correction * cohen_d

            with MAPS_PATH.open('wb') as stream:
                np.savez_compressed(
                    stream,
                    normal_mean=normal_mean,
                    pathologic_mean=pathologic_mean,
                    difference=difference,
                    pooled_std=pooled_std,
                    cohen_d=cohen_d,
                    hedges_g=hedges_g,
                    n_normal=np.asarray(n0),
                    n_pathologic=np.asarray(n1),
                )

            target_z = int(np.argmax(masks['target'].sum(axis=(1, 2))))
            map_z = int(np.argmax((np.abs(hedges_g) * masks['target']).sum(axis=(1, 2))))
            figure, axes = plt.subplots(2, 4, figsize=(17, 8))
            reference01 = normalize_display(consensus)
            mask_overlay = np.dstack([
                reference01[target_z],
                masks['target'][target_z].astype(float),
                masks['background'][target_z].astype(float),
            ])
            axes[0, 0].imshow(np.rot90(reference01[target_z]), cmap='hot', vmin=0, vmax=1)
            axes[0, 0].set_title('Consenso')
            axes[0, 1].imshow(np.rot90(mask_overlay), vmin=0, vmax=1)
            axes[0, 1].set_title('Intensidad / target / fondo')
            axes[0, 2].hist(registration_qc['correlation_after'].dropna(), bins=30)
            axes[0, 2].set_title('Correlación posterior al registro')
            sns.boxplot(
                data=features, x='is_pathologic',
                y='semiquant_log_target_background', ax=axes[0, 3]
            )
            axes[0, 3].set_title('log(1 + target/fondo robusto) por clase')
            images = [normal_mean[map_z], pathologic_mean[map_z], difference[map_z], hedges_g[map_z]]
            titles = ['Media normal', 'Media patológica', 'Diferencia', 'Hedges g descriptivo']
            cmaps = ['hot', 'hot', 'coolwarm', 'coolwarm']
            for axis, image, title, cmap in zip(axes[1], images, titles, cmaps):
                limit = np.nanpercentile(np.abs(image), 99) if cmap == 'coolwarm' else None
                axis.imshow(
                    np.rot90(image), cmap=cmap,
                    vmin=-limit if limit else None, vmax=limit if limit else None,
                )
                axis.set_title(title)
            for axis in axes.ravel():
                if not axis.has_data() or axis not in [axes[0, 2], axes[0, 3]]:
                    axis.axis('off')
            plt.tight_layout()
            plt.show()

            coverage = pd.DataFrame({
                'indicador': [
                    'Cohorte solicitada', 'Features extraídas', 'Fallos',
                    'Normal en mapas', 'Patológica en mapas', 'Flags de registro',
                    'Fallbacks GPU a CPU', 'Segundos medios por registro',
                    'Fondos QC válidos', 'Piso de fondo aplicado',
                ],
                'valor': [
                    len(cohort), len(features), len(failures), n0, n1,
                    int(registration_qc['registration_qc_flag'].sum()),
                    gpu_fallbacks, mean_registration_seconds,
                    background_valid, background_floored,
                ],
            })
            display(coverage.style.hide(axis='index'))
            display(Markdown('### Casos más sensibles a máscara/registro'))
            stability_columns = [
                'uid', 'is_pathologic', 'stability_threshold_span',
                'stability_mask_span', 'stability_translation_span',
                'stability_background_span',
            ]
            display(
                features.assign(
                    stability_max=features[[
                        'stability_threshold_span', 'stability_mask_span',
                        'stability_translation_span', 'stability_background_span',
                    ]].max(axis=1)
                ).nlargest(20, 'stability_max')[stability_columns + ['stability_max']]
                .style.hide(axis='index')
            )
            cleanup_registration_cache()
            '''
        ),
        md(
            """
            ## 8. Contrato de salida

            - `dat_radiomics_features_full.csv`: predictores candidatos por bloques.
            - `registration_qc_full.csv`: QC y dominio; no es predictor biológico por defecto.
            - `cohort_maps_full.npz`: medias, diferencia, Cohen d y Hedges g online.
            - `registered_crops/*.npz`: crops normalizados para una futura rama de imagen.
            - `registration_failures_full.csv`: denominador explícito y reintentos trazables.

            El notebook no elimina casos ni declara biomarcadores clínicamente validados.
            """
        ),
    ]
)


NOTEBOOK_05 = notebook(
    [
        md(
            """
            # 05 · Representación por bloques, anomalías y validación agrupada

            Usa los artefactos full-cohort del 03 y 04. La representación biológica se construye por
            bloques —semi-cuantificación, primer orden, forma y textura— después de ajustar cada feature
            dentro de familia de adquisición. Las variables técnicas se reservan para QC, dominio y una
            auditoría explícita de shortcuts.

            > PCA, t-SNE y anomalías son herramientas de revisión. No son predictores finales, subtipos
            > clínicos ni probabilidades de patología.
            """
        ),
        md(
            """
            ## 1. Carga, linaje y separación de roles

            El notebook exige cobertura full-cohort. No combina percentiles crudos de intensidad con
            biomarcadores. Los casos con flags de registro se conservan y se muestran por separado.
            """
        ),
        code(
            '''
            from __future__ import annotations

            import hashlib
            import json
            import os
            import sys
            import time
            from pathlib import Path

            import ipywidgets as widgets
            import matplotlib.pyplot as plt
            import numpy as np
            import pandas as pd
            import seaborn as sns
            from IPython.display import Markdown, clear_output, display
            from sklearn.covariance import LedoitWolf
            from sklearn.decomposition import PCA
            from sklearn.ensemble import IsolationForest
            from sklearn.impute import SimpleImputer
            from sklearn.manifold import TSNE
            from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
            from sklearn.neighbors import NearestNeighbors
            from sklearn.pipeline import Pipeline
            from sklearn.preprocessing import RobustScaler

            PROJECT_ROOT = Path.cwd()
            if PROJECT_ROOT.name.lower() == 'notebooks':
                PROJECT_ROOT = PROJECT_ROOT.parent
            if str(PROJECT_ROOT) not in sys.path:
                sys.path.insert(0, str(PROJECT_ROOT))

            # Importar desde los módulos concretos evita depender de que un kernel
            # Jupyter conserve una versión antigua de modeling.__init__ en memoria.
            from modeling.embedding_search import run_embedding_search
            from modeling.optuna_models import FoldData, run_optuna_experiment

            PRIVATE_OUTPUT_DIR = PROJECT_ROOT / 'outputs' / 'private_eda'
            NODE4_PROFILE = 'v3'
            NODE4_OUTPUT_DIR = PRIVATE_OUTPUT_DIR / f'full_cohort_{NODE4_PROFILE}'
            NODE5_RUN_ID = os.environ.get('DAT_NODE5_RUN_ID', 'optuna_radiomics_v1')
            if not NODE5_RUN_ID or any(
                character not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-._'
                for character in NODE5_RUN_ID
            ):
                raise ValueError('DAT_NODE5_RUN_ID sólo admite letras, números, punto, guion y _.')
            NODE5_OUTPUT_DIR = PRIVATE_OUTPUT_DIR / 'node5_runs' / NODE5_RUN_ID
            EMBEDDING_RUN_ID = os.environ.get('DAT_EMBEDDING_RUN_ID', 'umap_tsne_v1')
            if not EMBEDDING_RUN_ID or any(
                character not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-._'
                for character in EMBEDDING_RUN_ID
            ):
                raise ValueError(
                    'DAT_EMBEDDING_RUN_ID sólo admite letras, números, punto, guion y _.'
                )
            EMBEDDING_SEARCH_DIR = (
                NODE5_OUTPUT_DIR / 'embedding_runs' / EMBEDDING_RUN_ID
            )
            QC_PATH = PRIVATE_OUTPUT_DIR / 'image_qc_manifest.csv'
            FEATURES_PATH = NODE4_OUTPUT_DIR / 'dat_radiomics_features_full.csv'
            REGISTRATION_PATH = NODE4_OUTPUT_DIR / 'registration_qc_full.csv'
            FAILURES_PATH = NODE4_OUTPUT_DIR / 'registration_failures_full.csv'
            UPSTREAM_CONFIG_PATH = NODE4_OUTPUT_DIR / 'registration_radiomics_full_config.json'
            CROPS_DIR = NODE4_OUTPUT_DIR / 'registered_crops'
            NODE5_CONFIG_PATH = NODE5_OUTPUT_DIR / 'node5_analysis_config.json'
            COVERAGE_PATH = NODE5_OUTPUT_DIR / 'node5_coverage_qc_full.csv'
            EMBEDDING_PATH = NODE5_OUTPUT_DIR / 'embedding_coordinates_full.csv'
            OUTLIER_PATH = NODE5_OUTPUT_DIR / 'outlier_scores_full.csv'
            OPTUNA_OUTPUT_DIR = NODE5_OUTPUT_DIR / 'optuna_models'
            OOF_PATH = OPTUNA_OUTPUT_DIR / 'optuna_oof_predictions.csv'
            METRICS_PATH = OPTUNA_OUTPUT_DIR / 'optuna_oof_metrics.csv'
            LABEL_AUDIT_PATH = NODE5_OUTPUT_DIR / 'label_audit_full.csv'
            PROTOCOL_DIAGNOSTICS_PATH = NODE5_OUTPUT_DIR / 'protocol_diagnostics_full.csv'
            OPTUNA_MODULE_PATH = PROJECT_ROOT / 'modeling' / 'optuna_models.py'

            RANDOM_SEED = 20260821
            MAX_MISSING_FRACTION = 0.20
            CORRELATION_THRESHOLD = 0.95
            MAX_EMBEDDING_ROWS = 1500
            NEIGHBORS_TO_SHOW = 5
            CPU_JOBS = max(1, (os.cpu_count() or 2) - 1)
            ISOLATION_TREES = 500
            TSNE_MAX_ITER = 1500
            MIN_FEATURE_COVERAGE = 0.995
            MAX_FAMILY_FAILURE_RATE = 0.10
            MIN_FAMILY_GATE_SIZE = 10
            MIN_BACKGROUND_VALID_FRACTION = 0.80
            OPTUNA_TRIALS_PER_MODEL = int(os.environ.get('DAT_OPTUNA_TRIALS', '50'))
            EMBEDDING_TRIALS_PER_METHOD = int(
                os.environ.get('DAT_EMBEDDING_TRIALS', '30')
            )
            EMBEDDING_SEEDS = (RANDOM_SEED, RANDOM_SEED + 1, RANDOM_SEED + 2)
            OPTUNA_MODELS = ('logistic', 'random_forest', 'xgboost')
            OPTUNA_FEATURE_SET = 'radiomics_only'
            PREFER_XGBOOST_GPU = True
            RESUME = True
            SAVE_PRIVATE_OUTPUTS = True

            for required_path in [
                QC_PATH, FEATURES_PATH, REGISTRATION_PATH, FAILURES_PATH,
                UPSTREAM_CONFIG_PATH, CROPS_DIR,
            ]:
                if not required_path.exists():
                    raise FileNotFoundError(required_path)

            NODE5_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

            sns.set_theme(style='whitegrid')
            display(Markdown(
                f'**Upstream nodo 04:** `{NODE4_PROFILE}` (sólo lectura) · '
                f'**experimento nodo 05:** `{NODE5_RUN_ID}` · '
                f'**Optuna:** `{OPTUNA_TRIALS_PER_MODEL}` ensayos completos/modelo · '
                f'**embeddings:** `{EMBEDDING_RUN_ID}`, '
                f'`{EMBEDDING_TRIALS_PER_METHOD}` ensayos/método · '
                f'**CPU:** `{CPU_JOBS}` hilos · **GPU:** XGBoost CUDA con fallback CPU.'
            ))
            '''
        ),
        code(
            '''
            qc = pd.read_csv(QC_PATH, dtype={'uid': 'string'})
            features = pd.read_csv(FEATURES_PATH, dtype={'uid': 'string'})
            registration = pd.read_csv(REGISTRATION_PATH, dtype={'uid': 'string'})
            failures = pd.read_csv(FAILURES_PATH, dtype={'uid': 'string'})
            upstream_config = json.loads(UPSTREAM_CONFIG_PATH.read_text(encoding='utf-8'))


            def replace_with_retry(
                temporary: Path, destination: Path, attempts: int = 10
            ) -> None:
                delay_seconds = 0.25
                for attempt in range(attempts):
                    try:
                        temporary.replace(destination)
                        return
                    except PermissionError:
                        if attempt == attempts - 1:
                            raise
                        time.sleep(delay_seconds)
                        delay_seconds = min(delay_seconds * 1.8, 3.0)


            def atomic_csv(frame: pd.DataFrame, destination: Path) -> None:
                temporary = destination.with_suffix('.csv.tmp')
                frame.to_csv(temporary, index=False)
                replace_with_retry(temporary, destination)


            node5_config = {
                'algorithm_version': 'node5_optuna_radiomics_v1',
                'node4_profile': NODE4_PROFILE,
                'node5_run_id': NODE5_RUN_ID,
                'upstream_config_hash': upstream_config.get('config_hash'),
                'optuna_module_hash': hashlib.sha256(
                    OPTUNA_MODULE_PATH.read_bytes()
                ).hexdigest(),
                'cohort_uid_hash': hashlib.sha256(
                    '|'.join(qc['uid'].astype(str)).encode('utf-8')
                ).hexdigest(),
                'max_missing_fraction': MAX_MISSING_FRACTION,
                'correlation_threshold': CORRELATION_THRESHOLD,
                'max_embedding_rows': MAX_EMBEDDING_ROWS,
                'isolation_trees': ISOLATION_TREES,
                'tsne_max_iter': TSNE_MAX_ITER,
                'min_feature_coverage': MIN_FEATURE_COVERAGE,
                'max_family_failure_rate': MAX_FAMILY_FAILURE_RATE,
                'min_family_gate_size': MIN_FAMILY_GATE_SIZE,
                'min_background_valid_fraction': MIN_BACKGROUND_VALID_FRACTION,
                'optuna_trials_per_model': OPTUNA_TRIALS_PER_MODEL,
                'optuna_models': OPTUNA_MODELS,
                'optuna_feature_set': OPTUNA_FEATURE_SET,
                'prefer_xgboost_gpu': PREFER_XGBOOST_GPU,
                'random_seed': RANDOM_SEED,
            }
            node5_hash = hashlib.sha256(
                json.dumps(node5_config, sort_keys=True).encode('utf-8')
            ).hexdigest()
            node5_config['config_hash'] = node5_hash
            if NODE5_CONFIG_PATH.exists() and RESUME:
                previous_node5_config = json.loads(
                    NODE5_CONFIG_PATH.read_text(encoding='utf-8')
                )
                if previous_node5_config.get('config_hash') != node5_hash:
                    raise RuntimeError(
                        f'El experimento del nodo 5 `{NODE5_RUN_ID}` pertenece a otra '
                        'configuración. Usa un DAT_NODE5_RUN_ID nuevo; no es necesario '
                        'mover ni volver a ejecutar el nodo 04.'
                    )
            else:
                temporary_config = NODE5_CONFIG_PATH.with_suffix('.json.tmp')
                temporary_config.write_text(
                    json.dumps(node5_config, indent=2), encoding='utf-8'
                )
                replace_with_retry(temporary_config, NODE5_CONFIG_PATH)

            features_for_merge = features.drop(
                columns=[column for column in ['is_pathologic', 'acquisition_family']
                         if column in features.columns]
            )
            registration_for_merge = registration.drop(
                columns=[column for column in ['is_pathologic', 'acquisition_family']
                         if column in registration.columns]
            )
            data = (
                qc.merge(features_for_merge, on='uid', how='left', validate='one_to_one')
                .merge(registration_for_merge, on='uid', how='left', validate='one_to_one')
            )
            if data['is_pathologic'].isna().any():
                raise ValueError('Hay exámenes sin etiqueta en la cohorte combinada.')
            data['is_pathologic'] = data['is_pathologic'].astype(int)
            data['acquisition_family'] = data['acquisition_family'].astype(str)
            data['feature_available'] = data['uid'].isin(features['uid'])
            data['registration_available'] = data['uid'].isin(registration['uid'])
            if 'background_qc_valid' not in data:
                raise ValueError('Falta background_qc_valid: vuelve a ejecutar el nodo 04 v3.')
            data['background_qc_valid'] = data['background_qc_valid'].fillna(False).astype(bool)
            data['background_floor_applied'] = data[
                'background_floor_applied'
            ].fillna(False).astype(bool)

            coverage_by_family = (
                data.groupby('acquisition_family', dropna=False)
                .agg(
                    n=('uid', 'size'),
                    features=('feature_available', 'sum'),
                    registrations=('registration_available', 'sum'),
                    background_valid=('background_qc_valid', 'sum'),
                )
                .reset_index()
            )
            coverage_by_family['failures'] = (
                coverage_by_family['n'] - coverage_by_family['features']
            )
            coverage_by_family['feature_coverage'] = (
                coverage_by_family['features'] / coverage_by_family['n']
            )
            coverage_by_family['failure_rate'] = (
                coverage_by_family['failures'] / coverage_by_family['n']
            )
            coverage_by_family['background_valid_fraction'] = np.divide(
                coverage_by_family['background_valid'],
                coverage_by_family['features'],
                out=np.zeros(len(coverage_by_family), dtype=float),
                where=coverage_by_family['features'].to_numpy() > 0,
            )
            atomic_csv(coverage_by_family, COVERAGE_PATH)

            feature_coverage = float(data['feature_available'].mean())
            background_valid_fraction = float(
                data.loc[data['feature_available'], 'background_qc_valid'].mean()
            )
            gated_families = coverage_by_family[
                coverage_by_family['n'] >= MIN_FAMILY_GATE_SIZE
            ]
            worst_family_failure = float(
                gated_families['failure_rate'].max() if len(gated_families) else 0
            )

            coverage = pd.DataFrame({
                'indicador': [
                    'QC notebook 03', 'Features notebook 04', 'Fallos nodo 04',
                    'Cobertura de features', 'Fondos QC válidos',
                    'Peor tasa de fallo familiar (n>=gate)',
                ],
                'valor': [
                    len(qc), len(features), len(failures), feature_coverage,
                    background_valid_fraction, worst_family_failure,
                ],
            })
            display(coverage.style.hide(axis='index'))
            display(Markdown('### Cobertura por familia de adquisición'))
            display(
                coverage_by_family.sort_values(
                    ['failure_rate', 'n'], ascending=[False, False]
                ).head(30).style.hide(axis='index')
            )
            if feature_coverage < MIN_FEATURE_COVERAGE:
                raise RuntimeError(
                    f'Cobertura de features {feature_coverage:.1%} < '
                    f'{MIN_FEATURE_COVERAGE:.1%}. Corrige/reanuda el nodo 04 antes de modelar.'
                )
            if worst_family_failure > MAX_FAMILY_FAILURE_RATE:
                raise RuntimeError(
                    f'Tasa máxima de fallo familiar {worst_family_failure:.1%} > '
                    f'{MAX_FAMILY_FAILURE_RATE:.1%}. El subconjunto no es representativo.'
                )

            biological_blocks = {
                'semiquant': [column for column in data if column.startswith('semiquant_')],
                'firstorder': [column for column in data if column.startswith('firstorder_ratio_')],
                'shape': [column for column in data if column.startswith('shape_active_')],
                'texture': [column for column in data if column.startswith('texture3d_')],
            }
            background_blocks_enabled = (
                background_valid_fraction >= MIN_BACKGROUND_VALID_FRACTION
            )
            if not background_blocks_enabled:
                biological_blocks['semiquant'] = []
                biological_blocks['firstorder'] = []
                display(Markdown(
                    '**Gate de fondo:** se excluyen semiquant y first-order porque sólo '
                    f'`{background_valid_fraction:.1%}` tiene fondo QC válido.'
                ))
            ratio_columns = [
                column
                for block in ['semiquant', 'firstorder']
                for column in biological_blocks[block]
            ]
            for column in ratio_columns:
                values = pd.to_numeric(data[column], errors='coerce')
                data[column] = np.sign(values) * np.log1p(np.abs(values))
            stability_columns = [
                column for column in data if column.startswith('stability_')
            ]
            qc_source_columns = set(qc.select_dtypes(include=np.number).columns)
            registration_numeric = set(
                registration_for_merge.select_dtypes(include=np.number).columns
            )
            technical_allowed_prefixes = (
                'shape_', 'spacing_', 'fov_', 'voxel_volume_', 'finite_', 'zero_',
                'positive_fraction', 'positive_entropy_', 'foreground_', 'high_uptake_',
                'largest_component_', 'gradient_', 'slice_corr_', 'global_geometry_',
                'within_protocol_', 'technical_outlier_', 'correlation_',
                'metric_final_', 'registration_qc_', 'background_',
            )
            technical_columns = sorted(
                column for column in (qc_source_columns | registration_numeric)
                if column in data.columns
                and column not in {'is_pathologic'}
                and column.startswith(technical_allowed_prefixes)
                and not column.startswith('shape_active_')
            )
            biological_columns = [
                column for columns in biological_blocks.values() for column in columns
            ]

            missing_fraction = data[biological_columns].replace(
                [np.inf, -np.inf], np.nan
            ).isna().mean()
            biological_columns = missing_fraction[
                missing_fraction <= MAX_MISSING_FRACTION
            ].index.tolist()
            biological_blocks = {
                name: [column for column in columns if column in biological_columns]
                for name, columns in biological_blocks.items()
            }
            if not biological_columns:
                raise ValueError('No quedaron features biológicas utilizables.')

            display(pd.DataFrame({
                'bloque': [*biological_blocks.keys(), 'technical_qc', 'stability_qc'],
                'n_features': [
                    *[len(columns) for columns in biological_blocks.values()],
                    len(technical_columns), len(stability_columns),
                ],
                'rol': [
                    *['predictor candidato'] * len(biological_blocks),
                    'QC/dominio', 'QC de robustez',
                ],
            }).style.hide(axis='index'))
            '''
        ),
        md(
            """
            ## 2. Armonización por protocolo y PCA balanceado por bloque

            Para cada variable se resta la mediana y se divide por IQR dentro de la familia de adquisición;
            familias pequeñas o desconocidas usan estadísticos globales. Después se elimina redundancia
            Spearman > 0,95 y cada bloque se divide por la raíz de su número de features, evitando que el
            bloque más ancho domine el PCA.
            """
        ),
        code(
            '''
            def fit_protocol_statistics(
                frame: pd.DataFrame,
                columns: list[str],
                groups: pd.Series,
                min_group_size: int = 5,
            ) -> dict[str, object]:
                numeric = frame[columns].replace([np.inf, -np.inf], np.nan).astype(float)
                global_median = numeric.median()
                global_iqr = (numeric.quantile(0.75) - numeric.quantile(0.25)).replace(0, 1)
                by_group: dict[str, tuple[pd.Series, pd.Series]] = {}
                for group in sorted(groups.astype(str).unique()):
                    indices = groups.index[groups.astype(str) == group]
                    if len(indices) < min_group_size:
                        continue
                    subset = numeric.loc[indices]
                    median = subset.median().fillna(global_median)
                    iqr = (
                        subset.quantile(0.75) - subset.quantile(0.25)
                    ).replace(0, np.nan).fillna(global_iqr)
                    by_group[group] = (median, iqr)
                return {
                    'columns': columns,
                    'global_median': global_median,
                    'global_iqr': global_iqr,
                    'by_group': by_group,
                }


            def apply_protocol_statistics(
                frame: pd.DataFrame,
                groups: pd.Series,
                statistics: dict[str, object],
            ) -> pd.DataFrame:
                columns = statistics['columns']
                numeric = frame[columns].replace([np.inf, -np.inf], np.nan).astype(float)
                result = pd.DataFrame(index=frame.index, columns=columns, dtype=float)
                global_median = statistics['global_median']
                global_iqr = statistics['global_iqr']
                for group in groups.astype(str).unique():
                    indices = groups.index[groups.astype(str) == group]
                    median, iqr = statistics['by_group'].get(
                        str(group), (global_median, global_iqr)
                    )
                    result.loc[indices] = (
                        numeric.loc[indices].fillna(median) - median
                    ) / iqr
                return result.clip(-8, 8).astype(float)


            def prune_correlated(
                frame: pd.DataFrame, columns: list[str], threshold: float
            ) -> tuple[list[str], list[str]]:
                kept: list[str] = []
                dropped: list[str] = []
                correlation = frame[columns].corr(method='spearman').abs().fillna(0)
                for column in columns:
                    if any(correlation.loc[column, previous] > threshold for previous in kept):
                        dropped.append(column)
                    else:
                        kept.append(column)
                return kept, dropped


            full_statistics = fit_protocol_statistics(
                data, biological_columns, data['acquisition_family']
            )
            adjusted_biology = apply_protocol_statistics(
                data, data['acquisition_family'], full_statistics
            )

            retained_blocks: dict[str, list[str]] = {}
            correlation_drops: list[dict[str, str]] = []
            block_matrices: list[np.ndarray] = []
            representation_names: list[str] = []
            for block_name, block_columns in biological_blocks.items():
                if not block_columns:
                    retained_blocks[block_name] = []
                    continue
                kept, dropped = prune_correlated(
                    adjusted_biology, block_columns, CORRELATION_THRESHOLD
                )
                retained_blocks[block_name] = kept
                correlation_drops.extend(
                    {'block': block_name, 'dropped': column} for column in dropped
                )
                scaler = RobustScaler(quantile_range=(10, 90))
                block = scaler.fit_transform(adjusted_biology[kept])
                block = np.clip(block, -8, 8) / np.sqrt(max(len(kept), 1))
                block_matrices.append(block)
                representation_names.extend(kept)

            X_biology = np.hstack(block_matrices)
            n_components = min(12, X_biology.shape[0] - 1, X_biology.shape[1])
            pca = PCA(n_components=n_components, random_state=RANDOM_SEED)
            pca_scores = pca.fit_transform(X_biology)
            embedding = data[['uid', 'is_pathologic', 'acquisition_family']].copy()
            for index in range(pca_scores.shape[1]):
                embedding[f'PC{index + 1}'] = pca_scores[:, index]

            loadings = pd.DataFrame(
                pca.components_.T,
                index=representation_names,
                columns=[f'PC{index + 1}' for index in range(n_components)],
            )
            top_loadings = pd.concat([
                loadings['PC1'].abs().nlargest(12).rename('abs_loading_PC1'),
                loadings['PC2'].abs().nlargest(12).rename('abs_loading_PC2'),
            ], axis=1).fillna(0)

            figure, axes = plt.subplots(1, 3, figsize=(18, 5))
            axes[0].plot(
                np.arange(1, n_components + 1),
                np.cumsum(pca.explained_variance_ratio_),
                marker='o',
            )
            axes[0].set(
                xlabel='Componentes', ylabel='Varianza acumulada',
                ylim=(0, 1.02), title='PCA biológico balanceado por bloque',
            )
            sns.scatterplot(
                data=embedding, x='PC1', y='PC2', hue='is_pathologic',
                alpha=0.75, ax=axes[1],
            )
            axes[1].set_title('PCA por etiqueta')
            sns.scatterplot(
                data=embedding, x='PC1', y='PC2', hue='acquisition_family',
                legend=False, alpha=0.75, ax=axes[2],
            )
            axes[2].set_title('PCA por familia de adquisición')
            plt.tight_layout()
            plt.show()
            display(Markdown('### Cargas principales'))
            display(top_loadings.sort_values(['abs_loading_PC1', 'abs_loading_PC2'], ascending=False))
            display(Markdown(
                f'**Features retenidas:** `{len(representation_names)}` · '
                f'**redundantes removidas:** `{len(correlation_drops)}`'
            ))
            '''
        ),
        md(
            """
            ## 3. Proyección no lineal y vecinos biológicos

            t-SNE se ajusta sobre PCA y, si la cohorte excede 1.500 casos, usa una muestra reproducible
            sólo para visualización. Los vecinos usan el espacio biológico ajustado por protocolo y se
            muestran desde los crops ya registrados del nodo 04, sin reabrir los NIfTI crudos.
            """
        ),
        code(
            '''
            # Re-ejecutar esta celda no debe apilar widgets/figuras idénticos.
            # El kernel conserva los objetos anteriores, por lo que cerramos el
            # widget anterior y limpiamos la salida visual antes de redibujar.
            if 'neighbor_output' in globals():
                neighbor_output.close()
            clear_output(wait=True)
            rng = np.random.default_rng(RANDOM_SEED)
            saved_embedding = None
            if RESUME and EMBEDDING_PATH.exists():
                candidate_embedding = pd.read_csv(
                    EMBEDDING_PATH, dtype={'uid': 'string'}
                )
                if (
                    len(candidate_embedding) == len(embedding)
                    and set(candidate_embedding['uid']) == set(embedding['uid'])
                    and {'NL1', 'NL2'} <= set(candidate_embedding.columns)
                ):
                    saved_embedding = candidate_embedding.set_index('uid')
            if saved_embedding is not None:
                embedding['NL1'] = embedding['uid'].map(saved_embedding['NL1'])
                embedding['NL2'] = embedding['uid'].map(saved_embedding['NL2'])
                embedding['projection_method'] = 't-SNE sobre PCA biológico'
                projection_indices = np.flatnonzero(embedding['NL1'].notna().to_numpy())
            else:
                if len(data) > MAX_EMBEDDING_ROWS:
                    projection_indices = np.sort(
                        rng.choice(len(data), size=MAX_EMBEDDING_ROWS, replace=False)
                    )
                else:
                    projection_indices = np.arange(len(data))
                projection_input = pca_scores[
                    projection_indices, :min(20, pca_scores.shape[1])
                ]
                perplexity = max(5, min(40, (len(projection_indices) - 1) // 10))
                tsne = TSNE(
                    n_components=2, perplexity=perplexity, init='pca',
                    learning_rate='auto', random_state=RANDOM_SEED,
                    method='barnes_hut', angle=0.6, max_iter=TSNE_MAX_ITER,
                    n_jobs=CPU_JOBS,
                )
                nonlinear = tsne.fit_transform(projection_input)
                embedding['NL1'] = np.nan
                embedding['NL2'] = np.nan
                embedding.loc[projection_indices, 'NL1'] = nonlinear[:, 0]
                embedding.loc[projection_indices, 'NL2'] = nonlinear[:, 1]
                embedding['projection_method'] = 't-SNE sobre PCA biológico'
                if SAVE_PRIVATE_OUTPUTS:
                    atomic_csv(embedding, EMBEDDING_PATH)

            plot_frame = embedding.loc[projection_indices].merge(
                data[['uid', 'spacing_x_mm', 'technical_outlier_score']],
                on='uid', how='left',
            )
            figure, axes = plt.subplots(1, 3, figsize=(18, 5))
            sns.scatterplot(
                data=plot_frame, x='NL1', y='NL2', hue='is_pathologic',
                alpha=0.75, ax=axes[0],
            )
            axes[0].set_title('t-SNE · etiqueta')
            sns.scatterplot(
                data=plot_frame, x='NL1', y='NL2', hue='spacing_x_mm',
                palette='viridis', alpha=0.75, ax=axes[1],
            )
            axes[1].set_title('t-SNE · spacing')
            sns.scatterplot(
                data=plot_frame, x='NL1', y='NL2', hue='technical_outlier_score',
                palette='magma', alpha=0.75, ax=axes[2],
            )
            axes[2].set_title('t-SNE · QC técnico')
            plt.tight_layout()
            plt.show()

            neighbor_model = NearestNeighbors(
                metric='euclidean', n_neighbors=min(NEIGHBORS_TO_SHOW + 1, len(data))
            )
            neighbor_model.fit(X_biology)
            distances, indices = neighbor_model.kneighbors(X_biology)
            row_by_uid = {str(uid): index for index, uid in enumerate(data['uid'].astype(str))}


            def load_mip(uid: str) -> np.ndarray:
                crop_path = CROPS_DIR / f'{uid}.npz'
                if not crop_path.exists():
                    raise FileNotFoundError(
                        f'Falta el crop registrado para {uid}: {crop_path}'
                    )
                with np.load(crop_path, allow_pickle=False) as crop:
                    volume = np.asarray(crop['volume_intensity01'], dtype=np.float32)
                values = volume[np.isfinite(volume)]
                low, high = np.percentile(values, [1, 99.5])
                volume01 = np.clip((volume - low) / max(high - low, 1e-8), 0, 1)
                return np.rot90(volume01.max(axis=0))


            def show_neighbors(uid: str) -> None:
                query_index = row_by_uid[uid]
                neighbor_indices = indices[query_index]
                neighbor_distances = distances[query_index]
                figure, axes = plt.subplots(
                    1, len(neighbor_indices), figsize=(3.2 * len(neighbor_indices), 3.5)
                )
                axes = np.atleast_1d(axes)
                rows = []
                for rank, (axis, neighbor_index, distance) in enumerate(
                    zip(axes, neighbor_indices, neighbor_distances)
                ):
                    neighbor_uid = str(data.iloc[neighbor_index]['uid'])
                    label = int(data.iloc[neighbor_index]['is_pathologic'])
                    family = str(data.iloc[neighbor_index]['acquisition_family'])
                    axis.imshow(load_mip(neighbor_uid), cmap='hot', vmin=0, vmax=1)
                    axis.set_title(
                        f'{rank}. {neighbor_uid}\\ny={label} · {family} · d={distance:.2f}'
                    )
                    axis.axis('off')
                    rows.append({
                        'rank': rank, 'uid': neighbor_uid, 'is_pathologic': label,
                        'acquisition_family': family, 'distance': distance,
                    })
                plt.tight_layout()
                plt.show()
                display(pd.DataFrame(rows).style.hide(axis='index'))


            uid_selector = widgets.Dropdown(
                options=sorted(row_by_uid), value=sorted(row_by_uid)[0], description='UID:',
                layout=widgets.Layout(width='420px'),
                style={'description_width': '60px'},
            )
            neighbor_output = widgets.interactive_output(show_neighbors, {'uid': uid_selector})
            display(widgets.VBox([uid_selector, neighbor_output]))
            '''
        ),
        md(
            """
            ## 3.1 Comparación robusta t-SNE / UMAP

            La búsqueda varía los hiperparámetros de ambos métodos y repite cada ensayo con tres
            semillas. Se reportan dos selecciones: **estructura**, que prioriza fidelidad y estabilidad
            penalizando agrupamiento por adquisición/spacing, y **balanceada**, que además considera la
            separación de etiquetas. Las etiquetas sólo evalúan una proyección ya ajustada: nunca se
            entregan a t-SNE ni a UMAP. Por ello la selección balanceada sigue siendo exploratoria y no
            reemplaza las métricas OOF del clasificador.
            """
        ),
        code(
            '''
            baseline_coordinates = embedding[['NL1', 'NL2']].to_numpy(dtype=float)
            if not np.isfinite(baseline_coordinates).all():
                baseline_coordinates = None

            embedding_search_result = run_embedding_search(
                pca_scores,
                y=data['is_pathologic'].to_numpy(dtype=int),
                groups=data['acquisition_family'].astype(str).to_numpy(),
                spacing=data['spacing_x_mm'].to_numpy(dtype=float),
                uids=data['uid'].astype(str).to_numpy(),
                output_dir=EMBEDDING_SEARCH_DIR,
                experiment_config={
                    'node5_run_id': NODE5_RUN_ID,
                    'embedding_run_id': EMBEDDING_RUN_ID,
                    'source': str(EMBEDDING_PATH),
                    'pca_columns': [
                        f'PC{index + 1}' for index in range(pca_scores.shape[1])
                    ],
                },
                n_trials_per_method=EMBEDDING_TRIALS_PER_METHOD,
                seeds=EMBEDDING_SEEDS,
                cpu_jobs=CPU_JOBS,
                evaluation_rows=min(800, len(data)),
                evaluation_neighbors=15,
                baseline_coordinates=baseline_coordinates,
            )
            comparison_columns = [
                'method', 'selection', 'label_silhouette',
                'label_knn_balanced_accuracy', 'neighborhood_trustworthiness',
                'seed_neighborhood_stability', 'acquisition_family_confound',
                'spacing_confound', 'balanced_score',
                'labels_used_to_select_trial',
            ]
            display(Markdown('### Comparación cuantitativa de proyecciones'))
            display(
                embedding_search_result.comparison[comparison_columns]
                .style.format({
                    'label_silhouette': '{:.3f}',
                    'label_knn_balanced_accuracy': '{:.3f}',
                    'neighborhood_trustworthiness': '{:.3f}',
                    'seed_neighborhood_stability': '{:.3f}',
                    'acquisition_family_confound': '{:.3f}',
                    'spacing_confound': '{:.3f}',
                    'balanced_score': '{:.3f}',
                })
                .hide(axis='index')
            )

            optimized_plot = embedding_search_result.coordinates.merge(
                data[['uid', 'technical_outlier_score']],
                on='uid', how='left', validate='one_to_one',
            )
            figure, axes = plt.subplots(2, 3, figsize=(18, 10))
            for row, method in enumerate(('tsne', 'umap')):
                x_column = f'{method}_balanced_1'
                y_column = f'{method}_balanced_2'
                sns.scatterplot(
                    data=optimized_plot, x=x_column, y=y_column,
                    hue='is_pathologic', alpha=0.72, s=25, ax=axes[row, 0],
                )
                axes[row, 0].set_title(f'{method.upper()} optimizado · etiqueta')
                sns.scatterplot(
                    data=optimized_plot, x=x_column, y=y_column,
                    hue='spacing_x_mm', palette='viridis',
                    alpha=0.72, s=25, ax=axes[row, 1],
                )
                axes[row, 1].set_title(f'{method.upper()} optimizado · spacing')
                sns.scatterplot(
                    data=optimized_plot, x=x_column, y=y_column,
                    hue='technical_outlier_score', palette='magma',
                    alpha=0.72, s=25, ax=axes[row, 2],
                )
                axes[row, 2].set_title(f'{method.upper()} optimizado · QC técnico')
            plt.tight_layout()
            plt.show()

            display(Markdown(
                '**Lectura correcta:** una mejora sólo es convincente si aumenta la separación '
                'sin perder trustworthiness/estabilidad ni aumentar la confusión por familia o '
                'spacing. La proyección balanceada fue seleccionada mirando la etiqueta y, por '
                'tanto, no constituye validación fuera de muestra.'
            ))
            '''
        ),
        md(
            """
            ## 4. Anomalías separadas por rol

            `technical_anomaly` se ajusta sólo con QC/adquisición. `biological_anomaly` combina Isolation
            Forest, error de reconstrucción PCA y Mahalanobis regularizada en el espacio biológico. Sus
            percentiles priorizan revisión; no se usan como features del modelo.
            """
        ),
        code(
            '''
            technical_preprocessor = Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', RobustScaler(quantile_range=(10, 90))),
            ])
            X_technical = technical_preprocessor.fit_transform(
                data[technical_columns].replace([np.inf, -np.inf], np.nan)
            )
            technical_isolation = IsolationForest(
                n_estimators=ISOLATION_TREES, max_samples=min(512, len(data)),
                contamination='auto', random_state=RANDOM_SEED,
                n_jobs=CPU_JOBS,
            )
            technical_score = -technical_isolation.fit(X_technical).score_samples(X_technical)

            biological_isolation = IsolationForest(
                n_estimators=ISOLATION_TREES, max_samples=min(512, len(data)),
                contamination='auto', random_state=RANDOM_SEED,
                n_jobs=CPU_JOBS,
            )
            biological_isolation_score = -biological_isolation.fit(X_biology).score_samples(
                X_biology
            )
            cumulative = np.cumsum(pca.explained_variance_ratio_)
            components_90 = min(
                int(np.searchsorted(cumulative, 0.90) + 1), pca_scores.shape[1]
            )
            keep_components = max(1, min(components_90, pca_scores.shape[1] - 1))
            reduced_scores = np.zeros_like(pca_scores)
            reduced_scores[:, :keep_components] = pca_scores[:, :keep_components]
            reconstructed = pca.inverse_transform(reduced_scores)
            pca_error = np.mean((X_biology - reconstructed) ** 2, axis=1)
            covariance_components = min(8, pca_scores.shape[1], max(1, len(data) - 2))
            covariance = LedoitWolf().fit(pca_scores[:, :covariance_components])
            mahalanobis = covariance.mahalanobis(pca_scores[:, :covariance_components])

            outliers = data[[
                'uid', 'is_pathologic', 'acquisition_family', 'technical_outlier_score',
                'feature_available', 'background_qc_valid', 'background_floor_applied',
            ]].copy()
            outliers['technical_isolation_score'] = technical_score
            outliers['biological_isolation_score'] = biological_isolation_score
            outliers['pca_reconstruction_error'] = pca_error
            outliers['biological_mahalanobis'] = mahalanobis
            outliers['registration_qc_flag'] = data['registration_qc_flag'].fillna(True).astype(bool)
            outliers['technical_anomaly_rank'] = outliers[
                'technical_isolation_score'
            ].rank(pct=True)
            biological_ranks = []
            for column in [
                'biological_isolation_score', 'pca_reconstruction_error',
                'biological_mahalanobis',
            ]:
                rank_column = f'{column}_rank'
                outliers[rank_column] = outliers[column].rank(pct=True)
                biological_ranks.append(rank_column)
            outliers['biological_anomaly_rank'] = outliers[biological_ranks].mean(axis=1)
            outliers['registration_risk_rank'] = (
                1 - data['correlation_after'].rank(pct=True)
            ).fillna(1.0)
            outliers['background_risk'] = (
                (~outliers['background_qc_valid'])
                | outliers['background_floor_applied']
            ).astype(float)
            outliers['review_score'] = outliers[[
                'technical_anomaly_rank', 'biological_anomaly_rank',
                'registration_risk_rank', 'background_risk',
            ]].mean(axis=1)
            outliers = outliers.sort_values('review_score', ascending=False)
            if SAVE_PRIVATE_OUTPUTS:
                atomic_csv(outliers, OUTLIER_PATH)
            display(outliers.head(25).style.format({
                'technical_anomaly_rank': '{:.1%}',
                'biological_anomaly_rank': '{:.1%}',
                'registration_risk_rank': '{:.1%}',
                'review_score': '{:.1%}',
            }).hide(axis='index'))
            '''
        ),
        md(
            """
            ## 5. Validación agrupada y optimización bayesiana

            Los resultados previos fijan `radiomics_only` (forma + textura) como representación candidata:
            superó a las variables técnicas, mientras semiquantificación y first-order no pasan el gate de
            fondo. La armonización continúa ajustándose sólo con el fold de entrenamiento. Optuna ejecuta
            50 ensayos completos para regresión logística, random forest y XGBoost; cada estudio y cada
            fold OOF persisten por separado, por lo que una interrupción no obliga a repetirlos.
            """
        ),
        code(
            '''
            y = data['is_pathologic'].to_numpy()
            groups = data['acquisition_family'].astype(str).to_numpy()
            unique_groups = np.unique(groups)
            n_splits = min(5, len(unique_groups), int(pd.Series(y).value_counts().min()))
            split_method = 'StratifiedGroupKFold por acquisition_family'
            if n_splits >= 2:
                splitter = StratifiedGroupKFold(
                    n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED
                )
                candidate_splits = list(splitter.split(data, y, groups))
                valid_group_splits = all(len(np.unique(y[train])) == 2 for train, _ in candidate_splits)
            else:
                valid_group_splits = False
            if not valid_group_splits:
                n_splits = min(5, int(pd.Series(y).value_counts().min()))
                if n_splits < 2:
                    raise ValueError('No hay suficientes casos por clase para validación cruzada.')
                splitter = StratifiedKFold(
                    n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED
                )
                candidate_splits = list(splitter.split(data, y))
                split_method = 'StratifiedKFold fallback; revisar grupos insuficientes'

            radiomics_columns = retained_blocks['shape'] + retained_blocks['texture']
            semiquant_columns = retained_blocks['semiquant']
            biomarkers_all_columns = [
                column for columns in retained_blocks.values() for column in columns
            ]
            feature_sets = {
                'technical_only': {'biology': [], 'technical': technical_columns},
                'semiquant_only': {'biology': semiquant_columns, 'technical': []},
                'radiomics_only': {'biology': radiomics_columns, 'technical': []},
                'biomarkers_all': {'biology': biomarkers_all_columns, 'technical': []},
                'biomarkers_plus_technical': {
                    'biology': biomarkers_all_columns, 'technical': technical_columns,
                },
            }


            def fold_matrix(
                train_indices: np.ndarray,
                test_indices: np.ndarray,
                biology_columns: list[str],
                technical_columns_fold: list[str],
            ) -> tuple[np.ndarray, np.ndarray]:
                train_parts: list[np.ndarray] = []
                test_parts: list[np.ndarray] = []
                if biology_columns:
                    train_frame = data.iloc[train_indices]
                    test_frame = data.iloc[test_indices]
                    statistics = fit_protocol_statistics(
                        train_frame,
                        biology_columns,
                        train_frame['acquisition_family'],
                    )
                    train_biology = apply_protocol_statistics(
                        train_frame, train_frame['acquisition_family'], statistics
                    )[biology_columns]
                    test_biology = apply_protocol_statistics(
                        test_frame, test_frame['acquisition_family'], statistics
                    )[biology_columns]
                    scaler = RobustScaler(quantile_range=(10, 90))
                    train_parts.append(scaler.fit_transform(train_biology))
                    test_parts.append(scaler.transform(test_biology))
                if technical_columns_fold:
                    imputer = SimpleImputer(strategy='median')
                    scaler = RobustScaler(quantile_range=(10, 90))
                    train_technical = imputer.fit_transform(
                        data.iloc[train_indices][technical_columns_fold].replace(
                            [np.inf, -np.inf], np.nan
                        )
                    )
                    test_technical = imputer.transform(
                        data.iloc[test_indices][technical_columns_fold].replace(
                            [np.inf, -np.inf], np.nan
                        )
                    )
                    train_parts.append(scaler.fit_transform(train_technical))
                    test_parts.append(scaler.transform(test_technical))
                return np.hstack(train_parts), np.hstack(test_parts)


            if OPTUNA_FEATURE_SET not in feature_sets:
                raise ValueError(f'Feature set Optuna desconocido: {OPTUNA_FEATURE_SET}')
            selected_roles = feature_sets[OPTUNA_FEATURE_SET]
            if not selected_roles['biology'] and not selected_roles['technical']:
                raise ValueError(f'Feature set vacío: {OPTUNA_FEATURE_SET}')

            optuna_folds: list[FoldData] = []
            fold_id = np.full(len(data), -1, dtype=int)
            for fold, (train_indices, test_indices) in enumerate(candidate_splits):
                X_train, X_test = fold_matrix(
                    train_indices,
                    test_indices,
                    selected_roles['biology'],
                    selected_roles['technical'],
                )
                optuna_folds.append(FoldData(
                    fold=fold,
                    train_indices=np.asarray(train_indices, dtype=int),
                    valid_indices=np.asarray(test_indices, dtype=int),
                    X_train=np.asarray(X_train, dtype=np.float32),
                    X_valid=np.asarray(X_test, dtype=np.float32),
                    y_train=np.asarray(y[train_indices], dtype=int),
                    y_valid=np.asarray(y[test_indices], dtype=int),
                ))
                fold_id[test_indices] = fold

            optuna_result = run_optuna_experiment(
                optuna_folds,
                uids=data['uid'].astype(str).tolist(),
                y=y,
                groups=groups,
                output_dir=OPTUNA_OUTPUT_DIR,
                experiment_config={
                    'node5_config_hash': node5_hash,
                    'feature_set': OPTUNA_FEATURE_SET,
                    'feature_names': (
                        selected_roles['biology'] + selected_roles['technical']
                    ),
                    'split_method': split_method,
                    'n_splits': n_splits,
                },
                n_trials_per_model=OPTUNA_TRIALS_PER_MODEL,
                models=OPTUNA_MODELS,
                random_seed=RANDOM_SEED,
                cpu_jobs=CPU_JOBS,
                prefer_gpu=PREFER_XGBOOST_GPU,
            )
            optuna_oof = optuna_result.oof_predictions.merge(
                pd.DataFrame({'uid': data['uid'].astype(str), 'fold': fold_id}),
                on='uid', how='left', validate='one_to_one',
            )
            optuna_metrics = optuna_result.metrics.assign(
                feature_set=OPTUNA_FEATURE_SET,
                n_features=(
                    len(selected_roles['biology']) + len(selected_roles['technical'])
                ),
                split_method=split_method,
            )
            display(Markdown(
                f'**Validación:** `{split_method}` · **folds:** `{n_splits}` · '
                f'**XGBoost:** `{optuna_result.xgboost_device}`'
            ))
            display(Markdown('### Mejor ensayo Optuna por clasificador'))
            display(optuna_result.tuning_summary.style.format({
                'best_cv_log_loss': '{:.4f}',
            }).hide(axis='index'))
            display(Markdown('### Predicciones OOF con los hiperparámetros seleccionados'))
            display(optuna_metrics.style.format({
                'oof_auc': '{:.3f}', 'oof_log_loss': '{:.3f}',
                'oof_brier': '{:.3f}', 'oof_balanced_accuracy_0_5': '{:.3f}',
            }).hide(axis='index'))
            display(Markdown(
                '> Estas métricas son exploratorias: los mismos folds participan en la '
                'selección de hiperparámetros. La estimación final requiere validación anidada '
                'o un holdout intacto.'
            ))

            best_classifier = str(
                optuna_metrics.nsmallest(1, 'oof_log_loss').iloc[0]['model']
            )
            selected_probability_column = (
                'p_ensemble_oof' if best_classifier == 'mean_ensemble'
                else f'p_{best_classifier}_oof'
            )
            p_mean = optuna_oof[selected_probability_column].to_numpy(dtype=float)
            model_probability_columns = [
                f'p_{model_name}_oof' for model_name in OPTUNA_MODELS
            ]
            observed_probability = np.where(y == 1, p_mean, 1 - p_mean)
            label_audit = data[[
                'uid', 'is_pathologic', 'acquisition_family',
                'registration_qc_flag', 'correlation_after',
                'feature_available', 'background_qc_valid',
                'background_floor_applied',
            ]].copy()
            label_audit['selected_feature_set'] = OPTUNA_FEATURE_SET
            label_audit['selected_classifier'] = best_classifier
            for probability_column in model_probability_columns:
                label_audit[probability_column] = optuna_oof[
                    probability_column
                ].to_numpy(dtype=float)
            label_audit['p_mean_oof'] = p_mean
            label_audit['predictive_entropy'] = -(
                p_mean * np.log(p_mean) + (1 - p_mean) * np.log(1 - p_mean)
            )
            label_audit['model_disagreement'] = optuna_oof[
                model_probability_columns
            ].std(axis=1).to_numpy(dtype=float)
            label_audit['label_surprise'] = -np.log(observed_probability)
            label_audit = label_audit.merge(
                outliers[['uid', 'technical_anomaly_rank', 'biological_anomaly_rank', 'review_score']],
                on='uid', how='left', validate='one_to_one',
            )
            label_audit['expert_review_priority'] = (
                label_audit['label_surprise'].rank(pct=True)
                + label_audit['model_disagreement'].rank(pct=True)
                + label_audit['review_score'].rank(pct=True)
            ) / 3
            label_audit = label_audit.sort_values('expert_review_priority', ascending=False)
            display(Markdown(
                f'### Revisión experta · `{OPTUNA_FEATURE_SET}` + `{best_classifier}`'
            ))
            display(label_audit.head(25).style.format({
                **{column: '{:.3f}' for column in model_probability_columns},
                'p_mean_oof': '{:.3f}', 'expert_review_priority': '{:.1%}',
            }).hide(axis='index'))
            '''
        ),
        md(
            """
            ## 6. Diagnóstico por protocolo y persistencia

            Las métricas OOF son evidencia exploratoria del pipeline tabular, no una estimación final del
            modelo de imágenes. La promoción requiere repetir exactamente el mismo contrato sobre folds
            externos/holdout y agregar la rama de crops 3D mediante una ablación separada.
            """
        ),
        code(
            '''
            selected_oof = optuna_oof[[
                'uid', 'is_pathologic', 'acquisition_family', 'fold'
            ]].copy()
            selected_oof['p_mean_oof'] = p_mean
            protocol_diagnostics = (
                selected_oof.groupby('acquisition_family')
                .agg(
                    n=('uid', 'size'),
                    pathologic_rate=('is_pathologic', 'mean'),
                    mean_probability=('p_mean_oof', 'mean'),
                    mean_absolute_error=(
                        'p_mean_oof',
                        lambda values: float(np.mean(np.abs(
                            values.to_numpy()
                            - selected_oof.loc[values.index, 'is_pathologic'].to_numpy()
                        ))),
                    ),
                )
                .reset_index()
                .sort_values('n', ascending=False)
            )
            display(Markdown('### Diagnóstico por familia de adquisición'))
            display(protocol_diagnostics.head(30).style.hide(axis='index'))

            if SAVE_PRIVATE_OUTPUTS:
                atomic_csv(embedding, EMBEDDING_PATH)
                atomic_csv(outliers, OUTLIER_PATH)
                atomic_csv(label_audit, LABEL_AUDIT_PATH)
                atomic_csv(optuna_oof, OOF_PATH)
                atomic_csv(optuna_metrics, METRICS_PATH)
                atomic_csv(protocol_diagnostics, PROTOCOL_DIAGNOSTICS_PATH)
                embedding_config_path = NODE5_OUTPUT_DIR / 'embedding_full_config.json'
                temporary_embedding_config = embedding_config_path.with_suffix('.json.tmp')
                temporary_embedding_config.write_text(
                    json.dumps({
                        'algorithm_version': 'blockwise_optuna_v1',
                        'node4_profile': NODE4_PROFILE,
                        'node5_run_id': NODE5_RUN_ID,
                        'cpu_jobs': CPU_JOBS,
                        'isolation_trees': ISOLATION_TREES,
                        'tsne_max_iter': TSNE_MAX_ITER,
                        'embedding_run_id': EMBEDDING_RUN_ID,
                        'embedding_trials_per_method': EMBEDDING_TRIALS_PER_METHOD,
                        'embedding_search_dir': str(EMBEDDING_SEARCH_DIR),
                        'embedding_labels_used_for_fit': False,
                        'embedding_balanced_selection_uses_labels': True,
                        'neighbor_images_source': str(CROPS_DIR),
                        'background_blocks_enabled': background_blocks_enabled,
                        'feature_coverage': feature_coverage,
                        'background_valid_fraction': background_valid_fraction,
                        'biological_blocks_available': biological_blocks,
                        'retained_blocks_after_correlation_filter': retained_blocks,
                        'correlation_threshold': CORRELATION_THRESHOLD,
                        'protocol_adjustment': 'training-family median and IQR; global fallback',
                        'block_weighting': 'each block divided by sqrt(number of retained features)',
                        'pca_components': n_components,
                        'pca_components_for_90_percent': components_90,
                        'split_method': split_method,
                        'n_splits': n_splits,
                        'optuna_feature_set': OPTUNA_FEATURE_SET,
                        'best_classifier_exploratory': best_classifier,
                        'optuna_trials_per_model': OPTUNA_TRIALS_PER_MODEL,
                        'optuna_models': OPTUNA_MODELS,
                        'xgboost_device': optuna_result.xgboost_device,
                        'anomaly_scores_are_not_predictors': True,
                        'random_seed': RANDOM_SEED,
                    }, indent=2),
                    encoding='utf-8',
                )
                replace_with_retry(temporary_embedding_config, embedding_config_path)
                display(Markdown(
                    f'Artefactos del nodo 05 guardados en `{NODE5_OUTPUT_DIR}`. '
                    f'El nodo 04 permanece intacto en `{NODE4_OUTPUT_DIR}`.'
                ))
            '''
        ),
    ]
)


def main() -> None:
    NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)
    nbf.write(NOTEBOOK_03, NOTEBOOK_DIR / "03_eda_cohorte_3d_dat.ipynb")
    nbf.write(NOTEBOOK_04, NOTEBOOK_DIR / "04_registro_biomarcadores_radiomica.ipynb")
    nbf.write(NOTEBOOK_05, NOTEBOOK_DIR / "05_embeddings_outliers_3d.ipynb")


if __name__ == "__main__":
    main()
