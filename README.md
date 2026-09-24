# DaT Parkinson's Challenge

Entorno local para la competencia de DrivenData organizada por SFMN.

## Privacidad y reglas de datos

- Los datos de la competencia viven únicamente en `data/raw/`.
- `data/raw/` está excluido de Git y no debe sincronizarse, publicarse ni cargarse a servicios de IA.
- Codex no debe abrir, inspeccionar ni procesar los archivos de datos de la competencia.
- Los datos deben eliminarse al finalizar la competencia, salvo que exista otra licencia que autorice conservarlos.

## Entornos

- `.venv/`: entorno de trabajo local con Python 3.12 y dependencias modernas bloqueadas por `uv.lock`.
- `official-runtime/`: copia del runtime oficial para empaquetar y probar el envío en Docker.

## Inicio rápido

```powershell
uv sync
uv run jupyter lab
```

Para validar una entrega en condiciones equivalentes a DrivenData se debe usar el contenedor de `official-runtime/`, no sólo el entorno local.

## Exploración 3D

Los notebooks se ejecutan localmente y en este orden:

1. `notebooks/03_eda_cohorte_3d_dat.ipynb`: auditoría de toda la cohorte, gradiente físico por milímetro, familias de adquisición, QC dentro de protocolo y visor triplanar/MIP.
2. `notebooks/04_registro_biomarcadores_radiomica.ipynb`: registro rígido isotrópico full-cohort acelerado con CUDA, fallback trazable a SimpleITK/CPU, streaming, checkpoints, atlas opcional o consenso proxy, semicuantificación, forma, textura 3D, sensibilidad, mapas online y crops registrados.
3. `notebooks/05_embeddings_outliers_3d.ipynb`: representación biológica por bloques, ajuste por familia de adquisición, PCA balanceado, vecinos desde los crops registrados, anomalías técnicas/biológicas separadas y validación agrupada con paralelismo CPU.
4. `notebooks/06_cnn_25d_3d_fusion.ipynb`: seis modelos compactos 2.5D/3D (imagen, imagen+radiomics e imagen+radiomics+SBR condicionado), búsqueda Optuna de tres folds, evaluación congelada de cinco folds, ensamble, calibración y XAI 3D cuantitativo.
5. `notebooks/07_dat_spect_slab_multitemplate.ipynb`: receta DaT-SPECT específica con slab axial de 12 mm, self-normalization estriatal, augmentation físico fuerte, registro fold-safe contra varias plantillas, lateralidad canonizada, cuatro objetivos auxiliares y 930 variables regionales; compara slab 2D, 2.5D y 3D con las mismas tres ramas de fusión.

La explicación técnica y conceptual, celda por celda, del notebook 03 está en
`docs/notebook_03_eda_cohorte_3d.md`.
La documentación metodológica actualizada del notebook 04 está en
`docs/documentacion_notebook_04_registro_biomarcadores_radiomica.pdf`.

Las tablas derivadas se guardan en `outputs/private_eda/`, que está excluido de Git y debe tratarse como información privada de la competencia. Los notebooks se versionan sin ejecutar y sin outputs.

El nodo 06 no modifica el costoso nodo 04. Reutiliza sus crops en solo lectura, mantiene caches,
folds y experimentos en carpetas separadas, y reanuda preparación, trials, épocas, folds finales y
casos XAI confirmados. Para generar su notebook fuente y ejecutar una copia privada con visuales:

```powershell
uv run python scripts/materialize_cnn_notebook.py
uv run python scripts/run_cnn_notebook.py --run-id cnn_compact_v1
```

La copia con outputs queda en
`outputs/private_eda/cnn_runs/<run-id>/executed_notebooks/`. Si se interrumpe se conserva además una
copia `.partial.ipynb`; al repetir el mismo comando se reutilizan los checkpoints compatibles. Para
ejecutar sin notebook, use `scripts/run_cnn_experiments.py`. Un `run-id` nunca acepta una configuración
distinta ni derivados de otro estado del nodo 04.

El nodo 07 también consume sólo derivados privados del nodo 04 y mantiene separados el cache físico,
los experimentos y los splits congelados. La CV principal usa pacientes únicos, balancea etiqueta,
tamaño, fondo válido y distribución de familias; `acquisition_family` nunca es predictor. Se conserva
además un manifiesto de familias totalmente no vistas como prueba de estrés descriptiva. Una única
familia contiene 460 estudios, por lo que usarla como grupo indivisible hace imposible una CV de cinco
folds de tamaño comparable.

Para preparar/reanudar el pipeline desde terminal y ver su estado:

```powershell
uv run python scripts/run_node07_experiments.py --stage prepare --device cuda
uv run python scripts/run_node07_experiments.py --stage all --device cuda
uv run python scripts/run_node07_experiments.py --stage status
```

Para ejecutar el notebook y conservar sus outputs y figuras visibles:

```powershell
uv run python scripts/run_node07_notebook.py --run-id dat_spect_slab_v4
```

La copia ejecutada queda en
`outputs/private_eda/node07_runs/<run-id>/executed_notebooks/`. Los caches base, registros por fold,
features, bases SQLite de Optuna, checkpoints por época, OOF y modelos finales se reutilizan al repetir
el comando con el mismo contrato. Si cambia una decisión científica, se debe usar un `run-id` nuevo.

La validación integral usa únicamente NIfTI sintéticos temporales:

```powershell
uv run python scripts/validate_eda_notebooks_synthetic.py
```

Los artefactos de los notebooks 04 y 05 se guardan bajo
`outputs/private_eda/full_cohort_v3/`. Los nodos 04 y 05 comparten exclusivamente esa carpeta. Las
carpetas históricas `full_cohort/` y `full_cohort_gpu_v1/` no se mezclan ni se
sobrescriben. El registro es reanudable si la configuración no cambia. Si se
cambia referencia, atlas, spacing o contrato de features, se debe archivar esa carpeta antes de reiniciar;
el notebook bloquea automáticamente la mezcla de checkpoints incompatibles.

El nodo 04 usa la NVIDIA local mediante PyTorch para el registro rígido multirresolución. Si CUDA no
está disponible, ocurre un error de memoria o el resultado empeora el QC, el caso se repite con
SimpleITK/CPU y queda marcado en `gpu_fallback_used`. El ZIP se abre una sola vez y cada NIfTI se
extrae una sola vez a un caché temporal local durante la ejecución. El nodo 05 es tabular: usa todos
los núcleos CPU salvo uno, porque trasladar 1.362 filas de features a GPU no compensa el coste.

El fondo proxy v3 permanece dentro del soporte cerebral central, se intersecta con los valores positivos
de cada estudio y usa fallbacks trazables. Un piso relativo al percentil 90 del foreground evita razones
explosivas; `background_qc_valid` y `background_floor_applied` conservan la incertidumbre. Los mapas y
los crops modelables usan una normalización separada por el percentil 99 del foreground.

Para reconstruir los tres notebooks desde su fuente versionable y garantizar que no tengan outputs
embebidos:

```powershell
uv run python scripts/materialize_full_cohort_notebooks.py
```

Para ejecutarlos secuencialmente (`03` → `04` → `05`) sin sobrescribir los originales:

```powershell
uv run python scripts/run_full_cohort_notebooks.py
```

Se puede verificar primero el orden sin iniciar kernels con `--dry-run`. Las copias ejecutadas, con
tablas y visuales incrustados, quedan en `outputs/private_eda/full_cohort_v3/executed_notebooks/`. El ejecutor se
detiene ante el primer error y conserva una copia `.partial.ipynb`; al volver a lanzarlo, el notebook
04 reutiliza sus checkpoints compatibles. Un bloqueo exclusivo impide iniciar dos ejecutores a la vez
y evita que compitan por GPU o escriban simultáneamente el mismo checkpoint. Mientras una celda larga
sigue activa, la terminal imprime cada minuto el tiempo transcurrido y la cantidad de registros ya
confirmados por checkpoint. El nodo 04 confirma cada 20 estudios y fuerza un checkpoint al recibir
`Ctrl+C`; ante un apagado se repite como máximo el bloque no confirmado. El nodo 05 conserva t-SNE,
anomalías y predicciones OOF por fold: al reiniciar sólo recalcula la etapa o fold incompleto.

Para reanudar desde el notebook 04 sin repetir la auditoría completa del 03:

```powershell
uv run python scripts/run_full_cohort_notebooks.py --start-at 04
```

Para reanudar únicamente el nodo 05 desde sus checkpoints:

```powershell
uv run python scripts/run_full_cohort_notebooks.py --start-at 05
```

## Verificación

```powershell
uv run python scripts/verify_environment.py
```

Configuración local preparada:

- Python 3.12, alineado con el rango del runtime oficial.
- PyTorch 2.13 con CUDA 13.0 para Windows y la GPU NVIDIA local.
- Runtime oficial conservado en `official-runtime/`; su contenedor usa PyTorch 2.12.1 con CUDA 12.9 sobre Linux.
- Smoke test extraído en `official-runtime/data-demo/` sin modificar el archivo fuente.
- Versiones completas bloqueadas en `uv.lock`.

Docker Desktop y `just` son necesarios para ejecutar el contenedor oficial. No forman parte del entorno Python y deben instalarse antes de correr `just pull` o `just test-submission`.
