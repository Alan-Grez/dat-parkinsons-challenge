# Entrega 01 — CNN 2.5D image-only

Esta carpeta concentra la primera entrega ejecutable para DrivenData. Usa el modelo
primario congelado por el nodo 06: cinco CNN 2.5D, una por fold, con calibración por
temperatura y promedio de probabilidades.

## Estructura

```text
entrega_01_cnn25d/
├── README.md
├── scripts/
│   ├── build_delivery.py
│   └── validate_delivery.py
├── submission_src/
│   ├── main.py
│   ├── assets/
│   │   ├── manifest.json
│   │   ├── registration_template.npz
│   │   └── fold_0.pt ... fold_4.pt
│   └── src/
│       ├── __init__.py
│       ├── model.py
│       ├── predictor.py
│       └── preprocessing.py
├── submission/
│   └── submission.zip
└── validation/
    ├── build_manifest.json
    ├── submission.sha256
    └── validation_report.json
```

El archivo que se sube a DrivenData es únicamente:

```text
entrega_01_cnn25d/submission/submission.zip
```

`main.py` está en la raíz interna del ZIP. En ejecución lee
`/code_execution/data/submission_format.csv` y los NIfTI de
`/code_execution/data/niftis/`, procesa cada examen independientemente y escribe
`/code_execution/submission.csv`.

## Reconstruir y validar

Desde la raíz del proyecto:

```powershell
uv run python entrega_01_cnn25d/scripts/build_delivery.py
uv run python entrega_01_cnn25d/scripts/validate_delivery.py
```

Para verificar localmente el pipeline completo sobre dos NIfTI temporales del archivo
privado (los archivos extraídos se eliminan automáticamente al terminar):

```powershell
uv run python entrega_01_cnn25d/scripts/validate_delivery.py --local-smoke-cases 2
```

Para una prueba end-to-end con un directorio compatible que contenga
`submission_format.csv` y `niftis/*.nii.gz`:

```powershell
uv run python entrega_01_cnn25d/scripts/validate_delivery.py --data-dir RUTA_AL_DATA_DIR
```

## Prueba oficial con Docker

El repositorio oficial está en `official-runtime/`. Copia el ZIP preparado a
`official-runtime/submission/submission.zip`, abre Docker y ejecuta desde ese directorio:

```powershell
just check-submission
just test-submission
```

Después de superar la prueba local, se recomienda subir primero como **smoke test** y
sólo después como submission completa.

## Contrato científico

- Modelo: `43765c3f64a985b1`, CNN 2.5D image-only.
- Cinco checkpoints congelados, uno por fold.
- Temperatura global: `1.3052315465492708`.
- Inferencia: calibrar el logit de cada fold y promediar sus cinco probabilidades.
- La familia de adquisición no se usa como entrada.
- No se incluye ningún NIfTI raw ni etiqueta de entrenamiento.
- La plantilla incluida es un activo derivado del entrenamiento requerido para reproducir
  el grid físico del modelo.
