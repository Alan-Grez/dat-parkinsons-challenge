# Entrega 02 — ganador del nodo 09 (`slab2d`)

Esta carpeta contiene una entrega de código ejecutable para DrivenData construida con
el ganador final del nodo 09. La selección se hizo por **log loss OOF cross-calibrado
en cinco folds**, no por el mejor valor de la búsqueda CV3.

## Archivo que se sube

Sube únicamente:

```text
entrega_02_node09_slab2d/submission/submission.zip
```

`main.py` está en la raíz interna del ZIP. Durante la evaluación lee
`/code_execution/data/submission_format.csv` y los NIfTI de
`/code_execution/data/niftis/`, y escribe `/code_execution/submission.csv`.

## Modelo congelado

- Candidato: `555b81376733bb67`.
- Arquitectura: CNN residual `slab2d`, cinco modelos (uno por fold).
- Entrada de imagen: slab axial físico de 12 mm.
- Ramas adicionales: 32 features regionales seleccionadas dentro de cada fold y
  once variables SBR enmascaradas por validez del fondo.
- Registro: referencia rígida global y banco multitemplate específico de cada fold.
- Calibración: temperatura global `1.234489636989304` aplicada a cada logit antes
  de promediar las cinco probabilidades.
- Log loss OOF CV5 cross-calibrado: `0.5436546552477954`.
- `acquisition_family` no se usa como predictor.

## Estructura

```text
entrega_02_node09_slab2d/
├── README.md
├── scripts/
│   ├── build_delivery.py
│   └── validate_delivery.py
├── submission_src/
│   ├── main.py
│   ├── assets/
│   │   ├── manifest.json
│   │   ├── registration_template.npz
│   │   ├── fold_0.pt ... fold_4.pt
│   │   └── fold_0_templates.npz ... fold_4_templates.npz
│   ├── modeling/             # módulos mínimos congelados de entrenamiento
│   └── src/
│       ├── predictor.py
│       └── raw_registration.py
├── submission/
│   └── submission.zip
└── validation/
    ├── build_manifest.json
    ├── submission.sha256
    └── validation_report.json
```

No se incluyen NIfTI, etiquetas ni identificadores de pacientes de entrenamiento.
Las plantillas son activos agregados derivados del entrenamiento.

## Reconstrucción y validación

Desde la raíz del proyecto:

```powershell
& .venv/Scripts/python.exe entrega_02_node09_slab2d/scripts/build_delivery.py
& .venv/Scripts/python.exe entrega_02_node09_slab2d/scripts/validate_delivery.py
```

La validación comprueba el hash y la estructura del ZIP, la equivalencia exacta de
las cinco arquitecturas y una ejecución completa sobre los 20 exámenes del demo
oficial, preservando columnas y orden de UID.

## Secuencia recomendada de subida

1. Copiar el ZIP a `official-runtime/submission/submission.zip`.
2. Ejecutar `just check-submission` y `just test-submission` desde
   `official-runtime/` si Docker está disponible.
3. Subir primero como **smoke test** en DrivenData.
4. Sólo si el smoke test termina correctamente, subirlo como evaluación completa.

El runtime oficial exige Python 3.12, ejecución sin red y un máximo de tres horas;
el smoke test tiene un máximo de seis minutos.
