# Nodo 11: contrato confirmatorio HGB regional + CNN 2.5D

## Objetivo

Confirmar la complementariedad OOF detectada entre `regional_hgb` del nodo 10 y
`cnn_25d_image_only` del nodo 6. El estimando primario es el log loss OOF CV5 de
una mezcla preespecificada de probabilidades raw con peso `0.50 / 0.50`, seguida
por una única calibración de temperatura cross-fitted.

La curva completa de pesos es exploratoria. Su mínimo no se usa como score
confirmatorio ni como peso de despliegue.

## Particiones y leakage

- Los dos expertos comparten los mismos manifiestos de búsqueda CV3 y evaluación CV5.
- La semilla base es `20260902` y está congelada en la configuración del run.
- `acquisition_family` sólo balancea folds y alimenta auditorías post-OOF.
- El fold externo nunca decide una época o iteración.
- CNN selecciona `best_epoch` en dos folds internos del train externo; el refit
  usa la mediana de ambas selecciones y todo el train externo.
- HGB selecciona `best_iteration` con su validación interna y luego refitee usando
  todo el train externo.

## Optimización

Se ejecutan dos estudios Optuna locales y reanudables:

1. `node11_regional_hgb`.
2. `node11_cnn_25d_image_only`.

Cada estudio debe alcanzar al menos diez trials `COMPLETE`; los `PRUNED` no
cuentan. La poda sólo puede comenzar tras observar dos folds. SQLite, artefactos
por fold y checkpoints CNN atómicos permiten reanudar después de un corte.

La CNN tiene horizonte de 500 épocas, paciencia 80 y un learning rate menor que
el ganador original. El scheduler conserva `T_max=500` tanto durante selección
como durante el refit al `best_epoch`.

HGB tiene hasta 500 iteraciones y paciencia 60. Se guarda su curva interna de
train/validación y el modelo refitteado en la mejor iteración.

## Artefactos principales

Todos se escriben bajo
`outputs/private_eda/node11_runs/<run_id>/`:

- `config/search_common_3fold.csv` y `config/final_common_5fold.csv`.
- `search/optuna_node11.sqlite3`, tablas de trials y ganadores.
- `final/oof_regional_hgb.csv`.
- `final/oof_cnn_25d_image_only.csv`.
- `final/oof_fixed_raw_50_50.csv`.
- `final/final_metrics.csv` y `final/paired_bootstrap_differences.csv`.
- `final/cnn_epoch_selection.csv` y curvas por época dentro de cada checkpoint.
- `final/hgb_iteration_selection.csv` y curvas por iteración.
- `final/subgroup_metrics_by_acquisition_family.csv`.
- `final/deployment_manifest.json`.

## Reanudación

Volver a ejecutar la misma celda o invocar:

```powershell
.venv\Scripts\python.exe scripts\run_node11_confirmatory.py --stage search
.venv\Scripts\python.exe scripts\run_node11_confirmatory.py --stage final
```

No cambiar `run_id`, folds, semilla, espacio de búsqueda ni horizonte de
entrenamiento al reanudar. El número solicitado de trials completos sí puede
aumentarse con `--completed-trials`.

## Promoción

La mezcla se promueve sólo si mejora el HGB regional en log loss OOF, conserva
una diferencia bootstrap favorable, no degrada de forma material la peor familia
soportada y presenta curvas internas compatibles con convergencia estable. Antes
de un submission aún deben validarse inferencia, pesos y límite temporal dentro
del runtime oficial.
