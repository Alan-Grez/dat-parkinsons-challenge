# Nodo 10: contrato técnico y científico

## Objetivo

El nodo 10 prueba si la complementariedad observada entre los nodos 06 y 07 proviene de señales distintas, no de una CNN más grande. Todos los modelos usan los mismos pacientes y folds congelados; `acquisition_family` se reserva para construir y auditar los folds y está prohibida como predictor.

## Vistas y expertos

1. **CNN 3D dual multitarea.** Una rama recibe captación registrada truncada reconstruida como `volume_intensity01 × foreground_p99_registered`; el divisor de entrada se calcula sólo con el train del fold. La segunda rama recibe la proyección `sqrt(d) x / ||x||2`. Tres compuertas controlan intensidad, patrón y magnitud explícita. Cabezas auxiliares predicen seis regiones proxy, lado más afectado, asimetría y fragmentación.
2. **Regional HGB.** Usa 930 variables: intensidades/rangos, morfología multiumbral, texturas y relaciones. No son mediciones clínicas ni segmentaciones anatómicas validadas.
3. **Topología HGB.** Resume componentes, superficie, extensión y fracción activa en múltiples umbrales.
4. **Grafo controlado.** Elastic Net, Random Forest, MLP y GCN reciben exactamente los mismos 12 nodos (seis regiones proxy por dos bandas axiales). Sólo la GCN recibe aristas de continuidad axial, cadena caudado-putamen y homología bilateral.
5. **Diffusion Maps.** RobustScaler, PCA por varianza y Diffusion Maps con extensión Nyström se ajustan dentro de cada fold; una logística regularizada produce la probabilidad.
6. **Mezcla de subtipos.** RobustScaler y PCA fold-safe alimentan expertos pequeños con compuerta y penalizaciones de balance/diversidad.
7. **Fusión.** La media igual preespecificada es elegible para comparación OOF. El stacking Elastic Net se informa como exploratorio: para promoverlo se debe regenerar cada experto en un segundo nivel de CV estrictamente anidado.

## Validación y persistencia

- Búsqueda: tres folds balanceados por etiqueta y familia de adquisición.
- Cada estudio Optuna debe alcanzar al menos 10 trials `COMPLETE`; `PRUNED` no cuenta y se reemplaza.
- Final: cinco folds congelados y calibración cruzada fuera de fold.
- Auditoría final: bootstrap de 1000 remuestreos, peor familia con `n >= 10` y complementariedad pareada entre probabilidades OOF.
- Optuna persiste en SQLite con heartbeat y recuperación de trials abandonados.
- Redes: `last.pt`, `last.prev.pt`, `best.pt`, `history.csv`, preprocessor y predicciones del fold.
- Sklearn: modelo y transformer se escriben antes del CSV que actúa como marca de fold completo.
- Todo scaler, mediana, PCA, Diffusion Map y calibrador se ajusta únicamente en las particiones de entrenamiento correspondientes.

## Límites

- La intensidad reconstruida conserva diferencias de cuentas registradas hasta el p99, pero sigue siendo una unidad dependiente del protocolo; por eso requiere auditoría por familia y no debe interpretarse como SBR.
- Las regiones son proxies geométricos reproducibles en el espacio registrado, no un atlas clínico de caudado/putamen.
- Seleccionar hiperparámetros y comparar CV5 sobre la misma cohorte sigue siendo desarrollo interno; un submission o promoción requiere auditoría del runtime oficial y, cuando sea posible, repetición por semillas o validación externa.

## Ejecución

```powershell
.venv\Scripts\python.exe scripts\run_node10_hybrid.py --stage prepare
.venv\Scripts\python.exe scripts\run_node10_hybrid.py --stage search --device cuda
.venv\Scripts\python.exe scripts\run_node10_hybrid.py --stage final --device cuda
```

Repetir el mismo comando y `run_id` reanuda el trabajo compatible. Si cambia el contrato o la fuente del nodo 04, se debe usar un `run_id` nuevo en lugar de mezclar checkpoints.
