# Notebook 03: EDA de cohorte completa y control de calidad físico

## 1. Propósito y alcance

Este documento describe exclusivamente el notebook
`notebooks/03_eda_cohorte_3d_dat.ipynb`. Su función es auditar los 1.362 estudios
tridimensionales antes de registrarlos, extraer biomarcadores o entrenar un modelo.

El notebook responde cuatro preguntas:

1. ¿Cada archivo puede abrirse y contiene un volumen 3D numéricamente válido?
2. ¿Cuál es la geometría física de cada estudio?
3. ¿Qué características básicas presenta su señal y su organización espacial?
4. ¿Qué casos son técnicamente inusuales y deberían revisarse visualmente?

No diagnostica enfermedad, no segmenta estructuras anatómicas, no registra cerebros entre sí y no
elimina estudios automáticamente. Sus scores son herramientas de priorización para control de
calidad, no probabilidades clínicas.

### Flujo conceptual

```text
NIfTI originales
      │
      ├── lectura de cabecera y orientación
      ├── geometría física e intensidades robustas
      ├── proxies de organización espacial
      ├── familias técnicas de adquisición
      ├── scores robustos de atipicidad
      └── manifiesto, revisión visual y linaje
```

## 2. Terminología fundamental

### Cohorte completa o *full-cohort*

`Full-cohort` no es un término propio de este conjunto de datos ni una categoría de neurociencia.
Significa simplemente **procesar la cohorte completa**. En este proyecto corresponde a los 1.362
estudios disponibles. La configuración que lo activa es:

```python
MAX_SCANS: int | None = None
```

`None` indica que no se aplica un máximo ni se toma una muestra.

### NIfTI y datos `raw`

Cada `.nii.gz` es un archivo NIfTI que contiene:

- una matriz tridimensional de intensidades;
- dimensiones y tipo de dato;
- spacing o separación física entre vóxeles;
- matrices para relacionar índices de vóxel con coordenadas físicas;
- metadatos de orientación.

En el repositorio se consideran `raw` porque son los archivos originales entregados por la
competencia y el pipeline no los modifica. No son, sin embargo, mediciones crudas del detector: son
volúmenes 3D ya reconstruidos.

### QC y dominio técnico

`QC` significa *quality control* o control de calidad. Busca comportamientos técnicos inusuales:
geometrías diferentes, valores inválidos, fragmentación, ruido o incoherencia entre cortes.

Un **dominio técnico** es un conjunto producido bajo condiciones de adquisición parecidas. Puede
reflejar diferencias de scanner, centro, resolución, protocolo o reconstrucción. Como esos metadatos
no están necesariamente disponibles, el notebook aproxima el dominio mediante dimensiones y
spacing. Un dominio técnico no es una clase clínica.

### Vóxel, shape, spacing y FOV

Un vóxel es una posición de la matriz tridimensional. Si un volumen tiene:

```text
shape = 128 × 128 × 90
```

entonces contiene:

$$
128\times128\times90=1\,474\,560\text{ vóxeles}.
$$

El **spacing** es la separación física entre centros de vóxeles consecutivos. Se mide para cada eje:

```python
spacing_x_mm
spacing_y_mm
spacing_z_mm
```

La cobertura física o *field of view* (FOV) se aproxima en cada dimensión como:

$$
FOV_x=shape_x\times spacing_x.
$$

Análogamente se calculan $FOV_y$ y $FOV_z$. El volumen físico de un vóxel es:

$$
V_{vóxel}=spacing_x\times spacing_y\times spacing_z.
$$

### Proxy

Un proxy es una medición indirecta o aproximada. Por ejemplo, comparar la señal de las dos mitades
de la imagen puede sugerir asimetría, pero no equivale a segmentar y comparar los estriados derecho
e izquierdo. Los proxies son útiles para explorar y detectar anomalías; no deben interpretarse como
biomarcadores anatómicos validados.

## 3. Recorrido celda por celda

### Celda 1: presentación

Declara que se realizará un EDA de cohorte completa con control de calidad físico. EDA significa
*exploratory data analysis* o análisis exploratorio de datos.

La palabra “físico” indica que dimensiones, posiciones y gradientes se interpretan usando milímetros,
no solamente índices de una matriz.

### Celda 2: contrato metodológico

Documenta antes de ejecutar que:

- se procesará toda la cohorte;
- las fuentes permanecerán intactas;
- los resultados derivados serán privados;
- RAS homogenizará los ejes, pero no registrará anatómicamente los cerebros;
- las métricas serán exploratorias y no producirán exclusiones automáticas.

Este contrato evita atribuir al notebook capacidades que todavía no tiene.

### Celda 3: bibliotecas, rutas y configuración

Carga las bibliotecas y define las entradas y salidas:

```python
NIFTI_ARCHIVE = PROJECT_ROOT / "data" / "raw" / "niftis.zip"
LABELS_PATH = PROJECT_ROOT / "data" / "raw" / "train_labels.csv"
PRIVATE_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "private_eda"
```

Los parámetros son:

| Parámetro | Definición | Consecuencia |
|---|---|---|
| `MAX_SCANS=None` | Sin máximo de estudios | Procesa los 1.362 |
| `RANDOM_SEED=20260821` | Semilla reproducible | Sólo actuaría si se tomara una muestra |
| `MIN_PROTOCOL_FAMILY_SIZE=5` | Tamaño mínimo de una familia estable | Grupos menores usan referencia global |
| `SAVE_PRIVATE_OUTPUTS=True` | Activa persistencia | Guarda manifiestos y configuración |

La celda comprueba que existan el ZIP y las etiquetas. Si falta una entrada, detiene la ejecución en
lugar de generar resultados incompletos.

### Celda 4: identificación, lectura y orientación

#### Identificación de los archivos

`uid_from_member` extrae el UID desde el nombre del NIfTI. `list_nifti_members` enumera los `.nii` y
`.nii.gz` contenidos en el ZIP.

Si `MAX_SCANS` fuera menor que el total, se elegiría una muestra aleatoria sin reemplazo:

```python
rng = np.random.default_rng(RANDOM_SEED)
indices = np.sort(rng.choice(len(members), size=MAX_SCANS, replace=False))
```

Con `MAX_SCANS=None`, ese bloque no se ejecuta. El random no altera las imágenes ni sus intensidades.

#### Lectura de un volumen

Cada archivo se extrae a un directorio temporal, se carga y se elimina al salir de la función. Se
procesa un estudio a la vez para limitar el uso de memoria. Reabrir el ZIP en cada iteración aumenta
el costo de entrada/salida, pero conserva un flujo simple y acotado en RAM.

#### Orientación RAS

RAS significa:

- `R`: *Right*, hacia la derecha;
- `A`: *Anterior*, hacia el frente;
- `S`: *Superior*, hacia arriba.

```python
canonical = nib.as_closest_canonical(image)
```

Esta operación reordena o invierte ejes para que tengan un significado común. No alinea sujetos entre
sí ni cambia su resolución física. Su propósito es que “izquierda”, “derecha”, “anterior” y “superior”
no dependan de cómo cada archivo fue almacenado.

#### `qform`, `sform` y affine

Los índices $(x,y,z)$ sólo indican una posición dentro de la matriz. La matriz **affine** los
transforma a una posición física:

$$
\begin{bmatrix}X\\Y\\Z\\1\end{bmatrix}_{mm}
=
A
\begin{bmatrix}x\\y\\z\\1\end{bmatrix}_{vóxel}.
$$

NIfTI ofrece dos representaciones de esta relación:

- `qform`: orientación, escala y posición mediante una representación compacta basada en
  cuaterniones;
- `sform`: transformación affine más general.

`qform_code` y `sform_code` indican si cada transformación es válida y su significado previsto. El
notebook no decide cuál es clínicamente correcta; conserva ambos códigos para auditoría.

También registra el determinante de la parte espacial de la affine. Un valor casi cero puede indicar
una transformación singular o inválida.

### Celda 5: contrato del manifiesto

Introduce la construcción de una fila por estudio. La fila resume integridad, geometría, distribución
de intensidad, organización espacial y métricas de QC. El conjunto de filas será el manifiesto que
consumirán los notebooks posteriores.

### Celda 6: métricas por estudio

#### Integridad numérica

Se identifica qué vóxeles contienen números finitos:

```python
finite_mask = np.isfinite(volume)
```

Se calculan:

- `finite_fraction`: proporción finita;
- `zero_fraction`: proporción igual a cero;
- `positive_fraction`: proporción positiva.

Los ceros suelen representar fondo. Una fracción extrema no demuestra un error, pero puede revelar
una reconstrucción o un encuadre diferente.

#### Percentiles y normalización robusta

Sobre los valores positivos —o todos los finitos si no hay positivos— se calculan $p01$, $p05$,
$p50$, $p90$, $p95$ y $p99.5$. La normalización exploratoria es:

$$
I_{norm}=\operatorname{clip}\left(\frac{I-p01}{p99.5-p01},0,1\right).
$$

Los percentiles son menos sensibles a unos pocos valores extremos que el mínimo y el máximo.
Utilizar $p99.5$ evita que un único vóxel brillante reduzca el contraste efectivo del resto del
volumen. Esta normalización vive en memoria; el NIfTI original no se sobrescribe.

#### Geometría

Se guardan shape, spacing, FOV, volumen del vóxel y anisotropía. Esta última es:

$$
\text{anisotropía}=\frac{\max(spacing_x,spacing_y,spacing_z)}
{\min(spacing_x,spacing_y,spacing_z)}.
$$

Un valor 1 corresponde a vóxeles isotrópicos. Un valor 2 significa que la separación en alguna
dirección es dos veces la de otra.

#### Entropía

`safe_entropy` distribuye las intensidades en 64 intervalos y calcula:

$$
H=-\sum_i p_i\log_2(p_i).
$$

La entropía resume diversidad de intensidades. Es una medición técnica exploratoria, no entropía
anatómica ni biomarcador clínico.

#### Foreground y alta captación

Se define como foreground proxy la región sobre la mediana positiva:

```python
foreground = finite_mask & (volume > p50)
```

Su volumen se expresa en mililitros. No representa el volumen cerebral real porque no utiliza una
segmentación anatómica.

La alta captación se define sobre $p90$. Se cuentan sus componentes conectados y la fracción
perteneciente al mayor. Muchos componentes pequeños pueden indicar fragmentación o ruido; un
componente dominante indica señal espacialmente más concentrada. Como el umbral es relativo a cada
estudio, describe organización y no intensidad absoluta.

#### Centroide

El centro de masa se calcula ponderando los vóxeles del foreground por su intensidad normalizada. La
affine transforma el resultado a milímetros. Si el cálculo no es finito, se utiliza el centro
geométrico del volumen como respaldo.

#### Asimetría global proxy

Después de orientar a RAS, el volumen se divide por la mitad del eje X. Sean $S_D$ y $S_I$ las sumas
ponderadas de ambas mitades:

$$
AI_{abs}=\frac{|S_D-S_I|}{(S_D+S_I)/2},
$$

$$
AI_{signed}=\frac{S_D-S_I}{(S_D+S_I)/2}.
$$

La versión absoluta mide magnitud y la versión con signo conserva dirección. Puede alcanzar valores
próximos a 2 si toda la señal está en un lado. Sigue siendo un proxy global: puede responder a
asimetría biológica, posicionamiento, encuadre o artefactos.

#### Gradiente físico de intensidad

`np.gradient` aproxima las derivadas espaciales. Para un vóxel interior:

$$
\frac{\partial I}{\partial x}\approx
\frac{I(x+1,y,z)-I(x-1,y,z)}{2\,spacing_x},
$$

y análogamente para Y y Z. En los bordes utiliza diferencias de un solo lado.

La magnitud tridimensional es:

$$
|\nabla I|=\sqrt{
\left(\frac{\partial I}{\partial x}\right)^2+
\left(\frac{\partial I}{\partial y}\right)^2+
\left(\frac{\partial I}{\partial z}\right)^2}.
$$

Después se promedia dentro de los vóxeles sobre $p05$, para que el fondo negro no domine. La unidad
aproximada es intensidad normalizada por milímetro.

La columna histórica se llama `gradient_energy`, pero matemáticamente es la **media de la magnitud
del gradiente**, no energía cuadrática. `gradient_energy_per_mm` expresa mejor su contrato.

#### Correlación entre cortes

Se calcula la correlación de Pearson entre cada par de cortes Z consecutivos y se conserva la
mediana. Cortes constantes se omiten porque su correlación no está definida.

- Valores cercanos a 1: alta semejanza entre cortes vecinos.
- Valores menores: cambios más rápidos, ruido, movimiento o una resolución axial diferente.

La medida depende del protocolo; por eso después se compara preferentemente dentro de familias
técnicas.

#### Procesamiento y etiquetas

Cada archivo se procesa dentro de un `try/except`: un fallo queda registrado, pero no cancela toda la
cohorte. Al final se incorporan las etiquetas mediante UID con validación uno-a-uno. Las etiquetas no
intervienen en el cálculo de métricas ni en la creación de familias.

### Celda 7: motivación de las familias técnicas

Introduce el problema de comparar adquisiciones heterogéneas. Un volumen de $128\times128\times38$
con spacing de 3,895 mm no debe marcarse como anómalo sólo por diferir de uno de $256^3$ con spacing
de 2,46 mm.

### Celda 8: familias y scores robustos

#### Familias de adquisición

Cada firma contiene shape exacto y spacing redondeado a tres decimales:

```text
128x128x128|2.460,2.460,2.460
```

Cada firma recibe un identificador `AF001`, `AF002`, etc. “Familia de adquisición” es una
estratificación creada para este análisis, cercana a los conceptos de *batch*, dominio de scanner o
protocolo técnico. No es una categoría clínica oficial.

El redondeo a 0,001 mm evita separar archivos por ruido de representación numérica. No implica que el
scanner tenga una precisión real de una micra. La regla es deliberadamente trazable, aunque estricta:
no captura diferencias de scanner que no se reflejen en shape o spacing y puede fragmentar protocolos
parecidos.

#### Estandarización robusta

Para cada variable se calcula:

$$
z_{robusto}=\frac{x-\operatorname{mediana}(x)}{1.4826\times MAD(x)},
$$

donde:

$$
MAD=\operatorname{mediana}(|x-\operatorname{mediana}(x)|).
$$

En una distribución normal, $MAD\approx0.6745\sigma$ y
$1/0.6745\approx1.4826$. El factor calibra el MAD a la escala aproximada de una desviación estándar
normal. No obliga a que los datos sean normales: con distribuciones distintas, el resultado sigue
siendo una distancia robusta, pero no debe interpretarse literalmente como número de desviaciones
estándar gaussianas.

Los faltantes se imputan con la mediana. Si el MAD es cero, esa variable no aporta distancia porque no
existe dispersión robusta estimable en el grupo.

#### Score geométrico global

Se construye con cinco variables:

1. `log1p(voxel_volume_mm3)`;
2. `log1p(fov_volume_l)`;
3. anisotropía del spacing;
4. $shape_x/shape_y$;
5. $shape_z/shape_x$.

El logaritmo comprime variables positivas sesgadas. Tras obtener sus cinco z robustos:

$$
S_{geom}=\sqrt{\frac{z_1^2+z_2^2+z_3^2+z_4^2+z_5^2}{5}}.
$$

Es una distancia geométrica respecto del patrón global de la cohorte.

#### Score QC dentro del protocolo

Utiliza seis variables:

1. fracción de ceros;
2. entropía;
3. $\log(1+\text{componentes de alta captación})$;
4. fracción del componente más grande;
5. $\log(1+\text{gradiente por mm})$;
6. incoherencia axial $=1-\text{correlación entre cortes}$.

Para familias con al menos cinco estudios, los z robustos se calculan dentro de la familia. Las
familias menores usan la referencia global porque su mediana y MAD serían inestables. Luego:

$$
S_{QC}=\sqrt{\frac{z_1^2+z_2^2+\cdots+z_6^2}{6}}.
$$

#### Score técnico final

Ambos bloques reciben el mismo peso:

$$
S_{técnico}=\sqrt{\frac{S_{geom}^2+S_{QC}^2}{2}}.
$$

No existe un umbral clínicamente validado. Un score alto significa **técnicamente inusual**, no
“inválido” ni “patológico”. La tabla ordena los 20 valores mayores para revisión experta.

### Celda 9: integridad y gráficos de cohorte

La tabla de integridad informa:

- estudios auditados;
- UID duplicados;
- etiquetas ausentes;
- volúmenes con valores no finitos;
- spacing no positivo;
- affines casi singulares;
- número de familias y familias raras.

Los cuatro gráficos responden preguntas diferentes:

1. **Spacing X por clase:** ¿la etiqueta podría estar asociada al protocolo?
2. **FOV por familia:** ¿existen grupos geométricos definidos?
3. **Asimetría por clase:** ¿el proxy muestra una diferencia descriptiva?
4. **Gradiente y coherencia:** ¿hay estudios especialmente ruidosos o inconsistentes?

Colorear por etiqueta es descriptivo. No interviene en la construcción de los scores. Su utilidad es
detectar confusión técnica: un modelo podría aprender resolución o centro de adquisición en vez de
biología.

### Celda 10: contrato de revisión visual

Aclara que una medición atípica debe contrastarse con la imagen. El ranking selecciona casos para
inspeccionar; no reemplaza el juicio visual ni clínico.

### Celda 11: visor triplanar y MIP

El visor vuelve a cargar el UID elegido, usa $p01$ y $p99.5$ como ventana de visualización y localiza
un centro de masa aproximado. Presenta:

- sagital;
- coronal;
- axial;
- MIP X, Y y Z.

Una MIP (*maximum intensity projection*) conserva, para cada posición proyectada, el máximo encontrado
a lo largo del eje. Facilita ver la distribución global de captación, pero pierde profundidad.

El UID inicial es el caso con mayor score técnico. La ventana visual no modifica los valores usados en
el manifiesto. El selector requiere un kernel activo para cambiar de UID; una copia ejecutada conserva
el render inicial, pero no garantiza interactividad sin Jupyter y sus widgets.

### Celda 12: contrato de persistencia

Explica que los artefactos derivados son privados y que los NIfTI originales no serán sobrescritos.

### Celda 13: persistencia y linaje

Guarda:

| Artefacto | Contenido |
|---|---|
| `image_qc_manifest.csv` | Una fila de métricas por estudio |
| `image_qc_failures.csv` | Archivos que no pudieron procesarse |
| `acquisition_family_summary.csv` | Resumen de familias técnicas |
| `image_qc_config.json` | Parámetros y contratos utilizados |

El JSON registra orientación, definición del gradiente, construcción de familias y ausencia de
exclusiones automáticas. Esto permite reproducir y auditar el análisis.

## 4. Interpretación correcta

El notebook permite afirmar que un estudio:

- tiene cierta geometría y spacing;
- pertenece a una firma técnica observable;
- muestra un comportamiento más o menos inusual;
- merece o no una revisión visual prioritaria.

Por sí solo no permite afirmar que:

- una imagen sea clínicamente inválida;
- un score alto indique Parkinson;
- el foreground corresponda al cerebro completo;
- la división global izquierda/derecha mida asimetría estriatal;
- una familia técnica identifique inequívocamente un scanner o centro.

## 5. Supuestos y limitaciones

1. **RAS es necesario, pero insuficiente.** Homogeniza ejes; no registra sujetos.
2. **Shape y spacing son proxies de protocolo.** No sustituyen metadatos reales de scanner y centro.
3. **Los umbrales son relativos.** $p50$ y $p90$ facilitan comparabilidad interna, pero no tienen una
   interpretación fisiológica absoluta.
4. **La asimetría es global.** No utiliza regiones anatómicas validadas.
5. **Los scores usan igual ponderación.** Es una decisión exploratoria, no una calibración aprendida.
6. **El factor 1,4826 usa la normal como referencia de escala.** No demuestra normalidad.
7. **Familias pequeñas usan la cohorte global.** Su score QC puede estar más influido por diferencias
   de protocolo.
8. **Un score alto requiere inspección.** Nunca justifica exclusión automática.
9. **El procesamiento prioriza memoria sobre velocidad.** Abrir y extraer cada NIfTI por separado
   reduce RAM, pero aumenta el tiempo total.

## 6. Glosario breve

| Término | Significado en este notebook |
|---|---|
| Cohorte completa | Los 1.362 estudios |
| NIfTI | Contenedor de volumen 3D y metadatos espaciales |
| Vóxel | Elemento tridimensional de la matriz |
| Shape | Número de vóxeles por eje |
| Spacing | Separación física entre vóxeles, en mm |
| FOV | Cobertura física aproximada del volumen |
| RAS | Convención derecha-anterior-superior |
| Affine | Matriz que lleva índices de vóxel a coordenadas físicas |
| Proxy | Medición indirecta o aproximada |
| QC | Control de calidad técnico |
| Dominio | Condición técnica de adquisición |
| MAD | Desviación absoluta mediana |
| MIP | Proyección de máxima intensidad |
| Score técnico | Ranking exploratorio de atipicidad, no probabilidad clínica |

