# Notebook 04: registro, biomarcadores candidatos y radiomics 3D

## 1. Propósito y alcance

Este documento describe exclusivamente el notebook
`notebooks/04_registro_biomarcadores_radiomica.ipynb`. Su función es llevar todos los estudios
etiquetados a un espacio físico común, definir regiones de análisis, extraer variables comparables y
construir mapas descriptivos por clase.

El notebook responde seis preguntas:

1. ¿Cómo hacer comparables estudios con dimensiones y spacing diferentes?
2. ¿Cómo alinear cada volumen con una referencia común?
3. ¿Qué región se utilizará como objetivo y qué región como fondo?
4. ¿Cómo resumir captación, forma y textura tridimensional?
5. ¿Cuánto cambian los resultados si se modifica ligeramente la máscara?
6. ¿Cómo procesar 1.362 estudios sin conservarlos simultáneamente en memoria?

El flujo conceptual es:

```text
Manifiesto del notebook 03
        |
        +-- seleccionar referencia
        +-- orientar LPS y remuestrear a 2,5 mm
        +-- registrar cada volumen
        +-- aplicar atlas o máscara consenso proxy
        +-- estimar fondo robusto y su QC
        +-- separar cuantificación relativa de mapas de intensidad
        +-- extraer captación, forma, textura y estabilidad
        +-- actualizar mapas y checkpoints
        +-- persistir tablas, mapas y crops
```

El notebook no diagnostica enfermedad y no produce biomarcadores clínicamente validados por sí
solo. Cuando no se entrega un atlas validado, las regiones anatómicas son proxies reproducibles
construidos desde la propia cohorte.

## 2. Terminología fundamental

### Registro de imágenes

Registrar dos imágenes significa encontrar una transformación espacial que lleve la imagen móvil al
sistema de coordenadas de una imagen fija. El objetivo es que una misma posición represente una zona
comparable entre sujetos.

El registro no cambia la etiqueta ni crea señal biológica. Reubica la información mediante una
transformación y una interpolación.

### Imagen fija, imagen móvil y referencia

- **Imagen fija:** define el grid y el espacio al que se alinean los estudios.
- **Imagen móvil:** el estudio que se transforma.
- **Referencia:** estudio elegido como imagen fija común.

La referencia no representa una verdad anatómica universal. Su elección condiciona el grid, la
interpolación y la facilidad del registro. Para una versión definitiva conviene congelar su UID o
utilizar un template externo validado.

### LPS y relación con RAS

SimpleITK utiliza habitualmente la convención LPS:

- `L`: izquierda;
- `P`: posterior;
- `S`: superior.

El notebook 03 utilizó RAS con nibabel. Ambas convenciones describen coordenadas físicas; cambian los
signos de los ejes X e Y. La conversión se realiza usando la información espacial del NIfTI. No se
deben comparar arrays de nibabel y SimpleITK ignorando sus affines, direcciones y convenciones.

### Isotropía y grid común

Un grid isotrópico tiene el mismo spacing en X, Y y Z. Aquí se utiliza 2,5 mm. El nuevo tamaño se
aproxima preservando la cobertura física:

$$
N_{nuevo}=\operatorname{round}\left(
N_{original}\frac{spacing_{original}}{spacing_{nuevo}}
\right).
$$

La intensidad se interpola linealmente. Las máscaras se interpolan con vecino más cercano para no
crear etiquetas fraccionarias.

### Máscara, ROI y atlas

Una máscara es un volumen binario que indica qué vóxeles pertenecen a una región. ROI significa
*region of interest* o región de interés.

Un atlas validado contiene etiquetas anatómicas definidas externamente. Una máscara consenso proxy
se estima desde las intensidades de esta cohorte. La segunda es útil para exploración, pero no debe
presentarse como segmentación anatómica certificada.

### SBR-like

SBR significa *specific binding ratio*. En el notebook se utiliza una versión aproximada:

$$
SBR_{like}=\frac{\overline{I}_{target}-\overline{I}_{fondo}}
{\overline{I}_{fondo}}
=\frac{\overline{I}_{target}}{\overline{I}_{fondo}}-1.
$$

Se denomina `SBR-like` porque el fondo y el target pueden provenir de una máscara proxy, no de ROIs
clínicas estandarizadas ni de un protocolo de cuantificación validado.

### Streaming y checkpoint

Streaming significa procesar un estudio, actualizar los acumuladores y liberar su volumen antes de
cargar el siguiente. Un checkpoint es una fotografía persistente del avance que permite continuar
después de una interrupción.

## 3. Recorrido celda por celda

### Celda 1: presentación del flujo completo

Declara que se procesarán todos los exámenes etiquetados mediante:

```text
cargar -> orientar -> registrar -> extraer -> acumular -> liberar
```

No se conservan los 1.362 volúmenes simultáneamente en RAM. También establece la jerarquía de
máscaras: primero un atlas validado; si no existe, un consenso de controles utilizado como proxy.

### Celda 2: contrato de configuración

Indica que los artefactos se guardarán en:

```text
outputs/private_eda/full_cohort_v3/
```

Los notebooks 04 y 05 comparten esta única carpeta. El sufijo `v3` congela el nuevo contrato y evita
mezclarlo con resultados históricos producidos con otras definiciones. Los checkpoints incluyen
features, QC de registro y acumuladores voxel-a-voxel de media y varianza.

### Celda 3: bibliotecas, parámetros y rutas

Define las entradas, salidas y decisiones principales.

| Parámetro | Significado | Efecto |
|---|---|---|
| `VALIDATED_MASK_PATH=None` | No se configuró atlas | Se construye consenso proxy |
| `MAX_SCANS=None` | Sin máximo | Intenta procesar toda la cohorte etiquetada |
| `REFERENCE_UID=None` | Referencia automática | Selección determinística |
| `REGISTRATION_MODE='rigid'` | Registro rígido | Rotación y traslación, sin deformación |
| `REGISTRATION_BACKEND='torch_cuda'` | Backend preferido | Optimización rígida multirresolución en NVIDIA |
| `GPU_FALLBACK_TO_SITK=True` | Respaldo del registro | Repite en CPU si CUDA falla o parece poco fiable |
| `ISOTROPIC_SPACING_MM=2.5` | Spacing común | Vóxeles de 2,5 mm por eje |
| `MASK_TEMPLATE_SCANS=48` | Tamaño máximo del consenso | Hasta 48 controles representativos |
| `TARGET_PERCENTILES=(88,90,92,94)` | Umbrales alternativos | Sensibilidad del target |
| `ACTIVE_BACKGROUND_SD=2.0` | Umbral de señal activa | Fondo más dos desviaciones estándar |
| `BACKGROUND_SCALE_FLOOR_FRACTION=0.05` | Piso del denominador | Al menos 5% del p90 del foreground |
| `MIN_BACKGROUND_VOXELS=64` | Soporte mínimo | Exige al menos 64 vóxeles para el fondo nominal |
| `MIN_BACKGROUND_SUPPORT_FRACTION=0.20` | Cobertura mínima | Exige 20% de la máscara nominal con señal positiva |
| `TEXTURE_LEVELS=32` | Discretización | GLCM de 32 niveles |
| `CHECKPOINT_EVERY=20` | Frecuencia de guardado | Persistencia cada 20 intentos |
| `RESUME=True` | Reanudación | Reutiliza resultados compatibles |
| `RETRY_FAILURES=True` | Política de fallos | Reintenta UID fallidos bajo el contrato corregido |
| `SAVE_REGISTERED_CROPS=True` | Persistencia de imágenes | Guarda crops normalizados |
| `CROP_MARGIN_MM=20` | Contexto alrededor del target | Margen físico de 20 mm |

`MAX_SCANS=None` no limita los estudios. `MASK_TEMPLATE_SCANS=48` limita solamente la cantidad usada
para construir la máscara consenso.

La celda comprueba que el modo sea `rigid` o `affine`, que existan el ZIP y el manifiesto del notebook
03, y crea las carpetas privadas necesarias.

### Celda 4: contrato de cohorte y referencia

Aclara que no se excluye automáticamente el 5% técnicamente extremo. Los casos se intentan registrar
y los problemas se expresan como fallos o banderas de QC.

La referencia automática se escoge entre controles de la familia de adquisición más frecuente. Esto
reduce dos riesgos: elegir una geometría rara y utilizar como referencia un patrón posiblemente muy
alterado por la patología.

### Celda 5: cohorte etiquetada y selección de referencia

Primero carga `image_qc_manifest.csv` y exige las columnas necesarias del notebook 03. Después:

1. conserva sólo estudios con etiqueta disponible;
2. convierte la etiqueta a entero;
3. ordena determinísticamente por UID;
4. si hubiera muestreo, selecciona de forma aproximadamente estratificada por clase.

Con `MAX_SCANS=None`, el random no actúa.

La familia principal es la que contiene más estudios. Dentro de sus controles se calcula una
distancia geométrica usando spacing y FOV estandarizados. A esa distancia se añade el 20% del score QC
del notebook 03:

$$
S_{referencia}=D_{geometría}+0.20\,S_{QC}.
$$

Se elige el menor score. Es una regla de ingeniería reproducible, no una selección clínica. Si no hay
controles en la familia principal, utiliza todos los controles; si tampoco existen, utiliza toda la
cohorte.

### Celda 6: contrato del registro físico

Establece que SimpleITK conservará origen, dirección y spacing durante la lectura y el remuestreo
inicial. La referencia se remuestrea a 2,5 mm isotrópicos. La interpolación lineal es apropiada para
intensidades continuas; el vecino más cercano se reserva para máscaras discretas. El ajuste rígido
residual usa CUDA cuando está disponible y conserva un fallback trazable a SimpleITK/CPU.

### Celda 7: lectura LPS, grid isotrópico y registro

#### Lectura desde el ZIP

Cada NIfTI se extrae temporalmente, se lee como `float32`, se orienta a LPS y se separa del archivo
temporal. Igual que en el notebook 03, procesar uno por uno reduce RAM, pero reabrir el ZIP aumenta el
costo de entrada y salida.

#### Construcción del grid común

`make_isotropic_reference` calcula el tamaño necesario para aproximar la misma extensión física con
spacing de 2,5 mm. La referencia isotrópica define tamaño, origen, dirección y spacing para todos los
estudios registrados.

#### Intensidad utilizada por el optimizador

Antes de optimizar, cada imagen se reescala entre 0 y 1. Esta escala se utiliza para el registro; no es
la normalización final de biomarcadores.

#### Registro rígido

El modo predeterminado utiliza una transformación Euler 3D con seis grados de libertad:

- tres traslaciones;
- tres rotaciones.

No cambia tamaños ni formas anatómicas. El modo `affine` añade escalas y cizallamientos, por lo que es
más flexible, pero también puede absorber diferencias que interesa conservar.

#### Objetivo y optimización

El camino principal GPU optimiza la correlación cruzada normalizada con Adam y seis parámetros
rígidos. Trabaja a escalas 0,25, 0,50 y 1,00 con 28, 20 y 12 iteraciones máximas, respectivamente, y
detención temprana tras siete iteraciones sin mejora. Los ángulos y desplazamientos normalizados se
acotan para impedir transformaciones extremas.

El fallback SimpleITK usa una muestra aleatoria reproducible del 10% de los puntos, descenso de
gradiente regularizado y hasta 100 iteraciones por etapa.

El fallback CPU utiliza este esquema multirresolución:

```text
Nivel grueso: reducción 4, suavizado 2 mm
Nivel medio:  reducción 2, suavizado 1 mm
Nivel fino:   reducción 1, suavizado 0 mm
```

La escala gruesa captura desplazamientos grandes y la fina ajusta detalles. El suavizado se expresa
en unidades físicas.

El valor `metric_final_correlation_objective` es el objetivo interno del backend y puede usar una
convención de signo orientada a minimización. Para QC son más interpretables `correlation_before` y
`correlation_after`, calculadas directamente como correlación de Pearson sobre señal no nula. CUDA se
repite automáticamente en CPU si la correlación final no es finita, cae bajo 0,35 o empeora más de 0,03.

$$
r=\frac{\sum_i(a_i-\bar a)(b_i-\bar b)}
{\sqrt{\sum_i(a_i-\bar a)^2}\sqrt{\sum_i(b_i-\bar b)^2}}.
$$

La referencia no se registra contra sí misma: se asignan correlaciones iguales a uno.

### Celda 8: contrato de máscaras

Presenta dos caminos:

1. cargar un atlas validado;
2. construir un consenso proxy de controles.

El modo utilizado queda registrado en `mask_mode`, evitando presentar el fallback como atlas
anatómico.

### Celda 9: consenso, target y fondo

#### Normalización para el consenso

Cada estudio seleccionado se normaliza usando sus percentiles 1 y 99,5, se registra y se incorpora a
una media online. La normalización reduce el impacto de escalas absolutas diferentes.

#### Selección de controles representativos

Se priorizan controles con menor `within_protocol_qc_score` y se reparten cupos entre familias de
adquisición. Se usan como máximo 48 y se fuerza la inclusión de la referencia.

La media online se actualiza como:

$$
\mu_n=\mu_{n-1}+\frac{x_n-\mu_{n-1}}{n}.
$$

Esto evita almacenar los 48 volúmenes registrados simultáneamente.

#### Máscara target proxy

Sin atlas, el algoritmo:

1. conserva el 60% central del volumen en cada eje;
2. busca señal positiva por encima del percentil seleccionado, inicialmente p90;
3. divide el eje X en dos hemisferios;
4. conserva el componente conectado más grande de cada mitad;
5. aplica cierre binario y rellena huecos.

En un array SimpleITK la disposición es Z, Y, X. Bajo LPS, los índices X bajos corresponden al lado
derecho físico y los altos al izquierdo. Esta relación depende de haber orientado correctamente antes.

El procedimiento busca dos regiones dominantes de alta captación. Es reproducible, pero no identifica
por sí mismo caudado y putamen.

#### Máscara de cerebro y fondo proxy

Se construye soporte con señal superior a p10 y se conserva su mayor componente conectado dentro de un
elipsoide central. Este paso evita que el anillo del detector o el aire periférico se conviertan en
"fondo". Dentro de una banda posterior de ese soporte cerebral se buscan vóxeles bajo p70, excluyendo
una dilatación del target.

Si esa región es pequeña, se usa el soporte cerebral no target bajo p70. La regla sigue siendo una
aproximación reproducible y no certifica una ROI occipital clínica.

#### Atlas validado

Si se configura `VALIDATED_MASK_PATH`, el atlas se orienta a LPS y se remuestrea con vecino más
cercano. Se exigen al menos target derecho, target izquierdo y fondo. Opcionalmente puede contener
caudado y putamen por lado.

Las máscaras finales y el consenso se guardan en `analysis_masks_full.npz`.

### Celda 10: contrato de features

Declara tres familias principales de variables:

- semicuantificación relativa al fondo;
- forma de la región activa;
- textura GLCM realmente tridimensional.

La implementación es reproducible dentro del proyecto, pero aún no ha sido validada contra IBSI. Por
eso debe denominarse radiomics exploratoria, no radiomics clínica certificada.

### Celda 11: captación, forma, textura y estabilidad

#### Fondo robusto y normalización semicuantitativa

La máscara de fondo se intersecta con la señal positiva del estudio. Se acepta nominalmente cuando
contiene al menos 64 vóxeles y cubre al menos 20% de la ROI de fondo. Si no lo consigue, se buscan en
orden: el soporte cerebral no target bajo p70 y la mitad inferior de intensidades positivas. Sobre los
valores disponibles se usa una media recortada entre p5 y p95.

Para impedir que un fondo casi nulo genere razones enormes se define:

$$
B_{piso}=0.05\,P90(I_{foreground}).
$$

$$
B_{usado}=\max(\overline{I}_{fondo,recortado}, B_{piso}).
$$

Para cada estudio registrado, las variables relativas se calculan como:

$$
R(x,y,z)=\frac{I(x,y,z)}{B_{usado}}.
$$

Las features relativas usan esta razón. Se guardan además el origen del fondo, soporte, cobertura,
media cruda, piso aplicado y `background_qc_valid`. Un fallback o un piso ya no elimina al estudio:
lo conserva con una advertencia auditable. El caso sólo falla si ni siquiera existe foreground o fondo
robusto mínimo.

Los mapas usan otra escala, independiente del fondo:

$$
I_{01}(x,y,z)=\operatorname{clip}\left(
\frac{I(x,y,z)}{P99(I_{foreground})},0,1
\right).
$$

Separar ambos contratos evita que una ROI de fondo débil domine las medias voxel-a-voxel.

#### Variables semicuantitativas

Se calculan:

- SBR-like total, derecho e izquierdo;
- menor SBR entre ambos lados;
- asimetría absoluta y con signo;
- razón posterior/anterior;
- media, desviación y percentiles 10, 50 y 90 de la razón target/fondo;
- `log(1 + target/fondo)` como resumen comprimido para inspección y modelado robusto.

La asimetría bilateral es:

$$
AI_{abs}=\frac{|\bar I_L-\bar I_R|}{(\bar I_L+\bar I_R)/2}.
$$

Estas variables son candidatas biológicas sólo si las máscaras representan regiones apropiadas. Con
consenso proxy conservan explícitamente ese carácter exploratorio.

#### Región activa y forma

Dentro del target se define señal activa mediante:

$$
T_{activo}=\overline{I}_{fondo}+2\,SD(I_{fondo}).
$$

Se aplica cierre morfológico y se analiza el componente activo más grande. Sus variables incluyen:

- volumen en mililitros;
- superficie en milímetros cuadrados;
- elongación;
- esfericidad;
- extensión dentro de su caja envolvente;
- número de componentes;
- fracción perteneciente al componente principal.

El volumen es:

$$
V=N_{vóxeles}\,spacing_x\,spacing_y\,spacing_z.
$$

La elongación se estima desde los autovalores de la covarianza de las coordenadas físicas:

$$
E=\sqrt{\frac{\lambda_{max}}{\lambda_{min}}}.
$$

La superficie se aproxima mediante `marching_cubes`. La esfericidad es:

$$
\Psi=\frac{\pi^{1/3}(6V)^{2/3}}{A}.
$$

Una esfera ideal tiene un valor próximo a uno. La medida depende fuertemente de la máscara, el
spacing y la interpolación.

#### Textura GLCM 3D

Las intensidades relativas dentro del target se recortan robustamente entre p1 y p99 y se cuantizan
en 32 niveles. Se cuentan pares de vóxeles vecinos en 13 direcciones 3D y también en su dirección
inversa, obteniendo una matriz simétrica de probabilidades `P(i,j)`.

Las variables principales son:

| Variable | Definición conceptual |
|---|---|
| Contraste | Penaliza fuertemente niveles vecinos diferentes |
| Disimilitud | Diferencia absoluta promedio entre niveles |
| Homogeneidad | Favorece pares cercanos a la diagonal |
| Energía | Concentración de la matriz de probabilidades |
| Entropía | Diversidad o desorden de pares |
| Correlación | Asociación lineal entre niveles vecinos |

Por ejemplo:

$$
Contraste=\sum_{i,j}P(i,j)(i-j)^2,
$$

$$
Energía=\sqrt{\sum_{i,j}P(i,j)^2},
$$

$$
Entropía=-\sum_{P(i,j)>0}P(i,j)\log_2 P(i,j).
$$

Estas features dependen de la cuantización y la máscara. Sus valores sólo son comparables si se
mantiene el mismo contrato.

#### Sensibilidad y estabilidad

El notebook repite el SBR-like bajo pequeñas perturbaciones:

1. target construido con percentiles 88, 90, 92 y 94;
2. target erosionado, original y dilatado;
3. desplazamientos de un vóxel en X o Y;
4. fondo erosionado, original y dilatado.

Para cada familia se guarda el rango:

$$
Span=\max(SBR_{like})-\min(SBR_{like}).
$$

Un span pequeño significa que la medida cambia poco ante esa perturbación. Un span alto indica que el
resultado depende fuertemente de la máscara, posición o fondo. Es un indicador de fragilidad, no una
probabilidad de error.

### Celda 12: contrato de streaming y crops

Explica que cada crop comparte el grid de referencia y se guarda en `float16` comprimido. Conserva
dos representaciones: razón contra fondo para semicuantificación e intensidad `I01` para imagen. Esto
habilita una futura rama 3D o 2.5D sin conservar el volumen completo.

### Celda 13: configuración congelada, reanudación y procesamiento

#### Hash de configuración

Se serializan referencia, modo de registro, spacing, máscara, umbrales, niveles de textura y hash de
UID. Después se calcula SHA-256.

Si existe un checkpoint y el hash no coincide, el notebook se detiene. Esto evita mezclar features
producidas con contratos diferentes. El hash protege los parámetros principales, pero no reemplaza
un registro completo de versiones de librerías y entorno.

#### Estado reanudable

Con `RESUME=True` se cargan:

- features existentes;
- QC de registro;
- fallos;
- estado online de mapas.

Los UID cuyo feature, mapa y crop ya quedaron confirmados se omiten. Con `RETRY_FAILURES=True`, los
fallos anteriores se vuelven a intentar y desaparecen de la tabla de fallos si luego terminan bien.
Los CSV se deduplican por UID, por lo que reanudar no duplica filas.

#### Actualización online de mapas

Para cada clase se conserva cantidad, media y acumulador `M2` mediante el algoritmo de Welford:

$$
\delta=x_n-\mu_{n-1},
$$

$$
\mu_n=\mu_{n-1}+\frac{\delta}{n},
$$

$$
M2_n=M2_{n-1}+\delta(x_n-\mu_n).
$$

La varianza muestral posterior es `M2/(n-1)`. Esto permite calcular mapas sin guardar todos los
volúmenes en RAM.

#### Crops registrados

La caja se define alrededor del target más un margen físico de 20 mm. Se guarda:

- volumen razón recortado entre 0 y 20 en `float16`;
- volumen `I01` acotado entre 0 y 1 en `float16`;
- máscara target;
- inicio del crop en Z, Y, X;
- spacing X, Y, Z.

El recorte 0-20 se aplica sólo a los crops persistidos para controlar valores extremos y almacenamiento.

#### Escritura atómica y checkpoints

Los CSV y el estado se escriben primero en archivos temporales y luego se reemplazan. Esto disminuye
el riesgo de dejar archivos parcialmente escritos si el proceso se interrumpe.

Cada estudio se registra, se convierte a array, genera features, actualiza mapas y opcionalmente guarda
su crop. Los fallos quedan con UID y excepción. Cada 20 intentos se ejecuta un checkpoint. Si se pulsa
Ctrl+C o se produce `SystemExit`, primero se fuerza un checkpoint; ante un corte eléctrico sólo puede
repetirse el bloque aún no confirmado, sin duplicar resultados.

Los reemplazos son atómicos y reintentan bloqueos transitorios de Windows/OneDrive. Un crop existente
sólo se reutiliza si contiene ambas representaciones esperadas; un archivo truncado se reconstruye.

#### QC del registro

Se calcula:

$$
Ganancia_r=r_{después}-r_{antes}.
$$

Se activa `registration_qc_flag` cuando:

- la correlación posterior no es finita;
- la correlación posterior es menor que 0,55;
- la correlación empeora más de 0,05.

Estos umbrales son heurísticos. La bandera no elimina automáticamente el caso ni garantiza que un
registro sin bandera sea anatómicamente correcto.

### Celda 14: contrato de mapas y efectos

Aclara que las medias son voxel-a-voxel de volúmenes registrados y normalizados por p99 del foreground.
No son promedios de MIP ni usan el denominador SBR.

También establece que Cohen d usa varianza agrupada ponderada y que Hedges g corrige su sesgo de
muestra pequeña.

### Celda 15: mapas, visuales y cobertura

Para cada vóxel se calculan:

$$
\Delta=\overline{X}_{patológica}-\overline{X}_{normal}.
$$

La varianza agrupada es:

$$
s_p^2=\frac{(n_0-1)s_0^2+(n_1-1)s_1^2}{n_0+n_1-2}.
$$

Cohen d descriptivo es:

$$
d=\frac{\overline{X}_{patológica}-\overline{X}_{normal}}{s_p}.
$$

Hedges g aplica la corrección:

$$
g=Jd,\qquad J\approx1-\frac{3}{4(n_0+n_1)-9}.
$$

Estos mapas muestran tamaño de diferencia estandarizada. No son pruebas de significancia, no incluyen
intervalos de confianza y no corrigen comparaciones múltiples entre vóxeles.

#### Visual de máscaras

El overlay apila tres canales:

```python
R = intensidad del consenso
G = máscara target
B = máscara de fondo
```

La combinación intensidad roja más target verde aparece amarilla; el fondo se observa azul o púrpura.
Este gráfico permite comprobar posición, cobertura y solapamiento.

#### Panel descriptivo

Se muestran:

- consenso y overlay de máscaras;
- histograma de correlación posterior al registro;
- `log(1 + target/fondo robusto)` por clase;
- media normal y patológica;
- diferencia voxel-a-voxel;
- Hedges g descriptivo.

El corte de máscaras maximiza el área del target. El corte de mapas maximiza la suma del efecto
absoluto dentro del target; por tanto, es una selección descriptiva orientada a visualización.

La tabla de cobertura informa denominador solicitado, features obtenidas, fallos, contribuciones por
clase a los mapas y banderas de registro. Finalmente se ordenan los veinte casos con mayor sensibilidad
a máscara, traslación, umbral o fondo.

### Celda 16: contrato de salida

Resume los artefactos producidos:

| Artefacto | Contenido |
|---|---|
| `dat_radiomics_features_full.csv` | Variables candidatas por estudio |
| `registration_qc_full.csv` | Correlaciones, ganancia y bandera técnica |
| `registration_failures_full.csv` | UID y error de casos fallidos |
| `streaming_state_full.npz` | Media, M2 y conteos reanudables |
| `cohort_maps_full.npz` | Medias, diferencia, Cohen d y Hedges g |
| `analysis_masks_full.npz` | Consenso y máscaras utilizadas |
| `registered_crops/*.npz` | Crops con `volume_ratio` y `volume_intensity01` |
| `registration_radiomics_full_config.json` | Configuración y hash de compatibilidad |

El notebook no elimina casos ni declara las features como biomarcadores clínicamente validados.

## 4. Cómo interpretar cada bloque de features

| Bloque | Pregunta que intenta responder | Principal limitación |
|---|---|---|
| Semicuantitativo | ¿Cuánta señal relativa existe en target y por lado? | Depende de target y fondo |
| First-order ratio | ¿Cómo se distribuye la intensidad relativa? | Ignora organización espacial |
| Forma activa | ¿Qué volumen y geometría tiene la señal sobre fondo? | Muy sensible al umbral |
| Textura 3D | ¿Cómo se relacionan intensidades vecinas? | Depende de cuantización y máscara |
| Estabilidad | ¿Cuánto cambia el resultado ante perturbaciones? | Perturbaciones limitadas y heurísticas |
| QC de registro | ¿El alineamiento parece razonable numéricamente? | Correlación no garantiza anatomía |

Las variables técnicas de registro y dominio deben mantenerse separadas de las variables biológicas
al evaluar modelos. Pueden servir para auditoría, estratificación y análisis de confusión.

## 5. Supuestos y limitaciones

- Una referencia individual es suficiente para este análisis exploratorio, pero puede introducir sesgo de template.
- El registro rígido conserva forma, pero no corrige diferencias anatómicas locales.
- La interpolación modifica suavemente intensidades y textura.
- Un consenso de captación no equivale a un atlas anatómico.
- La región de fondo cerebral/posterior es una regla proxy dependiente de orientación y señal.
- El piso del denominador estabiliza razones, pero cambia su interpretación cuando se activa; por eso se persisten `background_floor_applied` y `background_qc_valid`.
- SBR-like no es una cuantificación clínica certificada.
- La forma activa depende del umbral fondo más dos desviaciones estándar.
- La GLCM es 3D y reproducible, pero todavía no está verificada contra IBSI.
- Los spans evalúan cuatro perturbaciones concretas; no cubren toda la incertidumbre posible.
- Las banderas de registro y los criterios de fallback usan umbrales heurísticos.
- Cohen d y Hedges g son descriptivos, no inferencia voxel-a-voxel.
- El hash de configuración no captura automáticamente todas las versiones del entorno.
- Procesar desde un ZIP limita memoria, pero aumenta el tiempo de entrada y salida.
- La persistencia limita la pérdida de trabajo; no puede guardar una operación que estaba a mitad de cálculo durante un corte abrupto.

## 6. Glosario breve

| Término | Significado en este notebook |
|---|---|
| Registro | Alineamiento espacial de una imagen móvil con una fija |
| Referencia | Imagen que define el grid común |
| LPS | Convención izquierda-posterior-superior |
| Isotrópico | Mismo spacing en X, Y y Z |
| Interpolación lineal | Estimación continua usada para intensidades |
| Vecino más cercano | Interpolación discreta usada para etiquetas |
| ROI | Región de interés |
| Atlas | Volumen externo con etiquetas anatómicas |
| Consenso | Media de controles registrados y normalizados |
| SBR-like | Razón de unión específica aproximada |
| GLCM | Matriz de coocurrencia de niveles de gris |
| Streaming | Procesamiento secuencial con memoria acotada |
| Checkpoint | Estado persistente que permite reanudar |
| Crop | Subvolumen alrededor del target |
| Welford | Algoritmo online de media y varianza |
| Cohen d | Diferencia estandarizada descriptiva |
| Hedges g | Cohen d con corrección de muestra pequeña |
| QC flag | Bandera de revisión, no exclusión automática |
