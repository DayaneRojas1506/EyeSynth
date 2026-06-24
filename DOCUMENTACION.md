# EyeSynth — Documentación del proyecto

Plataforma de **eye-tracking en el navegador** (WebGazer.js) para un experimento de
**detección de imágenes falsas (FAKE/REAL)**, con análisis posterior en Python
(heatmaps, scanpaths, AOI) y comparación de la **atención humana vs. la atención de
modelos de IA** (Grad-CAM y mapas de saliencia).

El participante ve grillas 2×2 de variantes de una misma imagen (una original y
tres editadas/generadas) y debe elegir cuál es la original. Mientras decide, se
registra su mirada. Después se comparan esos patrones de atención con los de tres
detectores de fakes.

---

## 1. Arquitectura general

```
EyeSynth/
├── frontend/                     App web (HTML/JS, sin build)
│   ├── experiment.html           Experimento principal (grilla 2×2 + eye-tracking)
│   ├── index.html                App de estímulo único (versión antigua)
│   ├── webgaze.html              Prueba/sandbox de WebGazer
│   ├── sets_manifest.json        Catálogo de sets de imágenes del experimento
│   └── dataset/images/<version>/ Imágenes del estímulo (4 versiones × 100 imgs)
│
├── server/                       Backend Flask
│   ├── server.py                 Servidor: sirve la app, guarda sesiones, /compare-models
│   ├── models_infer.py           Inferencia + Grad-CAM + métricas de similitud
│   ├── simple_model.py           Clasificador SIMPLE (regresión logística, 6 features)
│   ├── npr_model.py              Modelo NPR (ResNet-50 sobre residuos de píxeles)
│   ├── coco_model.py             Modelo COCO (backbone semántico Faster R-CNN)
│   ├── train_simple_fakereal.py  Entrenamiento del SIMPLE (numpy, sin GPU)
│   ├── train_npr_fakereal.py     Fine-tuning del NPR (torch)
│   ├── train_coco_fakereal.py    Entrenamiento del COCO (torch)
│   ├── simple_fakereal.json      Checkpoint del SIMPLE (pesos en JSON)
│   ├── npr_fakereal.pth          Checkpoint del NPR (no versionado)
│   └── coco_fakereal.pth         Checkpoint del COCO (no versionado)
│
├── analysis/                     Análisis offline de las sesiones
│   ├── eyesynth_analysis.py      Heatmaps/scanpaths/métricas (sesiones single)
│   ├── aoi_analysis.py           Análisis por zonas (Areas of Interest, rejilla 3×3)
│   ├── render_attention_heatmaps.py  Heatmap difuso sobre cada imagen
│   ├── render_attention_points.py    Puntos + scanpath sobre cada imagen
│   ├── export_attention_points.py    Exporta puntos de mirada por usuario/imagen
│   ├── sessions/                 JSON de sesiones del estímulo único
│   ├── experiment_sessions/      JSON de sesiones del experimento (lo que se usa)
│   ├── attention_heatmaps/       Salida: PNG de heatmaps por participante
│   ├── attention_maps/           Salida: PNG de scanpaths por participante
│   └── attention_points/         Salida: JSON de puntos por participante
│
├── dataset_new_model/{fake,real}/  Dataset para (re)entrenar los detectores
├── requirements.txt
├── run.sh                        Arranque simple (Mac/Linux, sin torch)
├── run-wsl.sh                    Arranque completo en WSL (con torch + detectores)
└── README.md                    (descripción antigua, ver este archivo)
```

> **Nota:** el `README.md` original describe únicamente la app de estímulo único.
> Este documento cubre el proyecto completo y actual (experimento 2×2 + detectores).

---

## 2. Flujo del experimento (frontend)

Archivo: [frontend/experiment.html](frontend/experiment.html) (~1550 líneas, todo
HTML+JS inline, sin build). Carga WebGazer 2.1.0 desde CDN.

Etapas:

1. **Consentimiento y datos** — id de participante (auto), género, edad, **uso de
   lentes** (`glasses`: `si`/`no`) y **modo de corrección de mirada**
   (`trackMode`: alta precisión / estándar; ver 2-bis #2), consentimiento.
2. **Encuadre de cámara** — `getUserMedia`; muestra preview, overlay de cara y caja
   de feedback de WebGazer. Antes de calibrar muestra un **aviso de recomendaciones
   de precisión** (distancia 50–70 cm, ojos a la altura de la cámara, luz frontal,
   lentes sin reflejos, cabeza quieta). Requiere contexto seguro → siempre `localhost`.
3. **Calibración** — 13 puntos (calibración densa, el valor por defecto:
   `tune.calibPoints=13`) o 5 puntos esquinas+centro (recalibración rápida, 3
   clics por punto). Existe también un modo de 9 puntos (`CALIB_FULL`), que solo
   se usa si se baja `calibPoints` por debajo de 13 desde la pantalla de prueba.
4. **Gate de precisión** — antes de empezar mide el error de mirada en px/%. Si
   supera el umbral, ofrece recalibrar. Resultado guardado en `tracking`.
5. **Sets** — por cada set se muestra una grilla 2×2 con las 4 versiones en orden
   barajado; el participante hace clic en la que cree original. Hay límite de
   tiempo (`set_time_s`, p.ej. 60 s) y recalibración cada N sets (`recalib_every`).
   La mirada se registra en `gaze_samples` durante la observación.
6. **Guardado** — al terminar se hace `POST /save-experiment` con todo el JSON.

**Suavizado de mirada:** el JS implementa varios filtros sobre el stream crudo de
WebGazer — media móvil, EMA y **One-Euro filter** (`filterOneEuro`) — más un
Kalman propio de WebGazer (`applyKalmanFilter`). Se descartan muestras pegadas a
los bordes.

**Selección de sets:** `mulberry32` (PRNG sembrado, `selection_seed`) + `shuffle`
permiten que todos los participantes vean el mismo subconjunto (`same_for_all`) y
en orden reproducible.

### Catálogo de sets — [frontend/sets_manifest.json](frontend/sets_manifest.json)

33 sets curados, cada uno con 4 versiones de una imagen:

| Versión | Significado |
|---|---|
| `original` | imagen real (respuesta correcta) |
| `semantic_aware` | edición generativa consciente del contenido |
| `semantic_agnostic` | edición generativa agnóstica al contenido |
| `pix2pix_magicbrush` | edición con pix2pix / MagicBrush |

Fuentes de imágenes: ADE20K, COCO, LHQ. El dataset completo en
`frontend/dataset/images/<version>/` tiene **100 imágenes por versión**.

---

## 2-bis. Técnicas para mejorar el eye-tracking (vs. WebGazer "crudo")

WebGazer.js por sí solo es un eye-tracker de webcam de **baja precisión y mucho
ruido**: el cursor de mirada salta varios cientos de píxeles entre frames, tiene
un **sesgo sistemático** que cambia con la postura/iluminación, y entrega muestras
basura cuando pierde el rostro. Para que los datos sirvan para un experimento se
montó, sobre WebGazer, una capa de mejoras (toda en
[frontend/experiment.html](frontend/experiment.html)).

> Esta sección describe **únicamente lo que corre durante la observación de los
> sets** (la fase real del experimento). El código incluye además filtros
> alternativos (EMA, media móvil) y ajustes que solo se cambian desde la pantalla
> de prueba; **no** se documentan aquí porque no intervienen en la recolección de
> datos. La configuración de producción es la de los valores por defecto.

### 1. Filtro 1€ (One-Euro Filter) — suavizado adaptativo
`filterOneEuro` (en [experiment.html](frontend/experiment.html)).
Es el filtro que se aplica a cada muestra durante los sets (`filterMode = "oneeuro"`,
el valor por defecto). Ajusta su corte (cutoff) según la velocidad de la mirada:
**suaviza fuerte cuando el ojo está fijo** (mata el jitter en reposo) y **deja
pasar la señal cuando el ojo se mueve** (no añade retardo en las sácadas).
Parámetros: `minCutoff=1.0`, `beta=0.007`.

- **Por qué es mejor:** un promedio móvil o un EMA con coeficiente fijo obligan a
  elegir entre *suave pero con lag* o *responsivo pero tembloroso*. El 1€ rompe
  ese compromiso: es suave en las fijaciones (que es lo que mide el experimento) y
  rápido en los movimientos.

### 2. Corrección de sesgo (bias / offset) tras la calibración — dos modos
`runCalibGate` / `correctGaze` (en [experiment.html](frontend/experiment.html)).
El participante elige el modo en la pantalla de datos (`trackMode`); queda
registrado en `tracking.correction_mode`. Tras calibrar, el gate muestra puntos
objetivo, mide el **desvío** (predicho − objetivo) en cada uno y construye la
corrección que se aplica a **cada muestra futura** (`correctGaze` en `onGaze`):

- **Modo "global" (estándar):** mide **5 puntos** (centro + esquinas), calcula un
  único offset `(-avg(dx), -avg(dy))` y lo resta a toda predicción
  (`x = rx + state.gazeOffset.x`). Asume error uniforme en la pantalla. Es el
  comportamiento histórico y más rápido de calibrar.
- **Modo "zonal" (alta precisión, por defecto):** mide **5 anclas** (centro +
  esquinas) y guarda el desvío de **cada** ancla en `state.gazeField`.
  Para cada muestra interpola el offset por **IDW** (inverse-distance weighting):
  las anclas más cercanas a la mirada pesan más. Corrige el error **no uniforme**
  de WebGazer (pequeño al centro, mayor y asimétrico en las esquinas) que un offset
  único no puede. Si solo hubiera una ancla válida degrada al comportamiento global.

- **Por qué es mejor:** WebGazer tiene un sesgo sistemático que el regresor no
  corrige solo. El modo global elimina el componente medio; el zonal elimina además
  la **variación espacial** del error, que es la mayor fuente de imprecisión en los
  bordes. La medición se hace sobre la **mirada cruda** para que la corrección sea
  correcta. En el gate zonal el error reportado se calcula con **validación
  leave-one-out** (cada ancla se corrige con las *otras*) para que la precisión
  mostrada sea honesta y no esté inflada por ajustar y medir en el mismo punto.

### 3. Gate de precisión + recalibración guiada
`runCalibGate` → `showGateSummary` (en [experiment.html](frontend/experiment.html)).
Antes de empezar el experimento se calcula el **error residual** (ya con el offset
aplicado) como % de la diagonal de pantalla y se traduce a una **precisión %**.
Según el umbral (`gateThresholdPct=12`): <5 % "buena", ≤12 % "aceptable", si no
"baja → recalibrar". El participante puede recalibrar antes de generar datos.

- **Por qué es mejor:** WebGazer no avisa de su propia calidad. El gate **impide
  recolectar sesiones inservibles**: o se afina la calibración o, como mínimo, el
  error real queda registrado en `tracking` (`accuracy_pct`, `error_pct`,
  `error_px`, `n_recalibrations`) para poder filtrar/ponderar después.

### 4. Calibración por seguimiento + clics, con bloqueo anti-prematuro
`showCalibPoint` / `onCalibClick` (en [experiment.html](frontend/experiment.html)).
Cada punto de calibración alimenta a WebGazer de dos formas: con **clics**
(`recordScreenPosition(...,"click")`) y, mientras la persona mira el punto activo,
con **muestras continuas** cada 200 ms (`...,"move"`). Hay un **bloqueo de 1.2 s
(`LOCK_MS=1200`)** que impide hacer clic hasta que la mirada se estabiliza, y se
exigen **5 clics por punto** (`CLICKS_FULL`). La calibración inicial usa
`CALIB_DENSE` (**13 puntos**) por defecto (`tune.calibPoints=13`); la
**recalibración cada N sets** (`recalib_every=5`) usa la versión rápida de 5
puntos (esquinas + centro) con 3 clics (`CLICKS_SHORT`).

- **Por qué es mejor:** la calibración estándar de WebGazer es solo por clic. Sumar
  muestras de seguimiento da **más datos de entrenamiento por punto** (regresor más
  estable), y el bloqueo evita registrar el momento en que el ojo aún viaja hacia
  el punto (que envenenaría la calibración). La recalibración periódica corrige la
  deriva del tracking a lo largo de la sesión sin repetir la calibración completa.

### 5. Calibración en el área del estímulo (no en toda la pantalla)
`mapPointsToArea` (en [experiment.html](frontend/experiment.html)).
Los puntos de calibración y de medición se mapean a la **zona central donde
realmente se ven las imágenes**, no a los bordes de la pantalla.

- **Por qué es mejor:** el regresor de WebGazer es más impreciso en los extremos
  del campo de visión. Calibrar y validar donde caerá la mirada durante la tarea
  concentra la precisión justo donde importa.

### 6. Control de calidad por muestra (marcar, no descartar)
`onGaze` (en [experiment.html](frontend/experiment.html)).
Cada muestra que cae fuera de la pantalla (con margen `MARGIN=0.15`) se marca
`ok:false` y se cuenta en `qcDropped`, **pero no se elimina**. La pérdida de
rostro se traduce en ausencia de muestras, no en muestras malas.

- **Por qué es mejor:** filtrar agresivamente en vivo podía **vaciar sets enteros**
  cuando el tracking se iba un poco fuera de borde. Marcar en vez de descartar
  conserva la sesión completa y deja el filtrado fino para el análisis offline (que
  puede usar el flag `ok`). Es más robusto que confiar en WebGazer sin red.

### 7. Filtro Kalman de WebGazer + regresor "ridge" estable
`beginWebgazer` (en [experiment.html](frontend/experiment.html)).
Se activa `applyKalmanFilter(true)` (suavizado adicional integrado de WebGazer) y
se usa explícitamente `setRegression("ridge")` en lugar de `weightedRidge`.

- **Por qué es mejor:** en WebGazer 2.1.0 el regresor `weightedRidge` lanza un
  `TypeError` interno que **congela el stream de predicciones** (0 muestras). Fijar
  `ridge` evita ese bug; el Kalman añade una segunda capa de suavizado que se
  combina con el filtro 1€ del cliente.

### 8. Coordenadas relativas a la imagen (`u`, `v`)
`onGaze` (en [experiment.html](frontend/experiment.html)).
Además de las coordenadas de pantalla (`x_norm`, `y_norm`), cada muestra guarda
`u`/`v`: la posición **dentro del rectángulo de la imagen** (`rect`) y un flag
`on_image`.

- **Por qué es mejor:** WebGazer solo da coordenadas de pantalla. Tener `u,v`
  permite comparar la atención **imagen-a-imagen** (y contra los Grad-CAM de los
  modelos, que viven en el espacio de la imagen) sin re-mapear nada después.

### En resumen

| | WebGazer crudo | EyeSynth |
|---|---|---|
| Jitter en reposo | Alto | Filtro 1€ adaptativo + Kalman |
| Sesgo sistemático | Sin corregir | Offset global o corrección **zonal (IDW, 5 anclas)** |
| Control de calidad | Ninguno | Gate de precisión + flag `ok` por muestra |
| Calibración | Solo clics, toda la pantalla | Clics + seguimiento, bloqueo, área del estímulo |
| Robustez de la sesión | Se puede perder | Muestras malas se marcan, no se borran |
| Estabilidad | Bug de `weightedRidge` | `ridge` fijo (stream estable) |
| Datos para análisis | Solo px de pantalla | + `u,v` relativos a la imagen + métricas de tracking |

El resultado: scanpaths utilizables, un error de mirada **medido y acotado** por
sesión (típicamente ~5–6 % de la diagonal en las sesiones recolectadas) y datos
directamente comparables con la atención de los modelos de IA.

---

## 3. Backend (Flask)

Archivo: [server/server.py](server/server.py). **Exige torch** al importar
`models_infer`; por eso el servidor completo se corre en **WSL/Linux** con
`run-wsl.sh`, no en el venv ligero de Windows.

### Endpoints

| Método | Ruta | Descripción |
|---|---|---|
| GET | `/` | Sirve `frontend/index.html` (estímulo único) |
| GET | `/experiment` | Sirve `frontend/experiment.html` (experimento 2×2) |
| POST | `/save-session` | Guarda sesión del estímulo único en `analysis/sessions/` |
| POST | `/save-experiment` | Guarda sesión del experimento en `analysis/experiment_sessions/` |
| POST | `/compare-models` | Compara heatmap humano vs. mapas de los 3 modelos para una imagen |
| GET | `/stimuli` | Lista imágenes en `frontend/stimuli/` |
| GET | `/sessions` | Lista sesiones de estímulo único con metadatos |
| GET | `/experiment-sessions` | Lista sesiones de experimento con metadatos |

### Gestión de memoria de los modelos

Por defecto los modelos se **cargan por petición y se liberan** (evita picos de RAM
/ OOM en WSL). Con `EYESYNTH_KEEP_MODELS=1` se mantienen en memoria (más rápido).
La inferencia se **serializa con un lock** porque los runners comparten estado
mutable (hooks de Grad-CAM) y Flask atiende peticiones concurrentes.

---

## 4. Los tres detectores FAKE/REAL

Implementados en [server/models_infer.py](server/models_infer.py) (clase
`ModelBundle`, runners `SimpleRunner`, `NPRRunner`, `COCORunner`).

| Modelo | Arquitectura | Señal que explota | Mapa de atención |
|---|---|---|---|
| **SIMPLE** | Regresión logística sobre 6 features globales | Suavidad / detalle de alta frecuencia. Las fakes son globalmente más suaves. ~87 % val_acc | Heatmap de laplaciano local (suavidad) |
| **NPR** | ResNet-50 sobre residuos de píxeles vecinos (Tan et al., CVPR 2024) | Artefactos de upsampling de los generadores | Grad-CAM en `layer4[-1]` |
| **COCO** | ResNet-50 con backbone Faster R-CNN (COCO) | Features **semánticas** de objetos (baseline) | Grad-CAM en `layer4[-1]` |

**Detalle del SIMPLE** ([server/simple_model.py](server/simple_model.py)): 6 features
= residuo horizontal, residuo vertical, laplaciano, std de gris, std de RGB,
magnitud media de FFT. **No** redimensiona la imagen (un resize destruiría la señal
de alta frecuencia). El checkpoint es un JSON con `w, b, mu, sd`.

**Detalle del NPR** ([server/npr_model.py](server/npr_model.py)): `NPRTransform`
genera 6 canales (residuos horiz+vert × RGB). Usa **CenterCrop, no Resize**, para
preservar los residuos. `conv1` adaptada a 6 canales (pesos ImageNet replicados).

**Detalle del COCO** ([server/coco_model.py](server/coco_model.py)): copia el
backbone de `fasterrcnn_resnet50_fpn` (pesos COCO) a un ResNet-50; cabeza de 2
clases con dropout opcional. Sí usa Resize estándar (la señal semántica sobrevive).
Soporta `freeze_backbone` (linear probe).

### Métricas de similitud (humano vs. modelo)

En `similarity()`, sobre mapas 224×224 normalizados:

- **CC** — correlación de Pearson entre ambos mapas.
- **SIM** — intersección de histogramas (suma de mínimos de distribuciones).
- **KL** — divergencia de Kullback-Leibler del humano respecto al modelo.

El heatmap humano se construye en `build_human_heatmap()`: acumula los puntos de
mirada en una rejilla 200×200 y la suaviza con un kernel gaussiano (σ=9) vía FFT.

---

## 4-bis. Cómo se entrenaron los modelos

Los tres detectores se entrenan/reentrenan con el **mismo dataset**:
`dataset_new_model/` con dos subcarpetas `fake/` y `real/` (ImageFolder,
insensible a mayúsculas). El etiquetado se fuerza a `class_to_idx = {FAKE:0, REAL:1}`
por nombre de carpeta (no por orden alfabético) para que coincida con la inferencia.
Split **85 / 15** train/val con `seed=42` en los tres. El dataset es **pequeño**
(~560 imágenes de train), así que toda la configuración gira en torno a **evitar
el overfitting**.

### Resumen comparativo

| | SIMPLE | NPR | COCO |
|---|---|---|---|
| Script | [train_simple_fakereal.py](server/train_simple_fakereal.py) | [train_npr_fakereal.py](server/train_npr_fakereal.py) | [train_coco_fakereal.py](server/train_coco_fakereal.py) |
| Framework | numpy (sin GPU) | torch (WSL/GPU) | torch (WSL/GPU) |
| Punto de partida | desde cero | checkpoint NPR previo (o ImageNet) | backbone COCO Faster R-CNN (o ImageNet) |
| Qué se entrena | regresión logística (6 pesos) | ResNet-50 completo (fine-tuning) | **solo la cabeza `fc`** (linear probe) |
| Optimización | grad. descendente, 5000 iters, LR 0.1 | Adam, LR 5e-5, ≤30 épocas | Adam, LR 1e-3, ≤60 épocas |
| Regularización | L2 = 1e-3 | weight decay 1e-3, crop aug, early stopping (pat. 6) | dropout 0.3, label smoothing 0.1, weight decay 1e-3, early stopping (pat. 10) |
| Checkpoint | `simple_fakereal.json` | `npr_fakereal*.pth` | `coco_fakereal.pth` |

Los tres guardan `class_to_idx` y `val_acc`; los de torch guardan `model_state` y
seleccionan el **mejor** checkpoint por `val_acc` (no el último).

### SIMPLE — regresión logística sobre 6 features
`train_simple_fakereal.py`. No usa torch ni GPU; entrena en segundos.

1. Extrae las 6 features globales (`extract_features`) de cada imagen **a
   resolución nativa** (sin resize, para no destruir la alta frecuencia).
2. Normaliza con la media/desv del **train** (esa normalización se guarda en el
   checkpoint para aplicarla idéntica en inferencia).
3. Ajusta una regresión logística por descenso de gradiente: `LR=0.1`,
   `ITERS=5000`, `L2=1e-3`.
4. Guarda un JSON diminuto con `w, b, mu, sd`. El checkpoint actual
   (`simple_fakereal.json`) reporta **val_acc = 0.8687** (~87 %), train_acc 0.831.

- **Por qué así:** la señal fake/real de este dataset es global (suavidad), no
  espacial; una regresión logística la captura y **no puede memorizar** 560
  imágenes con solo 6 parámetros → generaliza.

### NPR — fine-tuning de ResNet-50 sobre residuos de píxeles
`train_npr_fakereal.py`. Corre en WSL/Linux con torch (idealmente GPU).

1. **Parte de un checkpoint NPR ya entrenado** (`--base-ckpt`) y lo fine-tunea; con
   `--from-imagenet` arranca solo de pesos ImageNet (conv1 replicada a 6 canales).
2. Preprocesado por **CenterCrop/RandomCrop, no Resize** (preservar los residuos de
   alta frecuencia que NPR explota); en train añade flip horizontal. **No**
   ColorJitter (alteraría la alta frecuencia).
3. Adam, **LR muy bajo (5e-5)**, weight decay 1e-3, `CosineAnnealingLR`,
   CrossEntropy. Tope 30 épocas con **early stopping** (paciencia 6).
4. Guarda el mejor checkpoint por `val_acc`. Grad-CAM en `layer4[-1]` en inferencia.

- **Por qué así:** un ResNet-50 (23M params) sobre 560 imágenes memoriza el train
  trivialmente. El LR bajo + pocas épocas + early stopping + el crop (que da
  augmentación gratis sin re-muestrear) son la defensa anti-overfitting. Con la
  config antigua (resize + LR 1e-5) el modelo solo memorizaba en 5 épocas.

### COCO — linear probe sobre backbone semántico
`train_coco_fakereal.py`. Corre en WSL/Linux con torch.

1. Inicializa el backbone ResNet-50 con los **pesos COCO de Faster R-CNN**
   (features semánticas de objeto); `--from-scratch` lo deja aleatorio.
2. **`freeze_backbone=True` por defecto (linear probe): congela todo y entrena solo
   la cabeza `fc`.** El backbone se pone en `eval()` para **congelar también las
   stats de BatchNorm** (si no, seguirían actualizándose y desestabilizarían un
   backbone que no se entrena); solo `fc` queda en modo train.
3. Cabeza con **dropout 0.3** + **label smoothing 0.1**. Como solo se entrena `fc`,
   se sube el LR a **1e-3** y se permiten hasta 60 épocas (early stopping pat. 10),
   Resize estándar + ColorJitter en augmentación (la señal semántica sí sobrevive).
4. El checkpoint guarda `dropout` y `freeze_backbone` para que la inferencia
   reconstruya la cabeza idéntica (con dropout>0, `fc` es un `Sequential`).

- **Por qué así:** descongelar todo el backbone daba acc 0.95 en train pero val_acc
  ~0.52 (azar) — overfitting de manual. El **linear probe es el baseline honesto**:
  mide cuánta señal fake/real hay en las features COCO sin que el ResNet memorice.

---

## 5. Análisis offline (carpeta `analysis/`)

Todos los scripts son independientes entre sí y resuelven rutas absolutas.

| Script | Entrada | Salida |
|---|---|---|
| [eyesynth_analysis.py](analysis/eyesynth_analysis.py) | `sessions/*.json` | Heatmap 512², scanpath, perfiles X/Y, métricas NSS/CC/SSIM, `resumen_sesiones.png` |
| [aoi_analysis.py](analysis/aoi_analysis.py) | `sessions/*.json` + estímulo | Conteo de mirada por zona (rejilla 3×3) + CSV/JSON |
| [render_attention_heatmaps.py](analysis/render_attention_heatmaps.py) | `experiment_sessions/*.json` | `attention_heatmaps/<part>/<img>__<version>.png` |
| [render_attention_points.py](analysis/render_attention_points.py) | `experiment_sessions/*.json` | `attention_maps/<part>/...png` (puntos + recorrido temporal) |
| [export_attention_points.py](analysis/export_attention_points.py) | `experiment_sessions/*.json` | `attention_points/<part>/...json` + CSV resumen por usuario |

**Métricas de saliencia** (en `eyesynth_analysis.py`):
- **CC** — Pearson entre el heatmap de la sesión y el mapa promedio del grupo.
- **NSS** — Normalized Scanpath Saliency: valor medio del mapa de referencia
  (media 0, desv 1) en las posiciones de mirada.
- **SSIM** — índice de similitud estructural frente al mapa de referencia.

---

## 6. Formato de los datos

### Sesión del experimento — `analysis/experiment_sessions/exp_<pid>_<ts>.json`

```jsonc
{
  "participant_id": "P20260612181600_boqq",
  "gender": "masculino", "age": 23, "glasses": "no",
  "timestamp": "2026-06-12T23:19:12.593Z",
  "consent": true,
  "screen": { "width": 1920, "height": 911 },
  "config": {
    "sets_per_session": 10, "same_for_all": true, "selection_seed": 42,
    "set_time_s": 60, "recalib_every": 5
  },
  "calibration": { "n_points": 5, "points": [...], "completed": true, "short": true },
  "tracking": { "accuracy_pct": 94.1, "error_pct": 5.9, "error_px": 126, "n_recalibrations": 0, "correction_mode": "zonal" },
  "sets": [
    {
      "set_id": "set_069", "image_id": "COCO_000000567093", "source": "COCO",
      "grid_order": ["semantic_agnostic","original","pix2pix_magicbrush","semantic_aware"],
      "answer_version": "original",   // versión elegida por el participante
      "correct": true,
      "response_time_s": 7.49, "timed_out": false,
      "images": [ { "version": "...", "url": "...", "rect": {...}, "gaze_samples": [...] } ]
    }
  ]
}
```

Cada `gaze_sample` lleva `t`, `x`/`y` (pantalla), `x_norm`/`y_norm` (pantalla) y
`u`/`v` (normalizado dentro de la imagen) + `on_image`.

### Sesión de estímulo único — `analysis/sessions/session_<ts>.json`

```jsonc
{
  "session_id": "...", "timestamp": "...",
  "screen": { "width": 1920, "height": 1080 },
  "calibration": { "n_points": 9, "completed": true, "points": [...] },
  "duration_s": 42.7, "n_samples_raw": 1280, "n_samples_filtered": 1190,
  "gaze_samples": [ { "t": 0.12, "x": 940, "y": 510, "x_norm": 0.49, "y_norm": 0.47 } ],
  "heatmap_64x64": [ [ ... ] ]
}
```

---

## 7. Cómo ejecutar

### A. Solo el experimento (recolectar datos, sin detectores)

`server.py` exige torch, así que para correr **sin** los detectores usa el venv
con torch igualmente, o usa la app vía WSL. La ruta soportada es:

### B. Servidor completo en WSL (con detectores e inferencia)

```bash
# desde WSL
cd /mnt/c/Users/mitsu/Desktop/EyeSynth
bash run-wsl.sh                 # crea .venv-wsl, instala deps, arranca
bash run-wsl.sh --reinstall     # fuerza reinstalación de dependencias
```

Variables de entorno:
- `NPR_CKPT`, `SIMPLE_CKPT`, `COCO_CKPT` — rutas de los checkpoints.
- `EYESYNTH_KEEP_MODELS` — `1` mantiene modelos en RAM (rápido), `0` por petición.

Luego abrir **http://localhost:5000/experiment**.

### C. Entrenar / reentrenar los detectores

Dataset esperado: `dataset_new_model/fake/*` y `dataset_new_model/real/*`.

```bash
python server/train_simple_fakereal.py    # rápido, sin GPU → simple_fakereal.json
python server/train_npr_fakereal.py       # fine-tuning ResNet-50 NPR → npr_fakereal_v2.pth
python server/train_coco_fakereal.py      # backbone COCO → coco_fakereal.pth
```

> ⚠️ **Ojo con el nombre del checkpoint NPR:** `train_npr_fakereal.py` escribe por
> defecto en `npr_fakereal_v2.pth`, pero la inferencia (`models_infer.py`) carga
> `npr_fakereal.pth`. Tras entrenar, renombra el archivo o exporta
> `NPR_CKPT=.../npr_fakereal_v2.pth` para que el server use el nuevo peso.

### D. Análisis de las sesiones recolectadas

```bash
python analysis/eyesynth_analysis.py        # sesiones single (todas)
python analysis/aoi_analysis.py             # análisis por zonas
python analysis/render_attention_heatmaps.py  # heatmaps del experimento
python analysis/render_attention_points.py    # scanpaths del experimento
python analysis/export_attention_points.py    # exporta puntos por usuario/imagen
```

---

## 8. Requisitos

- **Python 3.8+**; para los detectores: `torch`, `torchvision`, `opencv-python`.
- Resto: `flask`, `flask-cors`, `numpy`, `scipy`, `matplotlib`, `pillow`
  (ver [requirements.txt](requirements.txt)).
- **Cámara web** y navegador con WebGL (Chrome / Edge / Firefox).
- Para la inferencia: **WSL/Linux** (torch no se instala en el venv ligero de Windows).

---

## 9. Notas de diseño

- **`localhost` obligatorio**: `getUserMedia` (cámara) necesita contexto seguro.
- **El estado de eye-tracking vive 100 % en el navegador**: WebGazer corre en JS;
  el servidor solo guarda JSON y hace la inferencia de los modelos.
- **Filtrado en cliente**: las muestras al borde se descartan antes de enviarse.
- **Dos apps en un mismo servidor**: `/` (estímulo único, legado) y `/experiment`
  (el experimento real de detección de fakes 2×2).
- **Tres detectores complementarios**: alta frecuencia (SIMPLE), artefactos de
  generación (NPR) y semántica de objetos (COCO) para contrastar qué mira cada uno
  frente a lo que mira un humano.
