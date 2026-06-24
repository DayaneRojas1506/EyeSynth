# EyeSynth

Plataforma de **eye-tracking en el navegador** (WebGazer.js) para un experimento de
**detección de imágenes falsas (FAKE/REAL)**, con análisis posterior en Python y
comparación de la **atención humana vs. la atención de modelos de IA** (Grad-CAM y
mapas de saliencia).

El participante ve grillas 2×2 con cuatro versiones de una misma imagen (una real y
tres editadas/generadas) y debe elegir cuál es la original. Mientras decide, se
registra su mirada. Después se contrastan esos patrones de atención con los de tres
detectores de fakes.

> 📖 **Documentación completa y detallada:** [DOCUMENTACION.md](DOCUMENTACION.md)
> (arquitectura, técnicas de eye-tracking, los tres detectores, cómo se
> entrenaron, formatos de datos y análisis).

---

## Estructura

```
EyeSynth/
├── frontend/
│   ├── experiment.html       Experimento principal (grilla 2×2 + eye-tracking)
│   ├── index.html            App de estímulo único (versión antigua)
│   ├── sets_manifest.json    Catálogo de sets de imágenes
│   └── dataset/images/<version>/   Imágenes (4 versiones × 100)
├── server/
│   ├── server.py             Servidor Flask (sirve la app, guarda sesiones, inferencia)
│   ├── models_infer.py       Inferencia + Grad-CAM + métricas de similitud
│   ├── {simple,npr,coco}_model.py        Los 3 detectores FAKE/REAL
│   └── train_{simple,npr,coco}_fakereal.py   Entrenamiento de cada detector
├── analysis/                 Scripts de análisis + sesiones guardadas
├── dataset_new_model/{fake,real}/   Dataset para (re)entrenar los detectores
├── requirements.txt
├── run-wsl.sh                Arranque completo en WSL (con torch + detectores)
└── run.sh / run.bat          Arranque simple (sin detectores)
```

---

## Cómo usarlo

### 1. Arrancar el servidor

El servidor **completo** (con los detectores de IA y la comparación humano-vs-IA)
requiere `torch` y se ejecuta en **WSL/Linux**:

```bash
# desde WSL
cd /mnt/c/Users/mitsu/Desktop/EyeSynth
bash run-wsl.sh            # crea .venv-wsl, instala dependencias y arranca
```

> Esto crea el entorno virtual, instala las dependencias (torch puede tardar la
> primera vez) y levanta el servidor. Usa `bash run-wsl.sh --reinstall` para forzar
> la reinstalación.

### 2. Hacer la sesión de eye-tracking

Abre **http://localhost:5000/experiment** en el navegador y:

1. Acepta el **consentimiento** y completa tus datos (id, género, edad).
2. Permite el acceso a la **cámara web** y encuadra tu rostro.
3. Completa la **calibración** (13 puntos por defecto; mira cada punto y haz clic;
   hay un bloqueo de 1.2 s antes de poder pulsar para que la mirada se estabilice).
4. Pasa el **gate de precisión**: si el error es alto, recalibra antes de empezar.
5. Resuelve los **sets**: en cada grilla 2×2, haz clic en la imagen que creas
   original. Hay límite de tiempo por set y recalibración cada 5 sets.
6. Al terminar, la sesión se guarda automáticamente en
   `analysis/experiment_sessions/exp_<participante>_<fecha>.json`.

### 3. Analizar las sesiones

```bash
python analysis/render_attention_heatmaps.py   # heatmaps de atención por imagen
python analysis/render_attention_points.py      # scanpaths sobre cada imagen
python analysis/export_attention_points.py       # exporta puntos por usuario/imagen
python analysis/eyesynth_analysis.py             # análisis de sesiones de estímulo único
python analysis/aoi_analysis.py                  # análisis por zonas (AOI)
```

Las salidas se generan en `analysis/attention_*/` y `results/figures/`.

---

## Los tres detectores FAKE/REAL

| Modelo | Arquitectura | Señal | Mapa de atención |
|---|---|---|---|
| **SIMPLE** | Regresión logística (6 features globales) | Suavidad / alta frecuencia (~87 % val_acc) | Heatmap de detalle local |
| **NPR** | ResNet-50 sobre residuos de píxeles | Artefactos de upsampling de generadores | Grad-CAM (`layer4[-1]`) |
| **COCO** | ResNet-50 con backbone Faster R-CNN | Features semánticas de objeto (baseline) | Grad-CAM (`layer4[-1]`) |

Entrenamiento (en WSL/Linux con torch; dataset en `dataset_new_model/{fake,real}`):

```bash
python server/train_simple_fakereal.py    # → simple_fakereal.json (rápido, sin GPU)
python server/train_npr_fakereal.py       # → npr_fakereal_v2.pth
python server/train_coco_fakereal.py      # → coco_fakereal.pth
```

Detalles de cómo se entrenó cada uno (hiperparámetros, anti-overfitting, linear
probe): ver [DOCUMENTACION.md](DOCUMENTACION.md#4-bis-cómo-se-entrenaron-los-modelos).

---

## Requisitos

- **Python 3.8+**; para los detectores: `torch`, `torchvision`, `opencv-python`.
- Resto de dependencias en [requirements.txt](requirements.txt).
- **Cámara web** y navegador con WebGL (Chrome / Edge / Firefox).
- La inferencia de los modelos se corre en **WSL/Linux** (torch no se instala en el
  venv ligero de Windows).
