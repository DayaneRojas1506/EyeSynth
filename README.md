# EyeSynth

Plataforma local de **eye-tracking en el navegador** (WebGazer.js) con análisis
posterior en Python: heatmaps gaussianos, scanpaths y métricas de saliencia
**NSS / CC / SSIM**.

```
eyesynth/
├── frontend/index.html          App web de eye-tracking (WebGazer.js)
├── server/server.py             Servidor Flask (sirve la app + guarda sesiones)
├── analysis/
│   ├── eyesynth_analysis.py     Análisis y figuras
│   └── sessions/                JSON de cada sesión (se generan solos)
├── models/{npr,univfd}/         (espacio para modelos)
├── results/figures/             PNG generados por el análisis
├── requirements.txt
├── run.sh / run.bat             Instala dependencias y arranca el servidor
└── README.md
```

## Cómo usarlo en 3 pasos

### 1. Arrancar el servidor

**Windows:**
```bat
./run.bat
```

**Mac / Linux:**
```bash
chmod +x run.sh
./run.sh
```

> Esto crea un entorno virtual, instala las dependencias y levanta el servidor.
> Si prefieres hacerlo a mano: `pip install -r requirements.txt` y luego
> `python server/server.py`.

### 2. Hacer la sesión de eye-tracking

Abre **http://localhost:5000/experiment** en el navegador y:

1. Permite el acceso a la **cámara web**.
2. Completa la **calibración de 9 puntos** (mira cada punto y haz clic; hay un
   bloqueo de 1.2 s antes de poder pulsar cada punto para que la mirada se
   estabilice).
3. Observa el estímulo durante la fase de **seguimiento**. Verás el cursor de
   mirada, el heatmap acumulado en vivo y un indicador de calidad
   (te avisa con *"GAZE FUERA DE RANGO · RECALIBRA"* si la señal se va al borde).
4. Pulsa **GUARDAR SESIÓN**. El JSON se guarda automáticamente en
   `analysis/sessions/session_AAAAMMDD_HHMMSS.json`.

### 3. Analizar las sesiones

```bash
python analysis/eyesynth_analysis.py
```

- Sin argumentos: procesa **todas** las sesiones y genera un PNG por sesión en
  `results/figures/`, más un comparativo `resumen_sesiones.png`.
- Con un archivo: `python analysis/eyesynth_analysis.py session_20260101_120000.json`
  procesa solo esa sesión.

## Detalles técnicos

**Formato del JSON de sesión**

```jsonc
{
  "session_id": "session_20260101_120000",
  "timestamp": "2026-01-01T12:00:00.000Z",
  "screen": { "width": 1920, "height": 1080 },
  "calibration": { "n_points": 9, "completed": true, "points": [ ... ] },
  "duration_s": 42.7,
  "n_samples_raw": 1280,
  "n_samples_filtered": 1190,
  "gaze_samples": [ { "t": 0.12, "x": 940, "y": 510, "x_norm": 0.49, "y_norm": 0.47 } ],
  "heatmap_64x64": [ [ ... 64 valores ... ], ... ]
}
```

Las muestras pegadas a los bordes (`x_norm`/`y_norm` ≤ 0.01 o ≥ 0.99) se
filtran en el navegador **antes** de guardar.

**Métricas del análisis**

- **CC** — correlación de Pearson entre el heatmap de la sesión y el mapa
  promedio del grupo.
- **NSS** — Normalized Scanpath Saliency: valor medio del mapa de referencia
  (normalizado a media 0, desv 1) en las posiciones de mirada de la sesión.
- **SSIM** — índice de similitud estructural (ventana gaussiana) frente al mapa
  de referencia.

**Endpoints del servidor**

| Método | Ruta            | Descripción                                   |
|--------|-----------------|-----------------------------------------------|
| GET    | `/`             | Sirve `frontend/index.html`                   |
| POST   | `/save-session` | Guarda el JSON recibido con timestamp         |
| GET    | `/sessions`     | Lista las sesiones guardadas con metadatos    |

## Requisitos

- Python 3.8+
- Cámara web
- Navegador con WebGL (Chrome / Edge / Firefox recomendados)
