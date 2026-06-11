#!/usr/bin/env python3
"""EyeSynth - mapas de calor (heatmaps) de la atencion sobre las imagenes.

Recorre las sesiones de experimento (analysis/experiment_sessions/*.json) y, por
cada participante y cada imagen vista, genera un PNG con:
  - la imagen-estimulo real de fondo (frontend/dataset/images/<version>/<id>.jpg)
  - un mapa de calor difuso encima, mas intenso donde se concentro la mirada.

El heatmap se construye acumulando los puntos de mirada en una rejilla y
suavizandolos con un kernel gaussiano (sin dependencias extra: solo numpy).

Salida en una carpeta nueva:
    analysis/attention_heatmaps/<participante>/<image_id>__<version>.png

No toca los otros scripts: es una visualizacion adicional e independiente.

Ejecutar:
    python analysis/render_attention_heatmaps.py
"""

import json
import os
import sys

# UTF-8 en consola de Windows
try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

# --- Rutas del proyecto (independientes del directorio de trabajo) ---
ANALYSIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(ANALYSIS_DIR)
EXP_SESSIONS_DIR = os.path.join(ANALYSIS_DIR, "experiment_sessions")
IMAGES_DIR = os.path.join(PROJECT_ROOT, "frontend", "dataset", "images")
OUTPUT_DIR = os.path.join(ANALYSIS_DIR, "attention_heatmaps")

# --- Parametros del heatmap ---
GRID = 200        # resolucion de la rejilla (GRID x GRID)
SIGMA = 9.0       # ancho del difuminado gaussiano, en celdas de la rejilla
ALPHA_MAX = 0.75  # opacidad maxima del calor sobre la imagen


def load_session(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def safe_name(value):
    keep = "-_.()"
    return "".join(c if c.isalnum() or c in keep else "_" for c in str(value))


def image_path_for(version, image_id):
    return os.path.join(IMAGES_DIR, version, f"{image_id}.jpg")


def gaussian_kernel(sigma):
    """Kernel gaussiano 2D normalizado (radio = 3*sigma)."""
    radius = max(1, int(round(3 * sigma)))
    ax = np.arange(-radius, radius + 1)
    xx, yy = np.meshgrid(ax, ax)
    k = np.exp(-(xx ** 2 + yy ** 2) / (2.0 * sigma ** 2))
    return k / k.sum()


def convolve2d_same(img, kernel):
    """Convolucion 2D 'same' con FFT (sin scipy)."""
    ih, iw = img.shape
    kh, kw = kernel.shape
    ph, pw = ih + kh - 1, iw + kw - 1
    fimg = np.fft.rfft2(img, s=(ph, pw))
    fker = np.fft.rfft2(kernel, s=(ph, pw))
    full = np.fft.irfft2(fimg * fker, s=(ph, pw))
    # Recorta al tamano original, centrando el kernel.
    sy, sx = kh // 2, kw // 2
    return full[sy:sy + ih, sx:sx + iw]


def build_heatmap(samples):
    """Acumula los puntos (u,v) en una rejilla y la suaviza. Devuelve (H, n)."""
    grid = np.zeros((GRID, GRID), dtype=float)
    n = 0
    for s in samples:
        u = s.get("u")
        v = s.get("v")
        if u is None or v is None:
            u, v = s.get("x_norm"), s.get("y_norm")
        if u is None or v is None:
            continue
        # Solo acumulamos lo que cae dentro de la imagen [0,1]x[0,1].
        if not (0.0 <= u < 1.0 and 0.0 <= v < 1.0):
            continue
        col = min(GRID - 1, int(u * GRID))
        row = min(GRID - 1, int(v * GRID))
        grid[row, col] += 1.0
        n += 1

    if n == 0:
        return None, 0

    heat = convolve2d_same(grid, gaussian_kernel(SIGMA))
    # La convolucion por FFT puede dejar valores minusculos negativos (ruido
    # numerico); los recortamos a 0 antes de normalizar.
    heat = np.clip(heat, 0.0, None)
    m = heat.max()
    if m > 0:
        heat = heat / m  # normaliza a 0..1
    return heat, n


def render_heatmap(participant, st, img, out_path):
    """Dibuja el heatmap de una imagen del set. True si se genero."""
    samples = img.get("gaze_samples", [])
    if not samples:
        return False

    version = img.get("version", "version")
    image_id = st.get("image_id", "img")

    heat, n = build_heatmap(samples)
    if heat is None:
        return False

    fig, ax = plt.subplots(figsize=(8, 8))
    fig.patch.set_facecolor("#07090f")
    ax.set_facecolor("#07090f")

    # Fondo: la imagen real, mapeada al cuadrado [0,1]x[0,1].
    img_file = image_path_for(version, image_id)
    if HAS_PIL and os.path.exists(img_file):
        bg = Image.open(img_file).convert("RGB")
        ax.imshow(bg, extent=[0, 1, 1, 0])  # y invertido: 0 arriba
    else:
        ax.add_patch(Rectangle((0, 0), 1, 1, color="#11161f"))
        ax.text(0.5, 0.5, f"(imagen no encontrada)\n{version}/{image_id}.jpg",
                ha="center", va="center", color="#7d8a99")

    # Heatmap encima, con transparencia proporcional a la intensidad:
    # donde no hubo mirada, se ve la imagen; donde si, domina el calor.
    alpha_layer = ALPHA_MAX * heat
    ax.imshow(heat, extent=[0, 1, 1, 0], cmap="inferno",
              alpha=alpha_layer, interpolation="bilinear", zorder=2)

    ax.set_xlim(0, 1)
    ax.set_ylim(1, 0)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(
        f"{participant} - {image_id} [{version}]\n"
        f"heatmap de {n} punto(s) dentro de la imagen",
        color="#00d4f0", fontsize=12, pad=12)

    fig.tight_layout()
    fig.savefig(out_path, dpi=110, facecolor="#07090f")
    plt.close(fig)
    return True


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not os.path.isdir(EXP_SESSIONS_DIR):
        print("No existe la carpeta de sesiones:", EXP_SESSIONS_DIR)
        return

    paths = [os.path.join(EXP_SESSIONS_DIR, n)
             for n in sorted(os.listdir(EXP_SESSIONS_DIR)) if n.endswith(".json")]

    print("=" * 60)
    print(f"  EyeSynth - heatmaps de atencion de {len(paths)} sesion(es)")
    print("=" * 60)
    if not paths:
        print("  No hay sesiones en", EXP_SESSIONS_DIR)
        return

    total_imgs = 0
    for p in paths:
        sess = load_session(p)
        participant = sess.get("participant_id") or "desconocido"
        part_dir = os.path.join(OUTPUT_DIR, safe_name(participant))
        os.makedirs(part_dir, exist_ok=True)

        n_made = 0
        for st in sess.get("sets", []):
            for img in st.get("images", []):
                fname = (f"{safe_name(st.get('image_id', 'img'))}__"
                         f"{safe_name(img.get('version', 'version'))}.png")
                out_path = os.path.join(part_dir, fname)
                if render_heatmap(participant, st, img, out_path):
                    n_made += 1

        total_imgs += n_made
        print(f"  + {participant:<12} {os.path.basename(p)} -> {n_made} heatmap(s)")

    print("-" * 60)
    print(f"  Heatmaps generados: {total_imgs}")
    print(f"  Carpeta de salida -> {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
