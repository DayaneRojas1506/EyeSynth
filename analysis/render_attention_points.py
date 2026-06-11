#!/usr/bin/env python3
"""EyeSynth - dibuja los puntos de atencion SOBRE las imagenes reales.

Recorre las sesiones de experimento (analysis/experiment_sessions/*.json) y, por
cada participante y cada imagen vista, genera un PNG con:
  - la imagen-estimulo real de fondo (frontend/dataset/images/<version>/<id>.jpg)
  - los puntos de mirada dibujados encima, en el espacio de la imagen (u, v)
  - una linea que une los puntos en orden temporal (el recorrido de la mirada)
  - color por orden temporal (inicio -> fin) para ver la secuencia.

Salida en una carpeta nueva:
    analysis/attention_maps/<participante>/<image_id>__<version>.png

No toca los otros scripts: es una visualizacion adicional e independiente.

Ejecutar:
    python analysis/render_attention_points.py
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
OUTPUT_DIR = os.path.join(ANALYSIS_DIR, "attention_maps")


def load_session(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def safe_name(value):
    keep = "-_.()"
    return "".join(c if c.isalnum() or c in keep else "_" for c in str(value))


def image_path_for(version, image_id):
    """Ruta al archivo de imagen real para una version/id."""
    return os.path.join(IMAGES_DIR, version, f"{image_id}.jpg")


def render_image(participant, st, img, out_path):
    """Dibuja los puntos de una imagen del set sobre la imagen real.

    Devuelve True si se genero la figura, False si no habia puntos.
    """
    samples = img.get("gaze_samples", [])
    if not samples:
        return False

    version = img.get("version", "version")
    image_id = st.get("image_id", "img")

    # Coordenadas dentro de la imagen (u, v) en 0..1. Si por algun motivo
    # faltara u/v, se cae a x_norm/y_norm como respaldo.
    us, vs, on_flags = [], [], []
    for s in samples:
        u = s.get("u")
        v = s.get("v")
        if u is None or v is None:
            u, v = s.get("x_norm"), s.get("y_norm")
        if u is None or v is None:
            continue
        us.append(u)
        vs.append(v)
        on_flags.append(bool(s.get("on_image")))

    if not us:
        return False

    us = np.array(us)
    vs = np.array(vs)

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

    # Recorrido de la mirada: linea fina que une los puntos en orden temporal.
    ax.plot(us, vs, "-", color="#00d4f0", lw=1.0, alpha=0.5, zorder=2)

    # Puntos coloreados por orden temporal (inicio -> fin).
    order = np.arange(len(us))
    ax.scatter(us, vs, c=order, cmap="plasma", s=90,
               edgecolors="white", linewidths=0.8, zorder=3)

    # Marca el primer punto (inicio de la mirada).
    ax.scatter([us[0]], [vs[0]], facecolors="none", edgecolors="#39ff14",
               s=220, linewidths=2.0, zorder=4)
    ax.text(us[0], vs[0], " inicio", color="#39ff14", fontsize=9,
            va="center", zorder=5)

    n_on = sum(on_flags)
    n_off = len(on_flags) - n_on

    ax.set_xlim(0, 1)
    ax.set_ylim(1, 0)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(
        f"{participant} - {image_id} [{version}]\n"
        f"{len(us)} puntos  (dentro img: {n_on}, fuera: {n_off})",
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
    print(f"  EyeSynth - mapas de atencion de {len(paths)} sesion(es)")
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
                if render_image(participant, st, img, out_path):
                    n_made += 1

        total_imgs += n_made
        print(f"  + {participant:<12} {os.path.basename(p)} -> {n_made} imagen(es)")

    print("-" * 60)
    print(f"  Mapas generados: {total_imgs}")
    print(f"  Carpeta de salida -> {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
