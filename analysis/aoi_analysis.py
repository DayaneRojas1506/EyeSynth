#!/usr/bin/env python3
"""EyeSynth - analisis por zonas (AOI).

Para cada sesion guardada:
  1. Lee la imagen-estimulo que se mostro (campo "stimulus").
  2. Divide la imagen en una rejilla de ROWS x COLS zonas.
  3. Cuenta cuantas muestras de mirada cayeron en cada zona.
  4. Genera una figura: imagen + rejilla + conteos/porcentajes.
  5. Guarda un resumen CSV/JSON con los numeros.

No toca eyesynth_analysis.py: es un analisis adicional e independiente.

Ejecutar:
    python analysis/aoi_analysis.py
"""

import csv
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
SESSIONS_DIR = os.path.join(ANALYSIS_DIR, "sessions")
STIMULI_DIR = os.path.join(PROJECT_ROOT, "frontend", "stimuli")
FIGURES_DIR = os.path.join(PROJECT_ROOT, "results", "figures")

# --- Configuracion de la rejilla de zonas (AOI) ---
ROWS = 3          # filas de la rejilla
COLS = 3          # columnas de la rejilla

# Etiquetas opcionales por celda (orden: fila por fila, de izq->der, arriba->abajo).
# Si tiene exactamente ROWS*COLS nombres, se usan; si no, se numeran Z1..Zn.
AOI_LABELS = [
    "Sup-Izq", "Sup-Centro", "Sup-Der",
    "Med-Izq", "Med-Centro", "Med-Der",
    "Inf-Izq", "Inf-Centro", "Inf-Der",
]


def load_session(path):
    """Carga un JSON de sesion."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def cell_label(idx):
    if len(AOI_LABELS) == ROWS * COLS:
        return AOI_LABELS[idx]
    return f"Z{idx + 1}"


def count_gazes_per_cell(session):
    """Devuelve (counts ROWSxCOLS, n_on_image, n_off_image).

    Mapea cada muestra (en coords normalizadas de pantalla) al espacio de la
    imagen usando el rectangulo donde se mostro, y la asigna a una celda.
    """
    counts = np.zeros((ROWS, COLS), dtype=int)
    n_on, n_off = 0, 0

    stim = session.get("stimulus") or {}
    rect = stim.get("rect")  # {x_norm, y_norm, w_norm, h_norm}

    samples = session.get("gaze_samples", [])
    for s in samples:
        xn = s.get("x_norm")
        yn = s.get("y_norm")
        if xn is None or yn is None:
            continue

        if rect:
            rx, ry = rect["x_norm"], rect["y_norm"]
            rw, rh = rect["w_norm"], rect["h_norm"]
            if rw <= 0 or rh <= 0:
                continue
            u = (xn - rx) / rw   # 0..1 dentro de la imagen
            v = (yn - ry) / rh
        else:
            # Sin rectangulo: se asume que la imagen ocupaba toda la pantalla.
            u, v = xn, yn

        if 0.0 <= u < 1.0 and 0.0 <= v < 1.0:
            col = min(COLS - 1, int(u * COLS))
            row = min(ROWS - 1, int(v * ROWS))
            counts[row, col] += 1
            n_on += 1
        else:
            n_off += 1

    return counts, n_on, n_off


def figure_aoi(session, counts, n_on, n_off, out_path):
    """Dibuja la imagen con la rejilla, conteos y porcentajes."""
    stim = session.get("stimulus") or {}
    img_file = stim.get("file")
    img_path = os.path.join(STIMULI_DIR, img_file) if img_file else None

    fig, ax = plt.subplots(figsize=(10, 7))
    fig.patch.set_facecolor("#07090f")
    ax.set_facecolor("#07090f")

    # Fondo: la imagen-estimulo si existe; si no, un lienzo neutro.
    if img_path and HAS_PIL and os.path.exists(img_path):
        img = Image.open(img_path).convert("RGB")
        ax.imshow(img, extent=[0, 1, 1, 0])  # y invertido: 0 arriba
    else:
        ax.add_patch(Rectangle((0, 0), 1, 1, color="#11161f"))
        if img_file:
            ax.text(0.5, 0.5, f"(imagen no encontrada:\n{img_file})",
                    ha="center", va="center", color="#7d8a99")

    total = max(1, n_on)
    cmax = max(1, counts.max())

    for r in range(ROWS):
        for c in range(COLS):
            x0, y0 = c / COLS, r / ROWS
            w, h = 1 / COLS, 1 / ROWS
            n = int(counts[r, c])
            pct = 100.0 * n / total
            # Sombreado proporcional al conteo (mas miradas = mas brillante).
            alpha = 0.15 + 0.55 * (n / cmax)
            ax.add_patch(Rectangle((x0, y0), w, h, facecolor="#00d4f0",
                                   alpha=alpha, edgecolor="#00d4f0", lw=1.5))
            idx = r * COLS + c
            ax.text(x0 + w / 2, y0 + h / 2,
                    f"{cell_label(idx)}\n{n}  ({pct:.0f}%)",
                    ha="center", va="center", color="white",
                    fontsize=11, fontweight="bold")

    ax.set_xlim(0, 1)
    ax.set_ylim(1, 0)
    ax.set_xticks([])
    ax.set_yticks([])
    sid = session.get("session_id", "sesion")
    ax.set_title(f"AOI {ROWS}x{COLS} - {sid}\n"
                 f"{n_on} muestras en la imagen, {n_off} fuera",
                 color="#00d4f0", fontsize=13, pad=14)

    fig.tight_layout()
    fig.savefig(out_path, dpi=110, facecolor="#07090f")
    plt.close(fig)


def main():
    os.makedirs(SESSIONS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    paths = [os.path.join(SESSIONS_DIR, n)
             for n in sorted(os.listdir(SESSIONS_DIR)) if n.endswith(".json")]

    print("=" * 60)
    print(f"  EyeSynth - analisis AOI de {len(paths)} sesion(es)")
    print("=" * 60)
    if not paths:
        print("  No hay sesiones en", SESSIONS_DIR)
        return

    summary_rows = []
    for p in paths:
        sess = load_session(p)
        sid = sess.get("session_id", os.path.basename(p).replace(".json", ""))
        counts, n_on, n_off = count_gazes_per_cell(sess)

        out_png = os.path.join(FIGURES_DIR, f"aoi_{sid}.png")
        figure_aoi(sess, counts, n_on, n_off, out_png)

        # zona mas mirada
        flat = counts.flatten()
        top_idx = int(flat.argmax())
        top_label = cell_label(top_idx)
        top_n = int(flat[top_idx])
        stim_file = (sess.get("stimulus") or {}).get("file", "(sin imagen)")

        print(f"  + {sid}  img={stim_file}")
        print(f"      en imagen: {n_on}   fuera: {n_off}   "
              f"zona top: {top_label} ({top_n})")

        for idx in range(ROWS * COLS):
            r, c = divmod(idx, COLS)
            summary_rows.append({
                "session_id": sid,
                "stimulus": stim_file,
                "zona": cell_label(idx),
                "fila": r,
                "col": c,
                "muestras": int(counts[r, c]),
                "pct_en_imagen": round(100.0 * counts[r, c] / max(1, n_on), 2),
            })

    # Resumen CSV
    csv_path = os.path.join(FIGURES_DIR, "aoi_resumen.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    print("-" * 60)
    print(f"  Figuras -> {FIGURES_DIR}")
    print(f"  Resumen -> {csv_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
