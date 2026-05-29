#!/usr/bin/env python3
"""EyeSynth - análisis de sesiones de eye-tracking.

Lee los JSON guardados en analysis/sessions/ y genera, por cada sesión:
  * heatmap gaussiano 512x512
  * scanpath (recorrido de la mirada)
  * perfiles temporales X / Y
  * métricas de la sesión

Al final produce results/figures/resumen_sesiones.png comparando todas
las sesiones entre sí mediante NSS / CC / SSIM contra el mapa promedio
del grupo.

Uso:
    python analysis/eyesynth_analysis.py                # todas las sesiones
    python analysis/eyesynth_analysis.py una_sesion.json  # solo ese archivo
"""

import json
import os
import sys

# en Windows la consola suele usar cp1252; forzamos UTF-8 para los mensajes
try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

import matplotlib

matplotlib.use("Agg")  # backend sin ventana (guarda PNG directamente)
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter

# --- Rutas del proyecto (independientes del directorio de trabajo) ---
ANALYSIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(ANALYSIS_DIR)
SESSIONS_DIR = os.path.join(ANALYSIS_DIR, "sessions")
FIGURES_DIR = os.path.join(PROJECT_ROOT, "results", "figures")

MAP = 512          # resolución del heatmap de análisis
SIGMA = 18.0       # desviación gaussiana (en px del mapa de 512)
EPS = 1e-12


# =========================================================================
#  Carga de datos
# =========================================================================
def load_session(path):
    """Carga un JSON de sesión y normaliza los campos que usaremos."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    samples = data.get("gaze_samples", [])
    xs, ys, ts = [], [], []
    for s in samples:
        xn = s.get("x_norm")
        yn = s.get("y_norm")
        if xn is None or yn is None:
            continue
        if not (0.0 <= xn <= 1.0 and 0.0 <= yn <= 1.0):
            continue
        xs.append(float(xn))
        ys.append(float(yn))
        ts.append(float(s.get("t", len(ts))))

    return {
        "session_id": data.get("session_id", os.path.basename(path)),
        "timestamp": data.get("timestamp", ""),
        "path": path,
        "x": np.array(xs),
        "y": np.array(ys),
        "t": np.array(ts),
        "duration": float(data.get("duration_s", ts[-1] if ts else 0.0)),
        "calibration": data.get("calibration", {}),
        "heatmap_64x64": data.get("heatmap_64x64"),
    }


# =========================================================================
#  Construcción de mapas
# =========================================================================
def build_heatmap(x_norm, y_norm, size=MAP, sigma=SIGMA):
    """Mapa de saliencia gaussiano a partir de coordenadas normalizadas."""
    grid = np.zeros((size, size), dtype=np.float64)
    if len(x_norm) == 0:
        return grid
    xi = np.clip((x_norm * (size - 1)).astype(int), 0, size - 1)
    yi = np.clip((y_norm * (size - 1)).astype(int), 0, size - 1)
    for gx, gy in zip(xi, yi):
        grid[gy, gx] += 1.0
    grid = gaussian_filter(grid, sigma=sigma)
    m = grid.max()
    if m > 0:
        grid /= m
    return grid


def build_fixation_map(x_norm, y_norm, size=MAP):
    """Mapa binario de posiciones de mirada (para NSS)."""
    fix = np.zeros((size, size), dtype=np.float64)
    if len(x_norm) == 0:
        return fix
    xi = np.clip((x_norm * (size - 1)).astype(int), 0, size - 1)
    yi = np.clip((y_norm * (size - 1)).astype(int), 0, size - 1)
    fix[yi, xi] = 1.0
    return fix


# =========================================================================
#  Métricas (NSS / CC / SSIM)
# =========================================================================
def metric_cc(map_a, map_b):
    """Coeficiente de correlación de Pearson entre dos mapas."""
    a = map_a.ravel().astype(np.float64)
    b = map_b.ravel().astype(np.float64)
    a = a - a.mean()
    b = b - b.mean()
    denom = np.sqrt((a * a).sum() * (b * b).sum())
    if denom < EPS:
        return 0.0
    return float((a * b).sum() / denom)


def metric_nss(sal_map, fixation_map):
    """Normalized Scanpath Saliency.

    Normaliza el mapa de saliencia (media 0, desv 1) y promedia su valor
    en las posiciones de fijación.
    """
    sal = sal_map.astype(np.float64)
    std = sal.std()
    if std < EPS:
        return 0.0
    sal = (sal - sal.mean()) / std
    mask = fixation_map > 0
    if mask.sum() == 0:
        return 0.0
    return float(sal[mask].mean())


def metric_ssim(map_a, map_b):
    """SSIM con ventana gaussiana (implementación propia, sin skimage)."""
    a = map_a.astype(np.float64)
    b = map_b.astype(np.float64)
    # rango dinámico asumido [0, 1] tras la normalización de los mapas
    L = 1.0
    c1 = (0.01 * L) ** 2
    c2 = (0.03 * L) ** 2
    win = 1.5

    mu_a = gaussian_filter(a, win)
    mu_b = gaussian_filter(b, win)
    mu_a2, mu_b2, mu_ab = mu_a * mu_a, mu_b * mu_b, mu_a * mu_b

    sigma_a2 = gaussian_filter(a * a, win) - mu_a2
    sigma_b2 = gaussian_filter(b * b, win) - mu_b2
    sigma_ab = gaussian_filter(a * b, win) - mu_ab

    num = (2 * mu_ab + c1) * (2 * sigma_ab + c2)
    den = (mu_a2 + mu_b2 + c1) * (sigma_a2 + sigma_b2 + c2)
    ssim_map = num / (den + EPS)
    return float(ssim_map.mean())


def session_metrics(sess, heatmap):
    """Métricas descriptivas de una sesión individual."""
    x, y = sess["x"], sess["y"]
    n = len(x)
    if n == 0:
        return {
            "n_samples": 0, "duration": sess["duration"], "hz": 0.0,
            "x_mean": 0.0, "y_mean": 0.0, "dispersion": 0.0,
            "coverage": 0.0, "path_len": 0.0,
        }
    # dispersión: desviación radial respecto al centroide
    cx, cy = x.mean(), y.mean()
    dispersion = float(np.sqrt(((x - cx) ** 2 + (y - cy) ** 2).mean()))
    # cobertura: fracción de celdas 32x32 visitadas
    cells = np.zeros((32, 32), dtype=bool)
    cells[np.clip((y * 31).astype(int), 0, 31),
          np.clip((x * 31).astype(int), 0, 31)] = True
    coverage = float(cells.mean())
    # longitud del recorrido (en unidades normalizadas)
    path_len = float(np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2).sum()) if n > 1 else 0.0
    hz = n / sess["duration"] if sess["duration"] > 0 else 0.0
    return {
        "n_samples": n, "duration": sess["duration"], "hz": hz,
        "x_mean": float(cx), "y_mean": float(cy), "dispersion": dispersion,
        "coverage": coverage, "path_len": path_len,
    }


# =========================================================================
#  Figuras
# =========================================================================
def figure_session(sess, heatmap, metrics, out_path):
    """Genera la figura de análisis de una sesión y la guarda."""
    x, y, t = sess["x"], sess["y"], sess["t"]
    fig = plt.figure(figsize=(14, 9), facecolor="#07090f")
    fig.suptitle(
        f"EyeSynth · {sess['session_id']}",
        color="#00d4f0", fontsize=16, fontweight="bold", y=0.98,
    )
    gs = fig.add_gridspec(2, 3, hspace=0.32, wspace=0.28,
                          left=0.06, right=0.97, top=0.90, bottom=0.08)

    # --- Heatmap ---
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(heatmap, cmap="inferno", origin="upper", extent=[0, 1, 1, 0])
    ax1.set_title("Heatmap gaussiano 512×512", color="#d8e2ec")
    _style_axes(ax1)

    # --- Scanpath ---
    ax2 = fig.add_subplot(gs[0, 1])
    if len(x) > 0:
        ax2.plot(x, y, "-", color="#00d4f0", alpha=0.35, lw=0.8)
        ax2.scatter(x, y, c=t if len(t) == len(x) else None,
                    cmap="cool", s=10, alpha=0.8)
        ax2.scatter([x[0]], [y[0]], c="#2fe089", s=60, marker="o", label="inicio")
        ax2.scatter([x[-1]], [y[-1]], c="#ff4d5e", s=60, marker="X", label="fin")
        ax2.legend(facecolor="#11161f", edgecolor="#1c2530", labelcolor="#d8e2ec",
                   fontsize=8, loc="upper right")
    ax2.set_xlim(0, 1); ax2.set_ylim(1, 0)
    ax2.set_title("Scanpath (recorrido)", color="#d8e2ec")
    _style_axes(ax2)

    # --- Métricas ---
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.axis("off")
    lines = [
        ("Muestras", f"{metrics['n_samples']}"),
        ("Duración", f"{metrics['duration']:.1f} s"),
        ("Frecuencia", f"{metrics['hz']:.0f} Hz"),
        ("Centro X", f"{metrics['x_mean']:.3f}"),
        ("Centro Y", f"{metrics['y_mean']:.3f}"),
        ("Dispersión", f"{metrics['dispersion']:.3f}"),
        ("Cobertura", f"{metrics['coverage'] * 100:.1f} %"),
        ("Long. recorrido", f"{metrics['path_len']:.2f}"),
    ]
    ax3.set_title("Métricas de sesión", color="#d8e2ec")
    for i, (k, v) in enumerate(lines):
        yy = 0.9 - i * 0.11
        ax3.text(0.02, yy, k, color="#7d8a99", fontsize=11, transform=ax3.transAxes)
        ax3.text(0.98, yy, v, color="#00d4f0", fontsize=11, fontweight="bold",
                 ha="right", transform=ax3.transAxes)

    # --- Perfil X temporal ---
    ax4 = fig.add_subplot(gs[1, 0:2])
    if len(x) > 0:
        ax4.plot(t, x, color="#00d4f0", lw=1.0, label="x_norm")
        ax4.plot(t, y, color="#ff8a3d", lw=1.0, label="y_norm")
        ax4.legend(facecolor="#11161f", edgecolor="#1c2530", labelcolor="#d8e2ec",
                   fontsize=8, loc="upper right")
    ax4.set_xlabel("tiempo (s)", color="#7d8a99")
    ax4.set_ylim(0, 1)
    ax4.set_title("Perfiles temporales X / Y", color="#d8e2ec")
    _style_axes(ax4)

    # --- Histograma de cobertura ---
    ax5 = fig.add_subplot(gs[1, 2])
    if len(x) > 0:
        ax5.hist2d(x, y, bins=24, range=[[0, 1], [0, 1]], cmap="viridis")
    ax5.set_ylim(1, 0)
    ax5.set_title("Densidad de mirada", color="#d8e2ec")
    _style_axes(ax5)

    fig.savefig(out_path, dpi=110, facecolor="#07090f")
    plt.close(fig)


def _style_axes(ax):
    ax.set_facecolor("#0d1119")
    ax.tick_params(colors="#7d8a99", labelsize=8)
    for spine in ax.spines.values():
        spine.set_color("#1c2530")


def figure_summary(sessions, heatmaps, fixmaps, metrics_all, out_path):
    """Reporte comparativo de todas las sesiones (NSS / CC / SSIM)."""
    n = len(sessions)
    # mapa de referencia = promedio del grupo
    ref = np.mean(heatmaps, axis=0)
    rm = ref.max()
    if rm > 0:
        ref = ref / rm

    names, ccs, nsss, ssims = [], [], [], []
    for sess, hm, fx in zip(sessions, heatmaps, fixmaps):
        names.append(sess["session_id"].replace("session_", ""))
        ccs.append(metric_cc(ref, hm))
        nsss.append(metric_nss(ref, fx))
        ssims.append(metric_ssim(ref, hm))

    fig = plt.figure(figsize=(15, 9), facecolor="#07090f")
    fig.suptitle("EyeSynth · Resumen comparativo de sesiones",
                 color="#00d4f0", fontsize=17, fontweight="bold", y=0.98)
    gs = fig.add_gridspec(2, 3, hspace=0.38, wspace=0.3,
                          left=0.06, right=0.97, top=0.90, bottom=0.16)

    # --- Mapa de referencia (promedio) ---
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.imshow(ref, cmap="inferno", origin="upper", extent=[0, 1, 1, 0])
    ax0.set_title(f"Mapa de referencia (n={n})", color="#d8e2ec")
    _style_axes(ax0)

    idx = np.arange(n)

    def _bar(ax, vals, title, color):
        ax.bar(idx, vals, color=color, alpha=0.85)
        ax.set_title(title, color="#d8e2ec")
        ax.set_xticks(idx)
        ax.set_xticklabels(names, rotation=45, ha="right", fontsize=7)
        _style_axes(ax)

    _bar(fig.add_subplot(gs[0, 1]), ccs, "CC vs. referencia", "#00d4f0")
    _bar(fig.add_subplot(gs[0, 2]), nsss, "NSS vs. referencia", "#2fe089")
    _bar(fig.add_subplot(gs[1, 0]), ssims, "SSIM vs. referencia", "#ff8a3d")

    # --- Dispersión vs. cobertura ---
    ax4 = fig.add_subplot(gs[1, 1])
    disp = [m["dispersion"] for m in metrics_all]
    cov = [m["coverage"] * 100 for m in metrics_all]
    ax4.scatter(disp, cov, c="#00d4f0", s=60)
    for i, nm in enumerate(names):
        ax4.annotate(nm, (disp[i], cov[i]), color="#7d8a99", fontsize=7,
                     xytext=(4, 4), textcoords="offset points")
    ax4.set_xlabel("dispersión", color="#7d8a99")
    ax4.set_ylabel("cobertura (%)", color="#7d8a99")
    ax4.set_title("Dispersión vs. cobertura", color="#d8e2ec")
    _style_axes(ax4)

    # --- Tabla resumen ---
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.axis("off")
    ax5.set_title("Resumen numérico", color="#d8e2ec")
    header = f"{'sesión':<10}{'CC':>7}{'NSS':>7}{'SSIM':>7}"
    ax5.text(0.0, 0.95, header, color="#00d4f0", fontsize=9,
             family="monospace", transform=ax5.transAxes)
    for i in range(n):
        row = f"{names[i][:9]:<10}{ccs[i]:>7.2f}{nsss[i]:>7.2f}{ssims[i]:>7.2f}"
        ax5.text(0.0, 0.88 - i * 0.08, row, color="#d8e2ec", fontsize=9,
                 family="monospace", transform=ax5.transAxes)
        if i >= 9:
            ax5.text(0.0, 0.88 - 10 * 0.08, "…", color="#7d8a99",
                     fontsize=9, transform=ax5.transAxes)
            break

    fig.savefig(out_path, dpi=110, facecolor="#07090f")
    plt.close(fig)


# =========================================================================
#  Orquestación
# =========================================================================
def gather_paths(arg):
    if arg:
        # acepta ruta absoluta, relativa, o solo el nombre dentro de sessions/
        cand = arg if os.path.isabs(arg) else os.path.join(os.getcwd(), arg)
        if not os.path.exists(cand):
            cand = os.path.join(SESSIONS_DIR, os.path.basename(arg))
        if not os.path.exists(cand):
            print(f"[EyeSynth] No se encontró el archivo: {arg}")
            sys.exit(1)
        return [cand]
    paths = sorted(
        os.path.join(SESSIONS_DIR, n)
        for n in os.listdir(SESSIONS_DIR)
        if n.endswith(".json")
    )
    return paths


def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)
    arg = sys.argv[1] if len(sys.argv) > 1 else None
    paths = gather_paths(arg)

    if not paths:
        print(f"[EyeSynth] No hay sesiones en {SESSIONS_DIR}")
        print("           Ejecuta el servidor y guarda al menos una sesión.")
        return

    print("=" * 60)
    print(f"  EyeSynth · análisis de {len(paths)} sesión(es)")
    print("=" * 60)

    sessions, heatmaps, fixmaps, metrics_all = [], [], [], []
    for path in paths:
        sess = load_session(path)
        hm = build_heatmap(sess["x"], sess["y"])
        fx = build_fixation_map(sess["x"], sess["y"])
        metrics = session_metrics(sess, hm)

        out_png = os.path.join(FIGURES_DIR, sess["session_id"] + ".png")
        figure_session(sess, hm, metrics, out_png)
        print(f"  ✓ {sess['session_id']:<28} "
              f"{metrics['n_samples']:>5} muestras  ->  {os.path.basename(out_png)}")

        sessions.append(sess)
        heatmaps.append(hm)
        fixmaps.append(fx)
        metrics_all.append(metrics)

    # reporte resumen (solo tiene sentido comparar si hay datos)
    summary_png = os.path.join(FIGURES_DIR, "resumen_sesiones.png")
    figure_summary(sessions, heatmaps, fixmaps, metrics_all, summary_png)
    print("-" * 60)
    print(f"  Resumen comparativo -> {os.path.basename(summary_png)}")
    print(f"  Figuras guardadas en: {FIGURES_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
