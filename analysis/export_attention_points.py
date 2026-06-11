#!/usr/bin/env python3
"""EyeSynth - exporta los puntos de atencion por usuario y por imagen.

Recorre las sesiones de experimento (analysis/experiment_sessions/*.json) y, por
cada participante y cada imagen vista, junta todos los puntos de mirada (gaze
samples) y los guarda en una carpeta nueva:

    analysis/attention_points/<participante>/<image_id>__<version>.json

Cada archivo de salida contiene:
  - participante, image_id, version, source
  - rect donde se mostro la imagen (para mapear a coords de imagen)
  - lista de puntos: t, x, y (pantalla), x_norm/y_norm (pantalla),
    u/v (normalizado 0..1 dentro de la imagen) y on_image.

Tambien guarda, por participante, un resumen CSV con cuantos puntos hay por
imagen/version.

No toca los otros scripts de analisis: es una exportacion adicional e
independiente.

Ejecutar:
    python analysis/export_attention_points.py
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

# --- Rutas del proyecto (independientes del directorio de trabajo) ---
ANALYSIS_DIR = os.path.dirname(os.path.abspath(__file__))
EXP_SESSIONS_DIR = os.path.join(ANALYSIS_DIR, "experiment_sessions")
OUTPUT_DIR = os.path.join(ANALYSIS_DIR, "attention_points")


def load_session(path):
    """Carga un JSON de sesion de experimento."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def safe_name(value):
    """Convierte un texto en un nombre de carpeta/archivo seguro."""
    keep = "-_.()"
    return "".join(c if c.isalnum() or c in keep else "_" for c in str(value))


def collect_points(image):
    """Devuelve la lista de puntos de atencion de una imagen del set.

    Conserva las coordenadas tal cual fueron guardadas: t (tiempo), x/y en
    pixeles de pantalla, x_norm/y_norm normalizados a la pantalla, u/v
    normalizados dentro de la imagen, y on_image.
    """
    points = []
    for s in image.get("gaze_samples", []):
        points.append({
            "t": s.get("t"),
            "x": s.get("x"),
            "y": s.get("y"),
            "x_norm": s.get("x_norm"),
            "y_norm": s.get("y_norm"),
            "u": s.get("u"),
            "v": s.get("v"),
            "on_image": s.get("on_image"),
        })
    return points


def export_session(session):
    """Exporta los puntos de atencion de una sesion. Devuelve filas resumen."""
    participant = session.get("participant_id") or "desconocido"
    session_id = session.get("session_id", "sesion")
    part_dir = os.path.join(OUTPUT_DIR, safe_name(participant))
    os.makedirs(part_dir, exist_ok=True)

    rows = []
    for st in session.get("sets", []):
        image_id = st.get("image_id", "img")
        source = st.get("source")
        set_id = st.get("set_id")

        for img in st.get("images", []):
            version = img.get("version", "version")
            points = collect_points(img)
            # Solo guardamos imagenes que efectivamente se miraron.
            if not points:
                continue

            out = {
                "participant_id": participant,
                "session_id": session_id,
                "set_id": set_id,
                "image_id": image_id,
                "version": version,
                "source": source,
                "url": img.get("url"),
                "rect": img.get("rect"),
                "view_duration_s": img.get("view_duration_s"),
                "n_points": len(points),
                "points": points,
            }

            fname = f"{safe_name(image_id)}__{safe_name(version)}.json"
            out_path = os.path.join(part_dir, fname)
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(out, f, ensure_ascii=False, indent=2)

            on_img = sum(1 for p in points if p.get("on_image"))
            rows.append({
                "participant_id": participant,
                "session_id": session_id,
                "set_id": set_id,
                "image_id": image_id,
                "version": version,
                "source": source,
                "n_points": len(points),
                "n_on_image": on_img,
                "n_off_image": len(points) - on_img,
                "archivo": os.path.relpath(out_path, OUTPUT_DIR),
            })

    # Resumen CSV por participante.
    if rows:
        csv_path = os.path.join(part_dir, "_resumen.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    return rows


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not os.path.isdir(EXP_SESSIONS_DIR):
        print("No existe la carpeta de sesiones:", EXP_SESSIONS_DIR)
        return

    paths = [os.path.join(EXP_SESSIONS_DIR, n)
             for n in sorted(os.listdir(EXP_SESSIONS_DIR)) if n.endswith(".json")]

    print("=" * 60)
    print(f"  EyeSynth - puntos de atencion de {len(paths)} sesion(es)")
    print("=" * 60)
    if not paths:
        print("  No hay sesiones en", EXP_SESSIONS_DIR)
        return

    all_rows = []
    for p in paths:
        sess = load_session(p)
        rows = export_session(sess)
        all_rows.extend(rows)

        participant = sess.get("participant_id") or "desconocido"
        n_imgs = len(rows)
        n_pts = sum(r["n_points"] for r in rows)
        print(f"  + {participant:<12} {os.path.basename(p)}")
        print(f"      {n_imgs} imagen(es) con puntos, {n_pts} puntos en total")

    # Resumen global de todos los participantes.
    if all_rows:
        global_csv = os.path.join(OUTPUT_DIR, "_resumen_global.csv")
        with open(global_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            writer.writeheader()
            writer.writerows(all_rows)
        print("-" * 60)
        print(f"  Carpeta de salida -> {OUTPUT_DIR}")
        print(f"  Resumen global    -> {global_csv}")

    print("=" * 60)


if __name__ == "__main__":
    main()
