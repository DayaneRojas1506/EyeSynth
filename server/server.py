#!/usr/bin/env python3
"""EyeSynth - servidor local.

Sirve el frontend de eye-tracking y guarda las sesiones en disco.

Ejecutar:
    python server/server.py

Luego abrir http://localhost:5000 en el navegador.
"""

import json
import os
from datetime import datetime

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

# --- Rutas del proyecto (independientes del directorio de trabajo) ---
SERVER_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SERVER_DIR)
FRONTEND_DIR = os.path.join(PROJECT_ROOT, "frontend")
SESSIONS_DIR = os.path.join(PROJECT_ROOT, "analysis", "sessions")

os.makedirs(SESSIONS_DIR, exist_ok=True)

app = Flask(__name__, static_folder=FRONTEND_DIR, static_url_path="")
CORS(app, resources={r"/*": {"origins": "*"}})


@app.route("/")
def index():
    """Sirve la aplicación web de eye-tracking."""
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.route("/save-session", methods=["POST"])
def save_session():
    """Recibe el JSON de una sesión y lo guarda con timestamp."""
    data = request.get_json(silent=True)
    if data is None:
        return jsonify({"ok": False, "error": "JSON inválido o ausente"}), 400

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"session_{timestamp}.json"
    path = os.path.join(SESSIONS_DIR, filename)

    # Garantiza que el id y el timestamp queden registrados en el archivo.
    data.setdefault("session_id", filename.replace(".json", ""))
    data.setdefault("timestamp", datetime.now().isoformat())

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    n_samples = len(data.get("gaze_samples", []))
    print(f"[EyeSynth] Sesión guardada: {filename} ({n_samples} muestras)")

    return jsonify({"ok": True, "filename": filename, "samples": n_samples})

@app.route("/stimuli")
def list_stimuli():
    """Lista las imágenes disponibles en frontend/stimuli/."""
    exts = (".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp")
    stim_dir = os.path.join(FRONTEND_DIR, "stimuli")
    os.makedirs(stim_dir, exist_ok=True)
    files = [n for n in sorted(os.listdir(stim_dir)) if n.lower().endswith(exts)]
    return jsonify({"ok": True, "files": files})

@app.route("/sessions", methods=["GET"])
def list_sessions():
    """Lista todas las sesiones guardadas con metadatos básicos."""
    sessions = []
    for name in sorted(os.listdir(SESSIONS_DIR)):
        if not name.endswith(".json"):
            continue
        path = os.path.join(SESSIONS_DIR, name)
        meta = {
            "filename": name,
            "size_bytes": os.path.getsize(path),
            "modified": datetime.fromtimestamp(os.path.getmtime(path)).isoformat(),
        }
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            meta["session_id"] = data.get("session_id")
            meta["timestamp"] = data.get("timestamp")
            meta["n_samples"] = len(data.get("gaze_samples", []))
        except (json.JSONDecodeError, OSError) as exc:
            meta["error"] = str(exc)
        sessions.append(meta)

    return jsonify({"ok": True, "count": len(sessions), "sessions": sessions})


if __name__ == "__main__":
    print("=" * 56)
    print("  EyeSynth - servidor de eye-tracking")
    print("  Frontend : http://localhost:5000")
    print(f"  Sesiones : {SESSIONS_DIR}")
    print("=" * 56)
    app.run(host="0.0.0.0", port=5000, debug=False)
