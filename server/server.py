#!/usr/bin/env python3
"""EyeSynth - servidor local.

Sirve el frontend de eye-tracking y guarda las sesiones en disco.

Ejecutar:
    python server/server.py

Luego abrir http://localhost:5000 en el navegador.
"""

import json
import os
import sys
import threading
import webbrowser
from datetime import datetime

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

# --- Inferencia del modelo único de detección de fakes (obligatoria) ---
# Por diseño el server "exige" torch: si falta, se aborta con un mensaje claro.
# Debe correrse en WSL/Linux con el entorno que lo tiene instalado.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from models_infer import ModelBundle
except ImportError as exc:  # noqa: BLE001
    sys.stderr.write(
        "\n[EyeSynth] ERROR: no se pudo importar la inferencia del modelo.\n"
        f"  Detalle: {exc}\n"
        "  Este server necesita torch y torchvision, y debe correrse en\n"
        "  WSL/Linux (no en el venv ligero de Windows). Usa run-wsl.sh.\n\n"
    )
    raise

# Instancia global. Los modelos se cargan explícitamente: bajo __main__ en el
# arranque directo; bajo WSGI se debe llamar BUNDLE.load_all() en el punto de
# entrada del runner (p.ej. en un módulo wsgi.py) antes de recibir peticiones.
BUNDLE = ModelBundle()
if __name__ != "__main__":
    # Carga preventiva para despliegues WSGI (gunicorn, uWSGI, etc.) donde el
    # bloque __main__ nunca se ejecuta. Si los pesos no están disponibles se
    # lanza aquí al importar, no silenciosamente en la primera petición.
    BUNDLE.load_all()

# --- Rutas del proyecto (independientes del directorio de trabajo) ---
SERVER_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SERVER_DIR)
FRONTEND_DIR = os.path.join(PROJECT_ROOT, "frontend")
SESSIONS_DIR = os.path.join(PROJECT_ROOT, "analysis", "sessions")
EXPERIMENT_SESSIONS_DIR = os.path.join(PROJECT_ROOT, "analysis", "experiment_sessions")

os.makedirs(SESSIONS_DIR, exist_ok=True)
os.makedirs(EXPERIMENT_SESSIONS_DIR, exist_ok=True)

app = Flask(__name__, static_folder=FRONTEND_DIR, static_url_path="")
# JSON estricto: NaN/Infinity NO son JSON válido. Si una respuesta los contiene,
# preferimos un 500 explícito (jsonify lanzará ValueError) antes que emitir el
# literal `NaN` con HTTP 200 — eso rompe res.json() en el navegador y aparece
# como "respuesta no-JSON (200)" sin pista del motivo real.
app.json.allow_nan = False
CORS(app, resources={r"/*": {"origins": "*"}})


@app.route("/")
def index():
    """Sirve la aplicación web de eye-tracking (estímulo único)."""
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.route("/experiment")
def experiment():
    """Sirve la app del experimento (grilla 2x2 de detección)."""
    return send_from_directory(FRONTEND_DIR, "experiment.html")


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


@app.route("/save-experiment", methods=["POST"])
def save_experiment():
    """Recibe el JSON de una sesión del experimento (grilla 2x2) y lo guarda."""
    data = request.get_json(silent=True)
    if data is None:
        return jsonify({"ok": False, "error": "JSON inválido o ausente"}), 400

    pid = (data.get("participant_id") or "anon").replace(" ", "_")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"exp_{pid}_{timestamp}.json"
    path = os.path.join(EXPERIMENT_SESSIONS_DIR, filename)

    data.setdefault("session_id", filename.replace(".json", ""))
    data.setdefault("timestamp", datetime.now().isoformat())

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    n_sets = len(data.get("sets", []))
    print(f"[EyeSynth] Experimento guardado: {filename} ({n_sets} sets)")

    return jsonify({"ok": True, "filename": filename, "sets": n_sets})


@app.route("/compare-models", methods=["POST"])
def compare_models():
    """Compara el heatmap de mirada humana con el mapa del modelo para la imagen
    elegida en un set. Devuelve, por modelo, un overlay PNG (base64), la
    predicción/confianza y las métricas de similitud (CC/SIM/KL)."""
    data = request.get_json(silent=True)
    if data is None:
        return jsonify({"ok": False, "error": "JSON inválido o ausente"}), 400

    version = data.get("version")
    image_id = data.get("image_id")
    gaze_samples = data.get("gaze_samples", [])
    if not version or not image_id:
        return jsonify({"ok": False, "error": "Faltan 'version' o 'image_id'"}), 400

    # Lista blanca de versiones válidas: evita path traversal con valores como
    # "../../etc". Solo se permiten los directorios conocidos bajo dataset/images/.
    _VALID_VERSIONS = {"real", "fake", "original", "semantic_aware",
                       "semantic_agnostic", "pix2pix_magicbrush"}
    if version not in _VALID_VERSIONS:
        return jsonify({"ok": False, "error": f"Versión no permitida: {version!r}"}), 400

    # La imagen puede ser .jpg (dataset antiguo) o .png (sets real/fake del nuevo
    # dataset). Probamos las extensiones soportadas y usamos la que exista.
    version_dir = os.path.join(FRONTEND_DIR, "dataset", "images", version)
    img_path = None
    for ext in (".png", ".jpg", ".jpeg", ".webp"):
        cand = os.path.join(version_dir, f"{image_id}{ext}")
        if os.path.exists(cand):
            img_path = cand
            break
    if img_path is None:
        return jsonify({"ok": False, "error": f"Imagen no encontrada: {version}/{image_id}.*"}), 404

    try:
        from PIL import Image
        pil_img = Image.open(img_path).convert("RGB")
        result = BUNDLE.compare(pil_img, gaze_samples)
    except Exception as exc:  # noqa: BLE001
        return jsonify({"ok": False, "error": f"Error de inferencia: {exc}"}), 500

    result["ok"] = True
    result["version"] = version
    result["image_id"] = image_id
    return jsonify(result)


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


@app.route("/experiment-sessions", methods=["GET"])
def list_experiment_sessions():
    """Lista las sesiones del experimento guardadas con metadatos básicos."""
    sessions = []
    for name in sorted(os.listdir(EXPERIMENT_SESSIONS_DIR)):
        if not name.endswith(".json"):
            continue
        path = os.path.join(EXPERIMENT_SESSIONS_DIR, name)
        meta = {
            "filename": name,
            "size_bytes": os.path.getsize(path),
            "modified": datetime.fromtimestamp(os.path.getmtime(path)).isoformat(),
        }
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            meta["participant_id"] = data.get("participant_id")
            meta["timestamp"] = data.get("timestamp")
            meta["n_sets"] = len(data.get("sets", []))
        except (json.JSONDecodeError, OSError) as exc:
            meta["error"] = str(exc)
        sessions.append(meta)

    return jsonify({"ok": True, "count": len(sessions), "sessions": sessions})


if __name__ == "__main__":
    print("=" * 56)
    print("  EyeSynth - servidor de eye-tracking")
    print("  >> Experimento : http://localhost:5000/experiment")
    print("  Estímulo único : http://localhost:5000/")
    print(f"  Sesiones        : {SESSIONS_DIR}")
    print(f"  Experimentos    : {EXPERIMENT_SESSIONS_DIR}")
    print("=" * 56)

    # Verifica el entorno del modelo ANTES de servir (que exista el peso y que
    # torch importe). Por defecto NO mantiene el modelo en memoria: se carga al
    # pedir una comparación y se libera, para evitar picos de RAM (OOM en WSL con
    # poca memoria). Exporta EYESYNTH_KEEP_MODELS=1 si tienes RAM de sobra y
    # quieres más velocidad.
    print("[EyeSynth] Verificando modelo de detección...")
    BUNDLE.load_all()

    # Abre el navegador en el experimento tras un breve retraso, sin lanzar
    # procesos externos (evita ventanas de consola y alertas de antivirus).
    # Se usa localhost para que getUserMedia (cámara) tenga contexto seguro.
    # En WSL no hay navegador gráfico: webbrowser recurriría a gio/xdg-open y
    # fallaría con "Operation not supported", así que ahí no se intenta abrir.
    def _open_browser():
        try:
            with open("/proc/version") as fh:
                is_wsl = "microsoft" in fh.read().lower()
        except OSError:
            is_wsl = False
        if is_wsl:
            print("[EyeSynth] WSL detectado: abre manualmente "
                  "http://localhost:5000/experiment en tu navegador.")
            return
        try:
            webbrowser.open("http://localhost:5000/experiment")
        except Exception:
            pass

    threading.Timer(1.5, _open_browser).start()

    app.run(host="0.0.0.0", port=5000, debug=False)
