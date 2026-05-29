#!/usr/bin/env bash
# EyeSynth - instala dependencias y arranca el servidor (Mac / Linux)
set -e

cd "$(dirname "$0")"

# crea un entorno virtual la primera vez
if [ ! -d ".venv" ]; then
  echo "[EyeSynth] Creando entorno virtual .venv ..."
  python3 -m venv .venv
fi

# activa el entorno
# shellcheck disable=SC1091
source .venv/bin/activate

echo "[EyeSynth] Instalando dependencias ..."
pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt

echo "[EyeSynth] Arrancando servidor en http://localhost:5000"
python server/server.py
