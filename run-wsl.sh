#!/usr/bin/env bash
# EyeSynth - arranca el servidor EN WSL con inferencia del modelo (torch).
#
# A diferencia de run.sh, este server carga 4 detectores de fakes (SIMPLE +
# NPR + Swin-B + UnivFD) y expone /compare-models para la comparación
# humano-vs-IA en la página "Tus respuestas". Requiere un entorno con torch,
# torchvision y el paquete `clip` (para UnivFD).
#
# Uso (desde WSL):
#   cd /mnt/c/Users/mitsu/Desktop/EyeSynth
#   bash run-wsl.sh
set -e

cd "$(dirname "$0")"

# Checkpoints de los modelos (override exportando NPR_CKPT / SIMPLE_CKPT /
# SWIN_CKPT / UNIVFD_CKPT).
export NPR_CKPT="${NPR_CKPT:-$(pwd)/server/npr_fakereal.pth}"
export SIMPLE_CKPT="${SIMPLE_CKPT:-$(pwd)/server/simple_fakereal.json}"
export SWIN_CKPT="${SWIN_CKPT:-$(pwd)/server/swin_fakereal.pth}"
export UNIVFD_CKPT="${UNIVFD_CKPT:-$(pwd)/server/univfd_fakereal.pth}"

# Con RAM suficiente (>= ~4GB para WSL) mantén el modelo en memoria: las
# comparaciones son más rápidas. En equipos con poca RAM, exporta
# EYESYNTH_KEEP_MODELS=0 antes de correr para cargarlo por petición (evita OOM).
export EYESYNTH_KEEP_MODELS="${EYESYNTH_KEEP_MODELS:-1}"

# Entorno virtual SEPARADO del .venv de Windows (binarios incompatibles).
VENV=".venv-wsl"
if [ ! -d "$VENV" ]; then
  echo "[EyeSynth] Creando entorno virtual $VENV ..."
  python3 -m venv "$VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"

# La instalación solo corre la PRIMERA vez (o si pasas --reinstall). Tras
# instalar con éxito se deja un archivo centinela; las siguientes ejecuciones
# arrancan directo sin re-verificar dependencias.
SENTINEL="$VENV/.deps-ok"
if [ "$1" = "--reinstall" ]; then rm -f "$SENTINEL"; fi
if [ ! -f "$SENTINEL" ]; then
  echo "[EyeSynth] Instalando dependencias (torch puede tardar la 1ª vez)..."
  pip install --quiet --upgrade pip
  # torch/torchvision: si tu máquina tiene CUDA, usa la rueda adecuada, p.ej.:
  #   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
  pip install --quiet -r requirements.txt
  touch "$SENTINEL"
  echo "[EyeSynth] Dependencias listas."
else
  echo "[EyeSynth] Dependencias ya instaladas (usa '--reinstall' para forzar)."
fi

echo "[EyeSynth] Checkpoints: $SIMPLE_CKPT | $NPR_CKPT | $SWIN_CKPT | $UNIVFD_CKPT"
echo "[EyeSynth] Arrancando servidor ..."
echo "[EyeSynth]   >> Experimento : http://localhost:5000/experiment"
echo "[EyeSynth]   Estímulo único : http://localhost:5000/"
python server/server.py
