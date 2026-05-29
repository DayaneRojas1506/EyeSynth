@echo off
REM EyeSynth - instala dependencias y arranca el servidor (Windows)
setlocal

cd /d "%~dp0"

REM crea un entorno virtual la primera vez
if not exist ".venv" (
  echo [EyeSynth] Creando entorno virtual .venv ...
  python -m venv .venv
)

REM activa el entorno
call .venv\Scripts\activate.bat

echo [EyeSynth] Instalando dependencias ...
python -m pip install --quiet --upgrade pip
python -m pip install --quiet -r requirements.txt

echo [EyeSynth] Arrancando servidor en http://localhost:5000
python server\server.py

endlocal
