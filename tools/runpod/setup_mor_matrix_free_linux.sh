#!/usr/bin/env bash
# Future approved pod only. Environment and tiny checks; no full study launch.
set -euo pipefail
cd "$(dirname "$0")/../.."
PYTHON_BIN="${TRAXIS_PYTHON:-python3.12}"
"$PYTHON_BIN" -c 'import sys, platform; assert sys.version_info[:2] == (3,12); assert sys.platform == "linux" and platform.machine() == "x86_64"'
"$PYTHON_BIN" -m venv .venv-mor-matrix-free
.venv-mor-matrix-free/bin/python -m pip install --require-hashes --only-binary=:all: -r tools/runpod/requirements-mor-matrix-free-linux-lock.txt
.venv-mor-matrix-free/bin/python -m pip check
.venv-mor-matrix-free/bin/python tools/runpod/verify_transfer.py --gpu
printf '\nMatrix-free MOR environment checked. No full-size build or simulation launched.\n'
