#!/usr/bin/env bash
# Environment setup only. Never starts a simulation or a pod.
set -euo pipefail
cd "$(dirname "$0")/../.."
PYTHON_BIN="${TRAXIS_PYTHON:-python3.12}"
"$PYTHON_BIN" -c 'import sys, platform; assert sys.version_info[:2] == (3,12); assert sys.platform == "linux" and platform.machine() == "x86_64"'
"$PYTHON_BIN" -m venv .venv-mor-runpod
.venv-mor-runpod/bin/python -m pip install --require-hashes --only-binary=:all: -r tools/runpod/requirements-mor-linux-lock.txt
.venv-mor-runpod/bin/python -m pip check
.venv-mor-runpod/bin/python tools/runpod/verify_transfer.py --gpu --dense-svd
printf '\nMOR environment checked. No plate simulation or full-size MOR build launched.\n'
