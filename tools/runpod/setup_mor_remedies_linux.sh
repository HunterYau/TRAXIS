#!/usr/bin/env bash
# Setup only, for the existing approved Linux pod. No compute schedule launch.
set -euo pipefail
cd "$(dirname "$0")/../.."
PYTHON_BIN="${TRAXIS_PYTHON:-python3.12}"
"$PYTHON_BIN" -c 'import sys,platform; assert sys.version_info[:2]==(3,12); assert sys.platform=="linux" and platform.machine()=="x86_64"'
"$PYTHON_BIN" -m venv .venv-mor-remedies
.venv-mor-remedies/bin/python -m pip install --require-hashes --only-binary=:all: -r tools/runpod/requirements-mor-remedies-linux-lock.txt
.venv-mor-remedies/bin/python -m pip check
.venv-mor-remedies/bin/python tools/runpod/verify_mor_remedies_transfer.py --gpu
printf '\nRemedy environment verified. Full schedule has NOT been started.\n'
