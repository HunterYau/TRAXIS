#!/usr/bin/env bash
# Install into this isolated checkout; preflight only. No full simulation.
set -euo pipefail
cd "$(dirname "$0")/../.."
PYTHON_BIN="${TRAXIS_PYTHON:-python3.12}"
"$PYTHON_BIN" -c 'import sys,platform; assert sys.version_info[:2]==(3,12); assert sys.platform=="linux" and platform.machine()=="x86_64"'
"$PYTHON_BIN" -m venv .venv-runpod
.venv-runpod/bin/python -m pip install --require-hashes --only-binary=:all: -r tools/runpod/requirements-linux-lock.txt -r tools/runpod/requirements-lts-eigensolver-linux-lock.txt
.venv-runpod/bin/python -m pip install --only-binary=:all: -r requirements-plots.txt
.venv-runpod/bin/python -m pip check
source tools/runpod/cuda12_env.sh
export CUPY_CACHE_DIR="${CUPY_CACHE_DIR:-/tmp/traxis-local-aluminum-cupy}"
.venv-runpod/bin/python tools/runpod/local_aluminum_preflight.py --gpu --root "${TRAXIS_STUDY_ROOT:-outputs/simulation-runs/spectral-q2-local-aluminum-001}"
echo 'Setup and tiny component checks complete. Full simulations have not started.'
