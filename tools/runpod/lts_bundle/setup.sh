#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../.."
TRAXIS_CPU_THREADS="$(python3 tools/runpod/lts_bundle/cpu_threads.py)"
export OPENBLAS_NUM_THREADS="$TRAXIS_CPU_THREADS" OMP_NUM_THREADS="$TRAXIS_CPU_THREADS" MKL_NUM_THREADS="$TRAXIS_CPU_THREADS"
echo "CPU maths threads: $TRAXIS_CPU_THREADS (container allocation)"
export CUPY_CACHE_DIR="$PWD/.cupy-cache"
python3 -c 'import sys,platform; assert sys.version_info[:2]==(3,12) and platform.system()=="Linux" and platform.machine()=="x86_64"'
if [[ ! -d .venv-lts ]]; then python3 -m venv .venv-lts; fi
.venv-lts/bin/python -m pip install --require-hashes --only-binary=:all: -r tools/runpod/requirements-linux-lock.txt
.venv-lts/bin/python -m pip check
.venv-lts/bin/python tools/runpod/lts_bundle/verify.py --gpu
