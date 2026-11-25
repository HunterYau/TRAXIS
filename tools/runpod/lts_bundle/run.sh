#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../.."
TRAXIS_CPU_THREADS="$(python3 tools/runpod/lts_bundle/cpu_threads.py)"
export OPENBLAS_NUM_THREADS="$TRAXIS_CPU_THREADS" OMP_NUM_THREADS="$TRAXIS_CPU_THREADS" MKL_NUM_THREADS="$TRAXIS_CPU_THREADS"
echo "CPU maths threads: $TRAXIS_CPU_THREADS (container allocation)"
export CUPY_CACHE_DIR="$PWD/.cupy-cache"
.venv-lts/bin/python tools/runpod/lts_bundle/verify.py
.venv-lts/bin/python -c 'import json,hashlib; from pathlib import Path; p=Path("ready.json"); r=json.loads(p.read_text()); assert r["tests"]==31 and r["manifest_sha256"]==hashlib.sha256(Path("bundle-manifest.json").read_bytes()).hexdigest(), "Run setup.sh successfully before launching"'
exec .venv-lts/bin/python -u -m examples.pzt_local_time_study --run --root outputs/simulation-runs/lts-005
