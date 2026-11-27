#!/usr/bin/env bash
# Explicit launch only; never changes the pod lifecycle.
set -euo pipefail
cd "$(dirname "$0")/../.."
source tools/runpod/cuda12_env.sh
export CUPY_CACHE_DIR="${CUPY_CACHE_DIR:-/tmp/traxis-local-aluminum-cupy}"
exec .venv-runpod/bin/python tools/runpod/local_aluminum_remote.py launch \
  --python .venv-runpod/bin/python \
  --root "${TRAXIS_STUDY_ROOT:-outputs/simulation-runs/spectral-q2-local-aluminum-001}"
