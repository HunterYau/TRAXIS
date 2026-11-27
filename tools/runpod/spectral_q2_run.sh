#!/usr/bin/env bash
# Explicit simulation launcher; only run after reviewing preflight and GPU capacity.
set -euo pipefail
cd "$(dirname "$0")/../.."
source tools/runpod/cuda12_env.sh
study_root="${TRAXIS_STUDY_ROOT:-outputs/simulation-runs/spectral-q2-precomputed-001}"
exec .venv-runpod/bin/python -u -m examples.pzt_mesh_study --run \
  --root "$study_root" "$@"
