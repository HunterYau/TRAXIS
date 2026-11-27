#!/usr/bin/env bash
# Install and preflight the prepared Q2 spectral study; never starts simulations.
set -euo pipefail
cd "$(dirname "$0")/../.."
PYTHON_BIN="${TRAXIS_PYTHON:-python3.12}"
"$PYTHON_BIN" -c 'import sys,platform; assert sys.version_info[:2]==(3,12); assert sys.platform=="linux" and platform.machine()=="x86_64"'
python -m venv .venv-runpod
.venv-runpod/bin/python -m pip install --require-hashes --only-binary=:all: -r tools/runpod/requirements-linux-lock.txt
.venv-runpod/bin/python -m pip install --only-binary=:all: -r requirements-plots.txt
# RunPod's CUDA 13 base image does not include the CUDA 12 libraries that the
# pinned CuPy 12.x wheel dynamically loads. Keep the runtime matched to CuPy.
.venv-runpod/bin/python -m pip install --only-binary=:all: \
  nvidia-cublas-cu12==12.9.3.13 nvidia-cusolver-cu12==11.7.5.82
.venv-runpod/bin/python -m pip check
source tools/runpod/cuda12_env.sh
.venv-runpod/bin/python - <<'PY'
import cupy as cp
from cupyx.scipy.sparse.linalg import SuperLU
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import splu
assert cp.cuda.runtime.runtimeGetVersion() == 12090
answer=SuperLU(splu(csc_matrix([[2.,0.],[0.,4.]]))).solve(cp.asarray([2.,8.]))
cp.testing.assert_allclose(answer,cp.asarray([1.,2.]),rtol=1e-14)
print('CUDA runtime:', cp.cuda.runtime.runtimeGetVersion())
print('GPU:', cp.cuda.runtime.getDeviceProperties(0)['name'].decode())
print('Free GPU GiB:', cp.cuda.runtime.memGetInfo()[0] / 2**30)
PY
.venv-runpod/bin/python -m examples.pzt_mesh_study --preflight --root "${TRAXIS_STUDY_ROOT:-outputs/simulation-runs/spectral-q2-precomputed-001}"
echo 'Environment and study preflight passed. No plate simulations started.'
