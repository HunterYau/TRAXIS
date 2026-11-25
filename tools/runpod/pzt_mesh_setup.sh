#!/usr/bin/env bash
# Install/check only. This script never launches a plate simulation or a pod.
set -euo pipefail
cd "$(dirname "$0")/../.."
python3 -c 'import sys,platform; assert sys.version_info[:2]==(3,12) and platform.machine()=="x86_64"'
runtime="${TRAXIS_MESH_RUNTIME:-/tmp/traxis-pzt-mesh-runtime-001}"
if [[ ! -d "$runtime" ]]; then python3 -m venv "$runtime"; fi
threads="$(python3 tools/runpod/lts_bundle/cpu_threads.py)"
export OPENBLAS_NUM_THREADS="$threads" OMP_NUM_THREADS="$threads" MKL_NUM_THREADS="$threads"
export CUPY_CACHE_DIR=/tmp/traxis-pzt-mesh-cupy
"$runtime/bin/python" -m pip install --require-hashes --only-binary=:all: --no-deps \
  -r tools/runpod/requirements-linux-lock.txt -r tools/runpod/requirements-lts-eigensolver-linux-lock.txt
"$runtime/bin/python" -m pip install --only-binary=:all: -r requirements-plots.txt
"$runtime/bin/python" -m pip check
"$runtime/bin/python" tools/runpod/pzt_mesh_transfer.py check-bundle
"$runtime/bin/python" - <<'PY'
import cupy as cp
import numpy as np
from cupyx.scipy.sparse.linalg import eigsh,LinearOperator
assert cp.cuda.runtime.runtimeGetVersion()==12090
assert cp.cuda.runtime.driverGetVersion()>=12090
op=LinearOperator((8,8),matvec=lambda x:cp.arange(1,9,dtype=cp.float64)*x,dtype=np.float64)
w,v=eigsh(op,k=1,which='LA',ncv=6,v0=cp.ones(8,dtype=cp.float64),tol=1e-8,maxiter=1000)
assert abs(float(w[0])-8.)<1e-8
assert float(cp.sqrt(cp.sum((op@v[:,0]-w[0]*v[:,0])**2)))<1e-8
print('Native CUDA FP64 eigensolver check passed; no plate simulation started.')
PY
"$runtime/bin/python" -m unittest tests.test_pzt_mesh_study
echo "Setup finished. Runtime: $runtime/bin/python"
echo 'No full simulations were launched. Launch the prepared schedule separately when authorized.'
