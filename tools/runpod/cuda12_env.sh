# Make pip-provided CUDA 12 libraries visible beside a CUDA 13 RunPod driver.
set -euo pipefail
_traxis_nvidia_libs="$(find .venv-runpod/lib/python3.12/site-packages/nvidia -maxdepth 4 -type d -name lib -print | paste -sd: -)"
export LD_LIBRARY_PATH="${_traxis_nvidia_libs}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
unset _traxis_nvidia_libs
