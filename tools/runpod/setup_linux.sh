#!/usr/bin/env bash
# Run explicitly on the future Linux pod, after extracting the transfer bundle.
# Installs the environment and checks inputs. Does not launch simulations.
set -euo pipefail
cd "$(dirname "$0")/../.."
PYTHON_BIN="${TRAXIS_PYTHON:-python3.12}"
"$PYTHON_BIN" -c 'import sys, platform; assert sys.version_info[:2] == (3,12), "Python 3.12 required"; assert sys.platform == "linux" and platform.machine() == "x86_64", "Linux x86_64 required"'
"$PYTHON_BIN" -m venv .venv-runpod
.venv-runpod/bin/python -m pip install --require-hashes --only-binary=:all: -r tools/runpod/requirements-linux-lock.txt
.venv-runpod/bin/python -m pip check
.venv-runpod/bin/python tools/runpod/verify_transfer.py --gpu
printf '\nEnvironment checked. No simulation launched.\n'
