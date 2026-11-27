# Refined aluminum / Q2 PZT transfer

The dedicated schedule runs six rings at fine/coarse 4.9/9.8 ns, then eight rings
at 4.7/9.4 ns. Each broadband kernel reconstructs 50, 100, 150, 200, 250 and
300 kHz. Both cases use the new local aluminum mesh, unchanged PZT formulations
and original LTS recurrence. Total and simulation stopwatches are printed and
saved, with failure/resume checkpoints. No stability certification is implied.

The archive is code, configuration and a prepared plan. No old waveform files are
needed for these two runs. Old implementations remain included as references.
The 21.3 GiB peak GPU estimate is unmeasured; provision at least 32 GiB free GPU
memory with headroom and sufficient host RAM (prefer at least 8 GiB free).
Use Linux x86_64, Python 3.12, a driver compatible with CUDA 12.9 and a persistent
workspace. CUDA 13 drivers can host the pinned CUDA 12.9 user-space libraries.
The setup installs CuPy 14.2.0 and the matched CUDA 12 libraries in an isolated
environment. The same-loaded-library native SpSM check is exercised before use.

## Upload and extract

Use the current pod's direct TCP SSH endpoint, not an old address. Upload the
archive and SHA256SUMS via SCP to a **new** directory such as
`/workspace/traxis-local-aluminum-001`. Do not extract over an existing checkout.
Commands below are preparation/launch instructions; packaging executes none.

```bash
cd /workspace/traxis-local-aluminum-001
sha256sum -c SHA256SUMS
tar -xzf traxis-local-aluminum.tar.gz
cd TRAXIS
python3 - <<'PY'
import hashlib,json
from pathlib import Path
manifest=json.loads(Path('local-aluminum-manifest.json').read_text())
for name,want in manifest['files'].items():
    assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==want,name
print('All package hashes passed')
PY
bash tools/runpod/local_aluminum_setup.sh
```

Setup performs environment, capacity and tiny native GPU/graph checks only. It
does not run either full plate case. It does not call the older study preflight.

## Launch and watch (only when instructed)

```bash
bash tools/runpod/local_aluminum_run.sh
tail -f outputs/simulation-runs/spectral-q2-local-aluminum-001/console.log
```

The launch command detaches the scheduler, records its process identity, and
refuses a duplicate process. Tracebacks go to `scheduler.log`; progress goes to
`console.log`. Rerun the same launch command to resume after fixing a failure.
Completed case/kernel receipts are retained. The same code/configuration
snapshot must be used for resume.

## Browser progress console on this PC

In PowerShell, from this checkout, set these to the current direct TCP endpoint:

```powershell
$env:TRAXIS_SSH_REMOTE = 'root@CURRENT_IP'
$env:TRAXIS_SSH_PORT = 'CURRENT_PORT'
$env:TRAXIS_SSH_KEY = "$env:USERPROFILE\.ssh\traxis_runpod_ed25519"
$env:TRAXIS_KNOWN_HOSTS = 'PATH_TO_VERIFIED_KNOWN_HOSTS'
$env:TRAXIS_REMOTE_PROJECT = '/workspace/traxis-local-aluminum-001/TRAXIS'
python tools/runpod/spectral_q2_web_console.py
```

Open `http://127.0.0.1:8765`. It shows total and simulation stopwatches, current
case, steps, percent, GPU status and saved debug output. SSH host verification is
required; confirm the current endpoint's host key before using a known-host file.

## Automatic result download to this PC

Keep this PC awake with the download watcher running. Replace endpoint/key paths:

```powershell
python tools/runpod/local_aluminum_remote.py watch --host CURRENT_IP --port CURRENT_PORT --key PATH_TO_KEY --known-hosts PATH_TO_VERIFIED_KNOWN_HOSTS --remote-project /workspace/traxis-local-aluminum-001/TRAXIS --output "E:\TRAXIS Project Outputs\Local Aluminum Q2" --detach
```

The watcher downloads on completion or a reported failure, verifies the archive
and individual result hashes, and extracts into a new timestamped directory.
It retries temporary connection failures and never overwrites a previous download.
It does not stop or pause the pod. A killed scheduler whose status remains running
must be inspected before exporting; a missing process alone is not completion.

No command in these helpers creates, starts, resumes, stops or terminates a pod.
