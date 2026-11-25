# LTS six-frequency transfer

This is a separate snapshot for one intact plate. Two full impulse solves:
normal broadband, followed by broadband with local time stepping. Each method
reconstructs 50, 100, 150, 200, 250 and 300 kHz. Normal broadband supplies the
numerical reference. No experimental recordings or earlier outputs are needed.

Time steps are 5.340576171875 ns for the baseline and the LTS fine regions
(PZTs and 4 mm edge strip); LTS uses 10.68115234375 ns elsewhere. Full-plate
stability is screened before propagation. The numerical screen is not a
formal proof. Full-plate accuracy/speedup have not been measured.

## Preparation completed locally

`build.py` creates `outputs/runpod-transfer/lts-six-frequency-003/`, containing
the archive, its SHA256 and a per-file manifest. It includes Python source,
small verification tests, the two necessary configuration files, pinned Linux
requirements, the prepared lts-005 plan and these scripts. It excludes virtual
environments, credentials, Git metadata, measured waveforms and old results.

No script creates, starts or resumes a pod. Upload and launch are separate
explicit commands. The optional PC-side watch.py stops only its specified pod
after a completed run has been downloaded and verified locally, as authorized
by the user. No credentials are embedded.

## Upload later

Use the pod's current SSH host/port, the existing local private-key path and
a known_hosts file whose host key has been verified. The helper enforces
strict host-key checking. It creates a NEW directory; if that name already
exists, it refuses and you must choose another name. Existing pod files and
environments are preserved.

From this worktree, using a working local Python:

```powershell
python tools/runpod/lts_bundle/transfer.py upload --host HOST --port PORT --key KEY_PATH --known-hosts KNOWN_HOSTS_PATH
```

Default new remote directory: `/workspace/traxis-lts-six-frequency-003`.
Use `--remote-root /workspace/traxis-lts-ANOTHER_NAME` if occupied. The archive
is checksummed remotely before extraction. Upload does not run setup or sim.

## Setup and launch later, after authorization

Inside the NEW remote project folder:

```bash
cd /workspace/traxis-lts-six-frequency-003/TRAXIS
bash tools/runpod/lts_bundle/setup.sh
# Separate explicit action: starts the full benchmark detached from SSH.
.venv-lts/bin/python tools/runpod/lts_bundle/start.py
tail -f outputs/simulation-runs/lts-005/console.log
```

Setup requires Linux x86_64, Python 3.12 and a CUDA 12.9-compatible NVIDIA
driver. It creates its own `.venv-lts` and installs hash-pinned NumPy/SciPy,
psutil, CuPy and CUDA runtime/NVRTC packages. It neither changes the driver
nor modifies an existing environment. It verifies all transferred file hashes,
plan fingerprints, runtime/compiler versions, free GPU/host memory and 31
tiny CPU/CUDA checks (zero skips required). It records hardware/readiness
results. The full plate is NOT run during setup. The full-plate workspace
requires more than the local 8 GB GPU; the deployment readiness screen requires
24 GiB currently free VRAM and 12 GiB available host RAM. Solver checks use
actual allocation and free memory again before the full calculation.

The launcher keeps running after SSH disconnects. Its PID/log paths are in
`outputs/simulation-runs/lts-005/launcher.json`; progress is in `status.json`.
A second launch is refused while a launcher receipt exists. Resume uses the
same run directory, preserves completed hash-verified kernels and reconstructs
missing output. To resume after a verified dead process, inspect the queue lock
and archive the old launcher receipt first. The remote launcher alone does not
stop the pod; use the PC-side watcher below for automatic backup and stopping.

## Download later

After completion or confirmed interruption:

```powershell
python tools/runpod/lts_bundle/transfer.py download --host HOST --port PORT --key KEY_PATH --known-hosts KNOWN_HOSTS_PATH
```

Use the same `--remote-root` if customized. Results download into a NEW local
`outputs/runpod-results/lts-six-frequency-TIMESTAMP` folder. The archive and
each result are SHA256 checked. Pod files are retained and the pod is left
running. Outputs include per-frequency voltage waveforms, error/similarity
comparisons, `summary.txt`, `performance.json`, `all-results.txt`, console logs,
source/plan fingerprints and stability/readiness reports. A partial run remains
marked incomplete; exported data does not imply validation or completion.

No volume persistence is assumed. Keep the downloaded backup before deleting
or replacing a pod/container. This package does not provision storage.

## Automatic download, then stop (run on your computer)

Use `watch.py` for future launches rather than launching `start.py` alone. It
starts a detached LOCAL watcher, launches only the simulation on an already
running pod, waits for completion and scheduler exit, downloads the entire run,
checks the archive and every saved file, checks both completed solve receipts
and all scheduled frequency outputs, then requests a pod STOP and reads back
its stopped state. It never deletes or terminates a pod.

```powershell
python tools/runpod/lts_bundle/watch.py --detach --launch --pod-id POD_ID --host HOST --port PORT --key KEY_PATH --known-hosts KNOWN_HOSTS_PATH --cli RUNPODCTL_PATH --receipt outputs/runpod-deployments/RUN_NAME/backup-state.json --output-root outputs/runpod-results
```

Use an authenticated runpodctl configuration or RUNPOD_API_KEY in the local
process environment. On this computer, the existing Windows user-encrypted
credential can instead be supplied by its PATH using `--dpapi-key PATH`.
Credentials stay on the PC and are never uploaded. Use a fresh receipt for a
new simulation run. To attach to an already running simulation omit `--launch`.
The watcher must point to the current SSH endpoint and verified host key.

Leave your computer awake and connected until the backup completes. The local
log beside the receipt shows progress, the printed final summary and the stop
confirmation. If a download, integrity check or connection fails, it keeps the
pod running and retries every 30 seconds. A crashed or incomplete simulation is
not automatically stopped. To resume a watcher after a PC reboot, rerun with
the same receipt and omit `--launch`; a previously verified backup is rechecked
before any stop request. Automatic backup depends on this local process running;
a remote launch on its own does not arm it.

Console progress includes elapsed seconds and an HH:MM:SS stopwatch for each
plate solve. Summary output records both solve times, both total method times
including reconstruction/output, and construction/stability check overhead.
All six frequencies share one broadband plate solve per method; they are not
reported as six separately timed simulations.
