# Runpod transfer preparation

This directory prepares a copy of TRAXIS without changing simulator code, physical
settings, meshes, time steps, sampling, precision, or comparison methods. It does
not create a pod, upload files, or automatically start a simulation.

On `research/broadband-comparison`, the bundle also contains the new, separate
direct/broadband workflow and its two prepared schedules. See
[the study guide](../../docs/broadband-comparison.md) before choosing which one to
run. The original direct drivers remain available. The earlier `transfer-001`
bundle remains unchanged; use the new broadband bundle for the new workflow.

## Local snapshot

From the project root on this Windows computer:

```powershell
& ./.venv/portable-runtime/python.exe -m tools.runpod.prepare_transfer --destination outputs/runpod-transfer/transfer-001
```

Choose a new destination for each snapshot. The archive includes the working code
(including uncommitted files in the allowed project directories), configuration,
tests, documentation, measured plate-1 recordings, preflight receipts, and compact
recent result references. Its manifest records file checksums, Git state, included
paths, and exclusions. No GitHub push is needed to use this snapshot.

It excludes credentials, SSH keys, the Windows environment, caches, Git internals,
unrelated old outputs, and large final displacement/velocity arrays. Those arrays
remain on the original computer. This is a run-ready input snapshot, not a complete
backup of every historical output. GitHub separately backs up code and documentation;
ignored data/results remain outside GitHub.

## Future pod setup (requires approval before provisioning)

Use a Linux x86_64 GPU image with Python 3.12, an NVIDIA driver compatible with CUDA
12.9, and SSH enabled. The Linux lock keeps the installed Windows package versions
but uses official Linux wheel hashes; do not use the Windows wheel lock on Linux.
Runpod's currently displayed default image may use a different CUDA/Python version:
check the selected image rather than assuming it matches this environment.

Upload the archive and SHA256SUMS using SCP over the pod's exposed SSH TCP port, or
the official Runpod file-transfer tool. Obtain the actual host, port, username, and
host-key information from the pod's connection details. Preserve host-key checking.
Keep the private SSH key on this Windows computer.

Extract into a new empty directory on persistent storage selected at deployment:

```bash
sha256sum -c SHA256SUMS
tar -xzf traxis-transfer.tar.gz
cd TRAXIS
bash tools/runpod/setup_linux.sh
```

The setup installs a separate `.venv-runpod` environment and checks all transferred
files, the scientific input fingerprints, all five measured excitation records,
and a tiny FP64 CUDA kernel. It never starts a plate simulation. The target Linux
GPU check can only be completed once an approved pod exists.

## Choosing what runs next

The production five-frequency queue is `examples.resolved_pzt_queue`. The mesh/time/
PZT audit is `examples.pzt_convergence`. The corrected continuous-source controlled
trial is `examples.pzt_analytic_trial`. These are different experiments.

The old production and convergence drivers still use the previous linearly
interpolated source. The controlled trial and the new direct/broadband A/B study
use the verified analytic source. This preparation keeps the original drivers
intact and does not silently promote the broadband method to production.

Historical plans and ownership receipts are copied as references only: they retain
Windows paths, old process identifiers, and original checksum provenance. Do not
resume them directly on Linux or edit them to claim a matching historical run.
After selecting the intended study, prepare a fresh run folder and verify its
source and settings before launching. Existing schedulers run their cases serially
and save results after each case; interrupted cases restart rather than resuming
from a numerical checkpoint.

Launch the approved scheduler inside a detached terminal session such as `tmux`
and tee its output into that run's console log. This keeps it running if the SSH
connection closes or the Windows computer sleeps. Reconnect to inspect progress
and copy results back before deleting storage. No automatic compute launch is
included in the setup script.

## SSH key created for this computer

Private key: `C:\Users\bobty\.ssh\traxis_runpod_ed25519`

Public key: `C:\Users\bobty\.ssh\traxis_runpod_ed25519.pub`

Fingerprint: `SHA256:lFkqK+15XnMXOEAuqPB2RyZ2F9/pXFSB9DWf6vlnZkY`

Only the public key belongs in Runpod's Credentials > SSH Public Keys. Neither key
is included in the transfer archive or GitHub. The public key was registered as
"Hunter's PC" and its displayed Runpod fingerprint was verified against the local
key on 2026-11-22. An actual pod SSH connection cannot be verified until a pod exists.
No pod has been created by this preparation.

For a future connection, use `ssh -i "$env:USERPROFILE\.ssh\traxis_runpod_ed25519"`
with the actual SSH host/port arguments shown by Runpod. Do not invent a host or
disable host-key validation.

References: [CuPy installation](https://docs.cupy.dev/en/stable/install.html),
[Runpod SSH and SCP](https://www.runpod.io/blog/how-to-connect-to-your-pod-with-ssh-and-move-files-with-scp-and-filezilla).
