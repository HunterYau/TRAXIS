# Q2 spectral study transfer

The current launcher defaults to the restored precomputed/graph schedule at
`outputs/simulation-runs/spectral-q2-precomputed-001`. See
`docs/spectral-pzt-precomputed.md` for its checks, resource forecasts and commands.
The original archive described below contains the earlier sparse version and
must be refreshed before using the new method on a pod. Both implementations
remain available.

The companion `spectral-q2-transfer-001.tar.gz` contains the complete source,
locked Linux dependencies, prepared two-mesh study, and imported baseline records.
It excludes Git internals, Windows virtual environments, local caches and unrelated
outputs. Verify `SHA256SUMS` after upload and extraction.

On a Python 3.12 Linux x86_64 RunPod GPU with a CUDA 12.9 compatible driver, upload
the archive and checksum file to persistent pod storage, then run:

```bash
sha256sum -c SHA256SUMS
tar -xzf spectral-q2-transfer-001.tar.gz
cd TRAXIS
bash tools/runpod/spectral_q2_setup.sh
```

Setup installs the locked Python/CUDA packages and runs study preflight only. The
prepared study estimates roughly 31 GiB free GPU memory and 22 GiB host memory.
Choose hardware that has at least those amounts available; the estimates are not
measured peak guarantees. The current local RTX 3060 Ti does not meet the GPU
memory requirement.

Only after reviewing the preflight, start the study explicitly in a persistent
terminal session:

```bash
mkdir -p outputs/simulation-runs/spectral-q2-convergence-001
nohup bash tools/runpod/spectral_q2_run.sh > outputs/simulation-runs/spectral-q2-convergence-001/runpod-console.log 2>&1 &
echo $!
tail -f outputs/simulation-runs/spectral-q2-convergence-001/runpod-console.log
```

This executes 6-ring baseline then 8-ring Q2 spectral PZT mesh. No baseline rerun.
The original LTS recurrence, 5.340576171875 ns fine / 10.68115234375 ns coarse
pair, six frequencies, four voltage channels, checkpointing and plotting protocol
are retained. Stability checks and half-step temporal comparison are disabled in
the requested study profile. The script never stops or pauses a pod.
