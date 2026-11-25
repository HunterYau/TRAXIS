# Prepared PZT mesh convergence study

Branch: `research/pzt-mesh-convergence`. Original pre-study LTS work is
checkpointed on `research/broadband-local-time-stepping` at `df32ef0`.
The earlier 12-configuration run was interrupted at the user's request.
Its saved outputs and original source package remain intact in run 001.
This revised combined-only schedule is prepared separately in run 002 and has not started. No optimal mesh is selected.

## Exact spatial schedule

| Order | Configuration | Radial | Angular | Thickness |
|---:|---|---:|---:|---:|
| Already completed | Baseline, imported only | 6 | 32 | 3 |
| 1 | combined L1 | 8 | 40 | 4 |
| 2 | combined L2 | 10 | 48 | 5 |
| 3 | combined L3 | 12 | 64 | 6 |

The plan contains exactly these 3 combined spatial configurations. Each has one
primary LTS broadband impulse solve and one mandatory half-time-step solve:
**6 new kernels**, rather than separate solves at each frequency. Each kernel reconstructs
50, 100, 150, 200, 250 and 300 kHz with the existing five-cycle Hann excitation.
There are no baseline, normal-broadband or beyond-Level-3 simulation jobs.

## Baseline and physical compatibility

The imported reference comes from the verified completed result directory
`outputs/runpod-results/lts-six-frequency-20261124T211750130998Z/lts-005` in the
main checkout. A small immutable copy of its LTS kernel, all six paired voltage
recordings, source receipts, plan, status, partition and timing records is kept
in the new study's `baseline` folder. Original results are untouched.

The plate stays 1 m x 1 m x 1 mm, 250 x 250 x 1 GLL cells of order 5,
E=71 GPa, Poisson ratio 0.33, density 2700 kg/m^3. Three nominal NCE51 PZT
bodies keep their positions, 5 mm radius, 0.5 mm thickness, electrodes,
poling, perfect ties, prescribed-voltage actuator and ideal-open receivers.
No adhesive, damping, instrument response, defect, material or plate mesh is
added/changed. The 4 mm fine edge strip and fine-region definitions are reused.
Polygon area changes with angular tessellation are numerical geometry error,
not a deliberate change of nominal PZT radius.

All baseline physical/integration source hashes are checked. Two PZT assembly
modules gained explicit resource-budget parameters because the old 1024-node,
8192-element limits reject most requested refinements. Their old defaults stay
in place. Mathematical AST checks exclude only resource guard statements and
resource-option arguments, and require the assembly formulas to remain equal.
A small attachment built through the old preserved function and the new
function must have byte-identical mass, stiffness, coupling, coordinates and
indices. The original five LTS/integration/operator files remain unchanged
relative to the pre-study checkpoint.

## Stability and time accuracy

On RunPod each mesh is actually assembled, including its consistent coupled
mass blocks and generalized local stiffness/mass eigenvalue bound. The complete
coupled stability limit is recomputed. Starting at the baseline coarse step
10.68115234375 ns, the scheduler halves the step only as required by the
unchanged fine-step safety factor (0.8) and LTS effective-operator screen.
The substep ratio remains 2, the source treatment remains split-LFC/coarse,
and the 350 microsecond duration is constant. Only exact nested recording
nodes are compared; no waveform interpolation, phase fit or gain fit is used.

The same mesh is then solved at half that coarse step. Both primary and
half-step kernels have their own stability checks, receipts and timings.
The mandatory temporal check target is 0.01% for raw, envelope and normalized
wiggle L2 differences over all six frequencies and all four channels, giving
a tenfold margin to the tighter 0.1% spatial threshold. A failed temporal
check is reported as `completed_temporal_unresolved`; it is not silently
accepted or used to choose an optimal mesh. This is an empirical two-grid
accuracy check, not a rigorous temporal-error certificate or observed-order
study. No third time grid is silently scheduled.

The reused baseline has no same-mesh half-step LTS check. Its old normal/LTS
method comparison is not substituted for one. Consequently baseline-relative
changes alone cannot establish 0.1% spatial accuracy. Previous-level and
finest-level comparisons within the new combined
series are also recorded. Even qualified small differences are numerical
agreement, not proof of accuracy against physical experiments.

The unchanged solver supports at most 500,000 coarse updates. If the actual
stable primary step would require a half-step check beyond this limit, the
configuration is recorded as unsuccessful, rather than changing the integrator,
shortening the recording or pretending time convergence passed.

## Resource and runtime forecasts

`schedule.txt` and each case's `estimate` in `plan.json` contain exact topology
and DOF counts, geometry quality checks, provisional step counts, conservative
memory forecasts and broad runtime ranges. These are forecasts, not measured
RunPod allocations or promises. Actual available RAM respects cgroup quota;
currently free VRAM and RAM are checked before building and before dense
attachment/merge stages. Precision or mesh resolution is never reduced to fit.

Global plate DOFs stay 28,170,018. Including the PZT free-body DOFs, totals range
from 28,181,574 to 28,211,544. Combined Level 3 has 5,383 nodes and 26,496
tetrahedra per PZT. Conservative forecasts range from about 20 to 50 GiB GPU
memory and 21 to 87 GiB host RAM. The largest case needs roughly 63 GiB free
VRAM and 109 GiB currently available host RAM with the launch headroom policy.
An H100 NVL with sufficient host allocation is a candidate; its actual free
resources must be checked on launch. The local 8 GB/16 GB machine does not fit.

Runtime ranges use the measured baseline 237-second LTS kernel and 10-second
construction time, scale dense PZT work quadratically per time step and
cubically during assembly, and use estimated step refinement. The unknown
dense-PZT fraction is bracketed from 5% to 50%. Stability-screen and network-IO
overheads are not known. The combined high-level cases can take substantially
longer than independent refinements; no full run was used to calibrate them.
`schedule.txt` is the authoritative per-configuration forecast table.

## Outputs, checkpointing and failure recovery

Everything is under `outputs/simulation-runs/pzt-mesh-convergence-002`:

```
plan.json, schedule.txt, status.json, all-results.json, all-results.txt, metrics.csv
baseline/                        # imported; never simulated
baseline-assembly-sources/       # old source snapshots for compatibility checks
cases/01-combined-L1/              # one folder per spatial configuration
  build.json, time-plan.json, comparisons.json, result.json, worker.log
  primary/trace.npz, result.json, 50khz/waveforms.npz ... 300khz/waveforms.npz
  half-step/trace.npz, result.json, 50khz/waveforms.npz ... 300khz/waveforms.npz
  plots/50khz/vs-baseline/ and temporal/ ... 300khz/
  attempts/                     # previous failed/interrupted receipts and logs
```

Each completed kernel is saved atomically and checksum-verified, so a later
plotting/analysis interruption or half-step failure does not repeat the primary
kernel. An interrupted kernel itself restarts from t=0; this does not add
mid-step state checkpointing to the original integrator. Completed cases are
skipped entirely. Changed/corrupted completed outputs cause an explicit error.
Workers are isolated processes, releasing their CPU/GPU allocations at exit.
Failed numerical cases are retained, reported and skipped on ordinary resume;
`--retry-failed` explicitly retries them while reusing verified kernels.

Metrics use top, bottom, top+bottom (S0-related) and top-bottom (A0-related),
without claiming pure separated modes. JSON records full raw/envelope/wiggle
relative L2 errors, signed and envelope Pearson correlations, RMSE, maximum
voltage difference, and independently located S0/A0 arrival candidates on each
channel. Existing theory-guided arrival code is reused, including ambiguity
flags. Candidate differences are retained even when unreliable. Its arrival
grid is 0.2 microseconds, so sub-grid time differences are not claimed resolved.
Both 1% and 0.1% L2 thresholds are evaluated; reliable arrival percentages have
their own threshold flags. Pearson remains a correlation, not an accuracy
percentage. Normalized envelopes are supplemental; raw envelope L2 retains
amplitude differences. Raw voltage is never scaled to improve agreement.

Waveform and envelope overlays use the existing headless plotting/export
infrastructure, producing PNG and SVG. Display curves are limited to 6000
samples for file size; metrics use every shared recording sample. Failed
configurations retain error details and explicit unavailable entries for all
24 frequency/channel combinations. Runtime, DOFs, stable steps and sampled
RAM/GPU memory peaks are recorded separately from predictions.

## Commands prepared for later RunPod execution

Do not run the following launch commands until execution is explicitly
authorized. No tool here creates, starts, pauses, terminates or stops a pod.

The prepared archive is under `outputs/runpod-transfer/pzt-mesh-convergence-002`.
Copy its `.tar.gz` and `.tar.gz.sha256` files to a NEW pod directory, using the
pod's actual SSH host, port and verified host key:

```bash
# Run on the already running pod; plain mkdir refuses an occupied directory.
mkdir /workspace/traxis-pzt-mesh-convergence-002
cd /workspace/traxis-pzt-mesh-convergence-002
# Upload the archive and checksum file here by scp, then:
sha256sum -c pzt-mesh-convergence-002.tar.gz.sha256
tar -xzf pzt-mesh-convergence-002.tar.gz
cd TRAXIS
bash tools/runpod/pzt_mesh_setup.sh
```

Setup uses a NEW isolated Python 3.12 environment on local `/tmp` disk,
hash-pinned CUDA 12.9 scientific libraries, the existing plotting requirements,
bundle/baseline verification, a tiny native eigenvalue check and lightweight
study tests. It does not start any full simulation.

Launch later (one serial scheduler for all 3 configurations):

```bash
/tmp/traxis-pzt-mesh-runtime-001/bin/python tools/runpod/pzt_mesh_transfer.py launch \
  --root outputs/simulation-runs/pzt-mesh-convergence-002 \
  --python /tmp/traxis-pzt-mesh-runtime-001/bin/python --stability-backend auto
# Progress, without starting any job:
/tmp/traxis-pzt-mesh-runtime-001/bin/python tools/runpod/pzt_mesh_transfer.py status \
  --root outputs/simulation-runs/pzt-mesh-convergence-002
```

After confirmed interruption, use the same launch command to resume. Add
`--retry-failed` only when intentionally retrying failed cases. Active scheduler
and worker ownership checks prevent duplicate paid calculations. Keep the
runtime path direct, rather than invoking a virtualenv through a /workspace
symlink, to avoid the previously observed CUDA-header network IO delay.

Arm the download-only watcher on THIS computer before the cloud launch:

```powershell
python tools/runpod/pzt_mesh_transfer.py watch --detach --host HOST --port PORT --key KEY_PATH --known-hosts KNOWN_HOSTS_PATH --remote-project /workspace/traxis-pzt-mesh-convergence-002/TRAXIS --output "E:\TRAXIS Project Outputs\PZT Mesh Convergence"
```

Use the helper from this branch/worktree with the existing SSH private-key and
verified known-hosts file paths. The computer must stay awake, connected, and
the external drive must remain available. The watcher records progress every
30 seconds. When the scheduler finishes (including a batch with failures), it
exports all results, downloads the archive, verifies its checksum and each
individual file, extracts it into a new timestamp folder, and writes a local
`download-verified.json` receipt. Connection/copy failure retries without
altering the pod. The large optional eigenvector warm hint is excluded from
the export; all scientific results and stability certificates remain included.
The watcher and simulations continue independently after Codex stops working.
**The pod is always left running.**

The nine independent refinement configurations are excluded from the active
schedule. Their old JSON files are retained under
`configs/pzt-mesh-convergence/historical-12-configuration-schedule`.
