# Delay-aware and mechanical MOR comparison

Implemented methods are candidates for validation, not accepted full-plate fixes.
The existing `examples.pzt_mor_study` ERA workflow and direct/broadband solver
remain available. This separate scheduler writes to a fresh run directory.

## Methods

`examples/pzt_era_delay.py` separates a common integer delay, following the
principle in [Pelling and Sarradj](https://arxiv.org/html/2506.08870v1).
It selects the largest early prefix whose L2 norm is at most 1e-8 of each
receiver, receiver sum, and receiver difference. That prefix is retained
exactly as a finite impulse branch. ERA reduces the remaining response on its
original timestep; the prefix and delay are restored before comparison. No
precursor is discarded and neither receiver receives a fitted time shift.
No unknown future samples are padded at the end of the identification record.
Delay separation does not itself guarantee that ERA will be stable.

`examples/pzt_mechanical_mor.py` builds a full physical-vector Krylov basis of
`M^-1 K` starting at `M^-1 f`, with two-pass mass orthogonalization. It projects
the actual system by `Mr=V.T M V`, `Kr=V.T K V`, `fr=V.T f`, and `Cr=C V`.
This is one-sided mechanical congruence, using the principle described in the
[mechanical MOR survey](https://arxiv.org/html/2010.06331v2).
The implemented basis is polynomial Krylov; it does not implement the optimal
rational shift selection in [Beattie and Gugercin's paper](https://skoge.folk.ntnu.no/prost/proceedings/cdc-ecc05/pdffiles/papers/2916.pdf).

The full plate's GLL mass and merged consistent 3-D PZT mass are applied exactly
through the existing operators. Open-circuit PZT backreaction and the original
voltage probes remain in the projected model. Dense full mass/stiffness matrices
are never assembled. The adapter deliberately rejects finite electrical loads
and damping configurations rather than omitting their additional dynamics.

Positive projected mass/stiffness, symmetry, orthogonality, and the discrete
Verlet stability condition are checked. No negative eigenvalues are clipped,
no damping is added, and no pole is moved. A projection containing zero rigid
eigenvalues is rejected rather than labelled bounded without justification.
Modal recurrence evaluates the same discrete Verlet response and original
5.340576171875 ns timestep, not a new continuous-time integration rule.
Stability does not guarantee waveform accuracy; order must pass the comparisons.

## Scheduled work

Default run: `outputs/simulation-runs/runpod-mor-remedies-v2`.

The 14 serial stages are:

1. Reconstruct the full-order 300 kHz reference from the saved broadband kernel.
2. Build delay-aware ERA once, then check 32, 64, 128, 256, 512, 1024, 2048 states.
3. Build the mechanical projection once, then check 16, 32, 64, 128 displacement
   coordinates (32, 64, 128, 256 first-order states).

Every candidate reconstructs 50, 100, 150, 200, 250, and 300 kHz. Five references
are independent direct solves saved from the prior H200 benchmark. The sixth is
synthetic, not a measured plate-1 recording. All four raw receiver/sum/difference
L2 errors must be below 0.1%; timing, amplitude, and polarity are not fitted.
Unstable delay-ERA candidates remain rejected even when diagnostic waveforms
are measured. All six frequencies must pass for a candidate to be eligible.

The original H100 ERA summaries are copied as historical controls with checksums.
No original result or scientific input is overwritten. There is no new
full-plate time integration in this schedule; the mechanical build does apply
the full physical mass and stiffness operators to basis vectors.

Progress is printed as text: stage, phase, elapsed time, basis/propagation steps,
percentage where measurable, and per-frequency results. A 15-second heartbeat
reports work during library operations without inventing a percentage.
Per-stage logs, models, waveforms, checksummed receipts, and comparison JSON are
saved. `all-results.txt` collects comparisons after each stage and at completion.
Verified completed stages are skipped on resume; incomplete attempts are kept
under `previous-attempts`. Numerical failures require review before retrying.

## Hardware and runtime

Preparation estimates 28,179,234 full DOFs conservatively, including body DOFs.
The maximum mechanical basis requires about **58.0 GiB extra GPU workspace**
plus up to 6 GiB for the existing operator screen. Host workspace is estimated
at 0.7 GiB plus a 4 GiB operator allowance. This configuration is for the H100
NVL (previously 93.6 GiB total) or H200 NVL (previously 141 GiB total).
It does not fit the local RTX 3060 Ti's 8 GiB GPU.

Delay-ERA needs about 6.3 GiB estimated host workspace and 0.3 GiB GPU workspace.
Both builds check currently available memory before large allocations and do
not silently shrink orders, meshes, timesteps, or precision. Estimates are not
measured peaks or proof of fit on a particular allocation. Check current pod
GPU/RAM and storage before launch. The H100's prior ERA build took 876 seconds;
delay-ERA has a similar-sized factorization. Budget roughly 15–30 minutes for
that branch as an unverified planning range. The mechanical build's bandwidth
and Krylov work differ; its runtime is unmeasured and could add minutes to much
longer. No reliable total ETA is established before its first cloud build.

## Prepare and package locally

```powershell
.\.venv\portable-runtime\python.exe -m examples.pzt_mor_remedies --prepare
.\.venv\portable-runtime\python.exe -m tools.runpod.prepare_mor_remedies_transfer --destination outputs/runpod-transfer/transfer-006-mor-remedies
```

The archive is a full curated project snapshot, including required measured
inputs, the verified saved broadband kernel/direct references, and the prepared
schedule. It excludes environments, SSH keys, account credentials, caches,
deployment-control files, and unrelated large outputs. Nothing is uploaded or
started by these commands.

## On the existing pod after transfer is authorized

Use a **new empty project directory**, preserve the earlier deployment, and
verify `SHA256SUMS` before extraction. Linux x86_64, Python 3.12, and a driver
supporting CUDA 12.9 are required. The dedicated environment pins CUDA 12.9
components and adds cuBLAS for physical-basis operations.

```bash
sha256sum -c SHA256SUMS
tar -xzf traxis-mor-remedies.tar.gz
cd TRAXIS
bash tools/runpod/setup_mor_remedies_linux.sh
```

Setup verifies every file and frozen scientific setting plus tiny GPU arithmetic;
it does not launch the schedule. After approval to run, launch from a detached
session or persistent supervisor:

```bash
set -o pipefail
.venv-mor-remedies/bin/python -u -m examples.pzt_mor_remedies --run 2>&1 | tee -a outputs/simulation-runs/runpod-mor-remedies-v2/console.log
```

The same command resumes the same run folder after an interruption. Copy the
whole run folder back, then verify stage artifacts with `assess`. Neither setup
nor scheduler stops, pauses, terminates, or releases a pod. No automatic cloud
downloader is launched by this files-only handoff.

## Verification scope

Small CPU checks cover exact prefix restoration, cancelling receivers, full-rank
mechanical agreement with production consistent-mass Verlet, nested-order
stability, true mass/probe adapter actions, invalid matrices/timestep rejection,
memory refusal, and the complete two-method scheduler/receipt pipeline.
GPU and full-plate accuracy checks remain pending on the target hardware.
