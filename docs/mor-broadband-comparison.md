# MOR and broadband comparison: prepared, not run

Branch: `research/mor-broadband-comparison`. The original direct and broadband
methods remain unchanged. This work adds `examples.pzt_era` and an explicitly
launched scheduler, `examples.pzt_mor_study`.

## Schedule

1. Reconstruct the **full-order broadband 300 kHz** reference from the completed
   H200 impulse response. It is the original full-order discrete model, without
   MOR. No additional full-plate solve is needed.
2. Build one ERA reduced model from that saved impulse response.
3. Check reduced orders **32, 64, 128, 256, 512, 1024, 2048**. Each order produces
   **50, 100, 150, 200, 250, 300 kHz**, in order, from one reduced impulse response.
   These are seven cheap-model size checks, not seven new full-plate solves.
4. Save and print differences after every frequency and update the combined
   `findings.json` and `findings.txt` after every stage.

Everything keeps the saved **5.340576171875 ns** internal grid: 65,536 steps,
350 microseconds, both voltage channels at every internal node. The reference
directory is `outputs/simulation-runs/runpod-h200-nvl-benchmark-001`. Its 75
downloaded artifacts, receipts, original source hashes and settings are verified
before preparation and execution.

For 50–250 kHz the comparison uses the **independent direct solves**, not merely
waveforms reconstructed from the identification kernel. The 300 kHz reference
uses a declared synthetic five-cycle Hann pulse. Plate 1 has no experimental
300 kHz generator/receiver record. This stage checks numerical method agreement,
not real-world accuracy. The same synthetic pulse is used on both sides.

## Method and equations

This implements the input-output **Eigensystem Realization Algorithm (ERA)**
from [Juang and Pappa, NASA, 1985](https://ntrs.nasa.gov/api/citations/19850022899/downloads/19850022899.pdf).
It reduces the fixed actuator-to-receiver dynamics learned from the full-order
simulation, using an exact dense FP64 SVD. It is not a Galerkin projection of
the plate's mass and stiffness matrices, and it does not recover full plate
displacements, energy, or internal PZT stress. Mechanical passivity and
second-order structure preservation are not claimed.

The original unit drive is applied at solver node 1. Its response satisfies
`h[0] = h[1] = 0`, and the Markov sequence starts at `m[0] = h[2] = C B`.
With outputs interleaved in each block row:

```text
H0[i*p + output, j] = m[i+j, output]
H1[i*p + output, j] = m[i+j+1, output]
H0 = U S V^T
Ar = S^(-1/2) U^T H1 V S^(-1/2)
Br = S^(1/2) V^T[:, :1]
Cr = U[:p, :] S^(1/2)
```

Retaining leading singular vectors defines the reduced orders. Truncating the
largest realization gives the same nested ERA matrices as the corresponding
smaller SVD truncations. Singular values below the standard FP64 numerical-rank
cutoff cannot support extra states; requested and effective orders are recorded.

Each reduced impulse starts with two zeros, followed by `Cr Br`, `Cr Ar Br`,
and so on. It uses a discrete FP64 recurrence at the original internal step;
there is no new continuous time integration scheme. Existing zero-padded causal
linear convolution reconstructs each target pulse. No FFT wraparound, spectral
division, response gain fitting, time alignment, phase fitting, filtering,
decimation, artificial damping, or pole clipping is introduced.

## Accuracy checks and selection

The predeclared target is **less than 0.1% raw relative voltage L2 difference**:
`100 * norm(MOR - full) / norm(full)`. It is checked separately for top and bottom
receivers and their S0-related sum and A0-related difference. These combinations
are not asserted to be perfectly pure modes. Raw waveform Pearson and envelope
Pearson correlations are also saved. An exactly zero reference norm is unresolved
and cannot pass. Neither signal is rescaled or shifted before scoring.

Choose the lowest-error eligible order using **only 50–250 kHz**. Report 300 kHz
as an additional drive-shape check without using it to select order. This is not
unseen-physics validation: the identification impulse already excites the model's
frequency content. No successful accuracy result is claimed before execution.

A candidate whose eigenvalues imply more than 1% exponential growth over the
recording is rejected and the next predeclared order is tried. This is a finite
recording screen, not proof of infinite-time stability or a transient norm bound.
Nonfinite waveforms also reject a candidate. Hardware, file, and SVD failures
stop the schedule visibly; they are not converted into successful results.

This comparison does **not** establish plate-mesh, PZT-mesh, or coupled-system
convergence, nor validate the simulation against reality. Those are separate tasks.

## Cost and limitations

ERA is useful only for the same linear, time-invariant configuration. Changing
the plate geometry, material, mesh, timestep, PZT positions, actuator/load vector,
electrical conditions, or receiver map requires a new full-order response and
a new ROM. It cannot by itself speed up the first simulation of each newly
randomized plate. A saved full-order broadband kernel already permits fast pulse
reconstruction without MOR; MOR is not guaranteed to improve that reuse case.

The full-record H0 and H1 matrices each have shape **65,534 × 32,768**. One FP64
matrix is about **16 GiB**. The initial screen estimates **112 GiB free GPU memory**
and **48 GiB available host RAM**; these are conservative estimates, not guarantees
of cuSOLVER workspace allocation. An H200-class device is the intended build
target; a typical 80 GB H100 fails this declared screen. The CPU backend rejects
large builds and exists only for small mathematical verification.

Exact SVD can have a substantial one-time cost. Large Hankel SVD cost is a known
ERA limitation discussed by [Minster et al., 2020](https://arxiv.org/abs/2003.11872).
Their randomized approximation is **not** used here. Building the model once
and evaluating a reduced system are timed separately. Future output reports the
build, all size-check costs, and selected-model online costs. Comparing current
ROM timing with the saved H200 full-order solver is historical timing, not a
fresh, controlled same-hardware benchmark. No speedup is promised.

## Preparation, transfer, and later launch

The prepared study lives in `outputs/simulation-runs/runpod-mor-comparison`.
`--prepare` writes the schedule, source snapshot, status and pending findings.
It performs no SVD, waveform reconstruction, reduced propagation, plate solve,
pod creation, or upload. `--assess` only checks and summarizes saved results.

The separate MOR Linux lock includes the original Python 3.12/CUDA 12.9
environment plus explicitly pinned cuBLAS, cuSOLVER, cuSPARSE and nvJitLink
libraries. `tools/runpod/setup_mor_linux.sh` installs and verifies that environment
on a future approved pod. Its GPU check uses only tiny matrices and an artificial
one-state system; it does not start the study. CUDA library versions and hashes
were checked against official PyPI metadata and NVIDIA's `cuda-toolkit==12.9.1`
component requirements. CuPy's SVD error mode follows the
[official CuPy documentation](https://docs.cupy.dev/en/stable/reference/generated/cupy.linalg.svd.html).

After a separately approved transfer and pod start, the explicit Linux command is:

```bash
.venv-mor-runpod/bin/python -u -m examples.pzt_mor_study --run
```

The command does not provision or start a pod. It runs stages sequentially,
skips checksum-verified completed stages on restart, and retains interrupted
attempts under `previous-attempts`. Every stage has `worker.log`, `status.json`
and a final receipt. Status prints include the stage, current frequency, elapsed
time and actual reduced propagation percentage. A heartbeat prints every 15
seconds during long operations. SVD has no invented percentage: phase and elapsed
time are reported. No unattended download/stop-pod automation has been launched
or repurposed for this study.

CPU verification covers known-order LTI recovery, six independent driven inputs,
a tiny production Verlet consistent-mass system with nearly cancelling receivers,
Hankel indexing, unstable/invalid inputs, serial resume, output checksums, and
prepare-only behavior. It does not establish full-plate MOR accuracy or GPU SVD
performance. All full-size MOR computations and simulations remain unstarted.

Validation on 2026-11-23: **13 new MOR tests and 9 existing broadband tests
passed** on CPU. Both previous transfer archive SHA256 values and all 193
previously packaged physics/example files were checked unchanged. GPU library
and full-size accuracy checks remain pending the later approved Runpod run.
