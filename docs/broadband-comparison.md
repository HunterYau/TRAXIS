# Direct and broadband comparison study

The new workflow is on `research/broadband-comparison`. The original direct-method
branch `research/pzt-body-coupling`, commit `277842a`, remains available, as does
the original transfer archive in `outputs/runpod-transfer/transfer-001`.
The existing simulator, production queue, analytic-source trial, and old convergence
driver have not been edited to implement this method.

## What is implemented

`examples.pzt_broadband_study` runs two alternatives with the same plate, resolved
PZT bodies, reciprocal coupling, consistent mass, open receivers, zero damping,
time step, recording length, and FP64 solver:

- Direct: a separate simulation for each verified analytic five-cycle Hann pulse.
- Broadband: one discrete impulse-response simulation, followed by causal linear
  convolutions to reconstruct the same five target pulses.

Both sides of this new comparison use the continuous source already verified
against the actual recorded generator samples. The old production queue's linear
interpolation is still preserved, but mixing it with the analytic source in this
A/B test would confound the method comparison with the known interpolation issue.

The impulse is a numerical basis input. It does not assume that the physical PZT
was driven by an impulse. It excites all frequencies represented by this fixed
discrete model, including its PZT resonances. The reconstructed signals use the
actual target pulse samples on the solver's internal grid. No spectral division,
regularization, cutoff, amplitude fitting, polarity fitting, or time fitting is used.
No waveform measurements are used to tune reconstruction.

The implementation is deliberately limited to the current fixed, linear,
zero-initial-state configuration. It must not reuse a kernel after changing a
mesh, time step, material, geometry, PZT position, source load vector, electrical
boundary condition, or actuator. Different actuators need independent kernels.
Nonlinear or time-varying physics would need a different method.

## Why the indexing and finite recording are valid

With `u[0]=v[0]=0` and `F[0]=0`, the production velocity-Verlet method has

```text
u[n+1] = (2I - dt^2 M^-1 K) u[n] - u[n-1] + dt^2 M^-1 b F[n]
u[0] = u[1] = 0
```

The same equation holds for consistent coupled mass, with its existing inverse
mass operation. Sampling force at node zero would have a special half-weight in
the first Verlet position update. To avoid that boundary convention, the
identification run uses `F[1]=1`, with all other discrete force samples zero.
Its recorded voltage is `h[n]`, with `h[0]=h[1]=0`. The target voltage is

```text
y[n] = sum(j=0..n) h[n-j] F[j+1]
```

`scipy.signal.fftconvolve(..., mode='full')` computes zero-padded linear
convolution, and only the first recording-length samples are retained. This
prevents a persistent lossless tail from wrapping to the beginning of the trace.
Only past kernel values affect any reconstructed output, so a complete decay of
the impulse tail is not required to reconstruct the available finite interval.
This differs from treating a truncated, still-ringing recording as a periodic
frequency response and dividing FFTs.

All internal time samples are retained in the new direct and impulse runs.
Convolution happens **before** reducing the recording cadence. Method agreement
uses those full internal-grid voltages. Nested time refinements use exact common
sample nodes, without interpolating or shifting.

## Comparisons and recorded results

The report records raw relative voltage L2 error, absolute voltage RMSE, signed
Pearson correlation, unit-normalized waveform difference, peak-to-peak voltages,
and envelope Pearson correlation. Top and bottom receivers, their S0-related sum,
and their A0-related difference are reported separately. The raw voltage metric
is primary; normalization is a separately named shape metric.

Both methods also pass through the **unchanged** experimental comparison function
at the original 42.724609375 ns recording cadence, with the existing arrival,
envelope, signed waveform, and separately labelled 5 MHz anti-alias metrics.
Point-sampling to that historical cadence is retained for consistency; it is not
a newly calibrated oscilloscope filter. Full internal-grid voltages remain saved.

Runtime records separate build, solver, reconstruction, comparison, and total
worker times. The five-frequency benchmark compares the sum of five direct
workers against one broadband worker including all five reconstructions and
their experimental comparisons. Both methods have the same output stride of one.
The comparison covers complete fresh-worker workflows, not an optimized direct
implementation sharing a preassembled system across frequencies. Startup/kernel
cache state can affect timing: workers share a cache, direct runs are first at
baseline, and order alternates at later resolutions. A single execution is an
initial speed measurement, not a statistically controlled timing benchmark.

Each task has a log, status, checksummed result receipt, and waveforms. Each
reconstructed frequency has its own comparison and reconstruction metadata.
`findings.json` and `findings.txt` aggregate method differences, convergence, and
runtime after each completed solver task. Corrupt or missing artifacts cannot be
silently skipped. Interrupted attempts are archived and restart from the beginning;
there is no mid-step restart checkpoint or automatic accuracy reduction.
`experimental-results.txt` collects the existing real-plate comparison scores;
those scores are also printed when each direct or reconstructed frequency finishes.

## Prepared schedules (not started)

Prepare the short method benchmark:

```bash
python -u -m examples.pzt_broadband_study --root outputs/simulation-runs/runpod-broadband-benchmark --prepare --suite benchmark
```

This schedules **six solver runs** at the current mesh/time step: five direct
frequencies from 50 to 250 kHz and one broadband impulse. Its results cannot
establish mesh or PZT convergence.

Prepare the larger convergence comparison:

```bash
python -u -m examples.pzt_broadband_study --root outputs/simulation-runs/runpod-broadband-convergence --prepare --suite convergence
```

This schedules **39 solver runs** over twelve resolutions. Baseline has all five
direct frequencies plus one broadband kernel. Eleven refinements have direct
100 and 250 kHz runs and one broadband kernel each; every kernel reconstructs all
five frequencies. The twelve resolutions are the existing audit's baseline,
half/quarter time step, plate meshes 200/230/260, plate thickness 1/2 at 190 cells,
PZT radial/angular/thickness refinements, and combined fine case.

It compares spatial, temporal, PZT, and coupled refinements for **100 and 250 kHz**
in both methods. It also reports temporal difference ratios and observed orders.
It does not establish convergence at every frequency or an absolute physical
error bound. The reporting target remains less than 1% raw difference per channel;
coarse levels can exceed that target while revealing a useful refinement trend.
No automatic mesh adoption or training-data approval is performed.

The convergence suite includes its own baseline; running both complete suites
would repeat the baseline benchmark. Review cost and choose the intended suite
before launching. Six runs suffice for the initial A/B check; 39 include the
planned refinement comparisons. Neither is launched by preparation or Linux setup.

After explicit approval to use a pod, activate its prepared environment and run
the chosen prepared schedule inside a detached `tmux` session, for example:

```bash
set -o pipefail
python -u -m examples.pzt_broadband_study --root outputs/simulation-runs/runpod-broadband-benchmark --run 2>&1 | tee -a outputs/simulation-runs/runpod-broadband-benchmark/console.log
```

That explicit `--run` command starts simulations. `--prepare` and `--assess` do not.
Prepared study paths are relative to the project and work after transfer; source
hashes are checked again before running. Any intended source change needs a fresh
prepared study folder. The unchanged direct drivers remain available separately.

## Local implementation checks

Cheap tests exercise the **actual production Verlet stepping function** with a
NumPy backend replacing only GPU array/probe operations. They verify a coupled
consistent-mass system with nearly cancelling receiver channels, random inputs,
multiple tone pulses, impulse timing, absence of circular wraparound, rejection
of unsupported initial forcing, and time refinement against an independent
`scipy.integrate.solve_ivp` continuous reference. A separate small LTI fixture
tests the complete worker/output/experimental-comparison pipeline and detects
tampered saved waveforms. Neither fixture represents the full physical plate.

**Full-plate GPU equivalence, runtime speedup, and convergence are not yet tested.**
Small CPU checks establish implementation behavior, not experimental validation.
The existing solver's GPU kernel checks are inherited; no new GPU/plate run was
started for this preparation.

## Primary references

- [Ansys: response to an arbitrary input from an impulse response](https://optics.ansys.com/hc/en-us/articles/360034915813-Impulse-Analysis-Calculating-the-response-to-an-arbitrary-time-signal-using-the-impulse-response).
  Supports the reuse/convolution principle, coverage requirement, and checking
  reconstructed signals against separate direct simulations. Its example concerns
  electromagnetic FDTD, not this plate; our discrete Verlet indexing is derived above.
- [COMSOL: reconstructing transient signals with FFT/IFFT](https://www.comsol.com/support/learning-center/article/using-the-inverse-fast-fourier-transform-to-reconstruct-a-transient-signal-68721).
  Supports linear-system reconstruction and the need to retain sufficient spectrum.
  We do not copy its periodic harmonic reconstruction onto a still-ringing record.
