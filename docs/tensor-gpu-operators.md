# Tensor GPU operator and full-size capacity

The profile-band study supports measuring the intact 3 mm plate with Q4 GLL,
3.333 mm lateral cells and one thickness cell. The coarser 5 mm carrier-only
choice fails the sampled pulse-band design screen. Uniform 1 mm regions need
the tested 2.5 mm spacing or another independently verified choice. Those
findings motivate this capacity measurement; they do not accept wave data.

## Direct structured action

`traxis/gpu_tensor.py` applies GLL elasticity in float64 with tensor
gradient/stress/weak-divergence contractions. One CUDA block handles an
element, with shared displacement and flux buffers. Global GLL indices are
computed directly from the structured cell coordinates. No global sparse
matrix, dense element matrix or element-index table is stored on the device.
Affine transforms and component ordering match the independent tensor model.

The large-layout API is separate from the bounded dense reference. It checks
counts, a conservative sixteen-vector device-storage budget and host-buffer
limits before allocating full vectors. Positive mass is assembled from the
one-dimensional shared-node weights. A local generalized Gershgorin bound
also bounds the global Rayleigh quotient because global mass/stiffness are
sums of those local forms. This supplies the explicit stability limit.

The independent streamed CPU action gathers bounded element chunks, applies
NumPy tensor contractions and scatters them. Small tests compare its complete
field with the native-checked assembled operator on rectangular/skewed grids,
including multiple chunk sizes and shared interfaces. Mass and coordinates
agree with the small reference. Tests also cover pre-allocation limits and
the local bound against a directly assembled global spectrum.

## Native CUDA and propagation verification

The first eight native cases cover Q1-Q4 rectangular/skewed two-element grids.
Every GPU stiffness column is compared with native GLL assembly. Additional
Q3 2-by-2-by-2 and skewed Q4 1-by-2-by-2 grids check sharing in all axes, up to
1,215 vector coefficients. All ten native cases pass at no more than 2.57e-15
relative full-matrix discrepancy. Positive mass agrees within 8.82e-16.
Negative operator scaling and independently streamed random fields also pass.

The checked action works with the existing device-resident Verlet integrator.
A small native highest-frequency eigenmode has an exact sinusoidal solution.
At the fixed 0.112387-microsecond interval, 32/64/128-step displacement errors
have ratios 4.001095 and 4.000274. Maximum modified-energy range is 5.08e-16.
Independent final-state energy recomputation agrees within 1.0e-15. This is
a discrete integrator check, not the full 500-microsecond plate waveform.

Ten selected CPU tests and two explicitly enabled native/CUDA tests pass
without skips. The groups overlap earlier tensor checks and are reported
separately from the last complete 268-test suite. The small reference worker
takes 7.1874 seconds at 492.10 MiB peak combined RSS and 51 C GPU peak.

## Actual 500 mm operator measurement

`examples.gpu_tensor_reference --mode full` measures the intact
500-by-500-by-3 mm plate with 150-by-150-by-1 Q4 elements. This constructs
22,500 elements, 1,806,005 scalar nodes and **5,418,015 displacement
coefficients**. It performs zero propagation steps.

The positive mass sums to 2.025 kg, with 2.23e-16 relative total-mass error.
Normalized first/second box moments agree within 1.14e-13. The full GPU action
on a prescribed algebraic field agrees with the independent streamed CPU
action to 9.87e-16 relative. This checks the full indexing/scatter operation,
not a small subset or extrapolated node count.

| Measurement | Result |
| --- | ---: |
| Complete monitored worker | 10.6254 s |
| Host GPU action including transfer | 0.02863 s |
| Independent streamed CPU action | 0.97381 s |
| Five isolated device actions | 4.488-5.386 ms |
| Conservative explicit stability bound | 37.8375 ns |
| Peak combined process RSS | 631.20 MiB |
| Peak system RAM / minimum available | 50.7% / 8,002.20 MiB |
| Sampled system-wide GPU memory / temperature | 1,097 MiB / 52 C |

The process-tree cap is 1.5 GiB, with 85% system RAM, at least 2 GiB available,
90% VRAM and 80 C GPU guards unchanged. CPU temperature remains unavailable.
The device screen budgets sixteen vectors (661.38 MiB); the capacity probe
does not allocate every future integrator vector. Isolated action timings
exclude sustained stepping, source/probe work and thermal pacing. They are
not a measured complete-window runtime or a speedup against another solver.

## Reproducible records and remaining work

```powershell
.\.venv\Scripts\python.exe -m examples.gpu_tensor_reference --mode small --output outputs/tensor-gpu-small
.\.venv\Scripts\python.exe -m examples.gpu_tensor_reference --mode full --output outputs/tensor-gpu-full
```

Actual runs are `outputs/tensor-gpu-small-20261109` and
`outputs/tensor-gpu-fullplate-operators-20261109`. The supplemental native
checks are in `outputs/tensor-gpu-shared-native-tests-20261109`. All processes
terminate normally. Plans bind sources, native/GPU runtime, material and
snapshots; raw matrix columns, modal states, full field, coordinates and mass
are retained. The nonfatal CuPy CUDA_PATH-detection warning remains in logs.
There is no global CUDA/toolkit configuration change.

`evidence/tensor-gpu-debug.json` binds an independent raw-column, modal-energy,
full-action and moment audit, fingerprint
`416e1a05f92203ced3548f65a4d8c486785512afe7b2bc72b7e64b486ccf5ad7`.
Its collector is `outputs/tensor-gpu-tools-20261109/collect_gpu.py`.

Physical circular source/probe integration, complete 500-microsecond waves,
space/thickness/time refinement, conforming recesses and PZT/electrical
calibration remain required. This capacity result does not change the
earlier failed tetrahedral thickness screen or accept a training bank.
No reconstruction/policy research training runs in these measurements.
