# Embedded solver source and resource review, November 15

This review checks the current work and retained numerical evidence. It does
not identify which chat model authored each change or establish that all
earlier project work has been audited. The consolidated research plan was
read again from Google Docs; the later 3.175 mm stock, PZT variability,
no-clamp and resource instructions continue to govern new work.

## Memory and probe corrections

The external-process monitor previously used the last retained history entry.
After the base guard reached 10,000 entries, it repeatedly added current
child RSS to the same old entry. Several stopped Q5 attempts therefore report
false multi-GiB process totals. The guard now exposes a fresh latest sample
on every check, and the monitor uses it. Resource enforcement remains active
after history fills. The extended launcher also compacts its history while
retaining extrema.

Receiver probes now store their exact sparse coefficients directly. This
avoids three dense host rows of 8,599,833 float64 entries at Q5 (about
197 MiB). Zero initial states are allocated directly on the GPU, avoiding
two unnecessary host vectors (about 131 MiB). Both storage changes preserve
float64 arithmetic, the stiffness operator and the Verlet recurrence.

An initial sparse implementation used interleaved component indices instead
of the tensor system's component-major ordering. That attempt was stopped
and excluded. The corrected indices are `component * scalar_dofs + node`.
A regression compares the sparse contraction with the original `(3, n)`
dense assignment, including multiple direction components and signed weights.

The November 14 Q5 v8 attempt reached the full step count but stopped at a
genuine 2,048.8 MiB process-tree measurement during final output. It has no
completed proof archive and remains excluded. Q5 now allows 4,096 MiB
process-tree RSS while retaining the 89.9% system-RAM stop and 768 MiB
minimum available memory. The user-paused v9 run also remains incomplete;
its last logged cooling event is at step 153,680. No checkpoint was saved.

## Archived source checks

Historical runs must be checked against their original source snapshots.
Requiring every old hash to equal today's working files made the previous
Q3/Q4 evidence unreadable after these corrections. The new reader verifies
the original archive hash, every member hash, exact member inventory and
saved case, with expansion bounds. It preserves the original receipts.

Only exact current sources or the specific reviewed hash transitions in
`configs/embedded-wave-source-review.json` are accepted. Unlisted code
changes fail. The reviewed differences cover the fresh resource sample,
sparse probes, zero-state storage and resource limits. Operators, geometry,
material, excitation, time steps and convergence criteria are unchanged.

An integrator change additionally requires a passing CUDA transition receipt.
The checker loads only the hash-verified archived implementation and compares
it with the current dense and sparse paths on small positive-definite systems.
It uses 512 steps with zero and nonzero initial conditions and checks every
trace, energy sample and final state. All measured differences are exactly
zero in the retained passing run. This isolates storage/integrator equivalence;
it is not a continuum or physical-plate validation.

The first checker launch could not create CuPy's user-profile cache. Its cache
is now inside its project output directory. A subsequent test operator used
matrix multiplication requiring an unavailable cuBLASLt library. The bounded
operator now uses CuPy elementwise products and row sums, exercising the same
integrators without adding that dependency. The failed attempt is retained.

## Results and limits

- The focused suite ran 43 tests: 41 passed, two optional reference tests
  skipped, no failures or errors. Actual CUDA transition checks ran separately.
- The previously failing process-termination test referenced an unavailable
  base Python installation. It now launches the active `sys.executable`; the
  real sleeping child is terminated and reaped by the time guard.
- The archived intact Q4 time comparison was rerun after all source/hash/CPU
  checks. All three receivers still pass, with observed order about 2.000001
  and finest relative changes from 7.68e-6 to 1.16e-5.
- A fresh Q5 full window is running at
  `outputs/embedded-wave-intact-q5-n196608-20261115-v1`. It has 8,599,833
  vector degrees of freedom and requests 196,608 steps over 500 microseconds.
  Its outcome is pending. A partial log cannot establish spatial convergence.

Receipts: `docs/evidence/embedded-source-review-20261115.json` and the named
ignored output directories it binds. No training or accepted dataset was
produced. Recessed-plate convergence, representative frequency/thickness
coverage, calibrated PZT response and physical measurements remain open.
