# Independent elastic assembly with Elmer

Elmer provides an independent check of the local 3D elastic stiffness and mass
assembly without a commercial license. The tested rectangular and skewed
meshes agree near floating-point precision. This does **not** resolve the
[failed spatial waveform screen](waveform-validation.md): two implementations
can agree on the same inadequate coarse discretization.

## Project-local setup

The reference trial uses CSC's portable, no-GUI, serial Windows release ZIP:

- [Official release 26.1 directory](https://www.nic.funet.fi/pub/sci/physics/elmer/bin/windows/rel26.1/)
- Package: ElmerFEM-nogui-nompi-Windows-AMD64-rel26.1.zip
- Download size: 129,530,276 bytes; unpacked files: 454,719,500 bytes.
- Locally measured SHA256: `8ce866cc6ff16a842f01edc1dc56c9a801cdd758210ec02cf1353398e6839455`.

The ZIP is unpacked under outputs/reference-tools/elmer-26.1. Binaries and
downloaded sources are ignored outputs, not committed dependencies. No
installer ran, and no global PATH, registry or environment settings changed.
The command supplies Elmer's home/library paths and one-thread settings only
to its child process. It never downloads or installs software automatically.
The checksum identifies this trial's artifact; it is not a separately
published vendor checksum.

The solver banner reports **9.0, Rev: Release, compiled 2026-01-20**, despite
the distribution being named release 26.1. Reports preserve that banner and
executable/module hashes instead of replacing it with an inferred version.
Both ElmerSolver and ElmerGrid version/help probes succeeded. The current
nightly-download notice says that new builds may instead be available through
GitHub Actions; the trial deliberately used the pinned release package.
See [CSC's download notice](https://www.nic.funet.fi/pub/sci/physics/elmer/bin/windows/Readme1st.txt)
and [Elmer's official repository](https://github.com/ElmerCSC/elmerfem).

Elmer uses GPL/LGPL components; preserve its included license notices if
redistributing binaries. The command can also point at an existing
installation through `--elmer-home`.

## Run the bounded check

```powershell
.\.venv\Scripts\python.exe -m traxis verify-elmer --elmer-home outputs/reference-tools/elmer-26.1/ElmerFEM-nogui-nompi-Windows-AMD64 --output outputs/elmer-reference
```

Use a new/empty output directory. The default guards allow 60 seconds and
512 MiB sampled RSS for the Python monitor plus external process tree, with
system RAM, GPU memory and temperature limits. External windows stay hidden.
An exceeded guard terminates and reaps the external process and its observed
descendants. Logs are capped at 4 MiB. CPU temperature is best-effort and may
be unavailable. File/measurement errors fail the check instead of reporting
zero external memory.

The workflow exports an 18-node, 24-tetrahedron free block twice: rectangular,
then under an invertible affine skew with a different material. The skew is
an operator benchmark, not a physical plate specimen. All coordinates are
meters; unused nodes and degenerate tetrahedra are rejected. There are no
displacement constraints, springs, damping, loads or transducer domains.

Elmer's StressSolver assembles stiffness and consistent mass. It saves at
the assembly slot and stops before eigen solution, matrix scaling or
constraints. The reader checks serial matrix dimensions, finite values,
integer indexes, duplicates and a complete node permutation; it restores
node-major x/y/z coordinates before comparing matrices. The export flags
and serialization are implemented in
[the pinned SolverUtils.F90](https://github.com/ElmerCSC/elmerfem/blob/5a8de867068be0568f09af40fb90fee300dfbede/fem/src/SolverUtils.F90),
and the assembly comes from
[StressSolve.F90](https://github.com/ElmerCSC/elmerfem/blob/5a8de867068be0568f09af40fb90fee300dfbede/fem/src/modules/StressSolve.F90).
This source tag is a reference for interpreting the release's interface;
the downloaded binary has no embedded commit identity establishing an exact
source-to-binary match.

The release's keyword dictionary warns that some implemented save flags are
unlisted. The expected messages concern `linear system save mass`, `linear
system save and stop` and `linear system save prefix`. They are retained in
the log, together with the intentional no-BC warning for a free mesh. An exit
code alone is insufficient: the save-and-stop marker and all fresh matrix
exports must exist and pass the reader checks.

The reference stiffness is compared with the local elastic assembly. The
consistent mass is checked against the exact linear-tetrahedron formula
`rho * V * (ones + identity) / 20`; its row sum must equal the local lumped
mass. Symmetry and positive-definite mass are checked before a small dense
generalized eigen analysis. There must be six near-zero rigid modes followed
by a positive seventh mode. Matrix errors must be below 1e-10 relative norm;
the first six eigenvalues must be below 1e-10 of the largest in magnitude.
These thresholds are fixed in code, not fitted to waveform agreement.

## Measured results, 2026-11-04

| Quantity | Rectangular block | Affine-skewed block |
| --- | --- | --- |
| Degrees of freedom | 54 | 54 |
| Relative stiffness Frobenius error | 8.76e-15 | 1.38e-14 |
| Relative consistent-mass Frobenius error | 2.57e-15 | 3.70e-15 |
| Relative row-summed/lumped-mass error | 2.53e-15 | 3.04e-15 |
| Six rigid modes with positive seventh | Passed | Passed |

The command completed in 0.53 seconds, with sampled combined RSS 105.0 MiB,
system RAM 61.3%, GPU usage 921/8,192 MiB at 47 C. Computation used CPU; GPU
readings describe the system, not allocations by Elmer. CPU temperature was
unavailable. The compact committed record is
[evidence/elmer-reference-debug.json](evidence/elmer-reference-debug.json).
Full SIF/native mesh inputs, logs, matrix exports and fingerprints remain in
outputs/elmer-reference.

Set the process-local `TRAXIS_ELMER_HOME` environment variable to include the
real solver integration in the regression suite. Otherwise only that optional
test is skipped; parser, exporter and process cleanup checks still run.

All 150 tests passed with the real Elmer test enabled. The monitored regression
run took 59.49 seconds, peaking at 779.1 MiB sampled combined RSS and 64.8%
system RAM. GPU usage stayed at 921/8,192 MiB with a maximum observed
temperature of 47 C; CPU temperature was unavailable.

This check uses P1 tetrahedra and does not establish Elmer's high-order or
transient performance for this plate. A higher-order/mesh/time-converged
finite-plate reference is still needed. Elmer's source contains a separate
[2D static piezoelectric test](https://github.com/ElmerCSC/elmerfem/blob/5a8de867068be0568f09af40fb90fee300dfbede/fem/tests/piezo/piezo.sif);
it has not been used here to validate transient bonded-PZT transmission or
voltage readout. Actual PZT, adhesive, poling, drive and receiver parameters
remain required. Every result retains `ready_for_training_data=false`.
