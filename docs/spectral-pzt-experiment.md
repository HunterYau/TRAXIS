# Spectral hexahedral PZT component experiment

Branch: `research/pzt-spectral-hexahedra`, separate worktree
`.worktrees/pzt-spectral-hexahedra`. All 404 inherited Python/configuration
files match the original quadratic worktree byte for byte (SHA-256 evidence
under `docs/evidence/spectral-pzt/`). Its uncommitted work was copied without
changing or discarding anything there. No RunPod action occurred, and no
existing plate solver, LTS recurrence, or simulation schedule was changed.

## Method and controls

The new optional component backend uses conforming tensor-product Lagrange
fields at Gauss–Lobatto–Legendre nodes, for both mechanical displacement and
electrical potential. Q2 and Q3 are tested. A square core and four annular
blocks produce an all-hexahedral disk without collapsed polar elements.
The bilinear in-plane / affine thickness geometry has the **same regular
64-sided polygon boundary** as the P2 tetrahedral reference. Each side has
16 in-plane cells; this is not a circular curved-boundary approximation.

The material is the same full NCE51 3D tensor profile; radius is 5 mm and
thickness 0.5 mm. The same analytical S0/A0 surface displacements are
prescribed on the entire lower face, for a 1 mm aluminum plate with
E=71 GPa, nu=0.33, rho=2700 kg/m3, at 50/100/150/200/250/300 kHz. The 1 nm
input scale, free mechanical exterior, grounded lower electrode,
equipotential ideal-open upper electrode, and electrical interior DOFs are
preserved. Differences use complex voltages with no fitted phase, gain,
arrival alignment or normalization. No actual plate is propagated.

All arrays and solves are FP64. Full consistent mass is integrated with
Gauss quadrature. The usual collocated GLL diagonal mass approximation is
deliberately excluded to control mass treatment. Thus these tests do not
measure the potential benefit of changing to a diagonal-mass SEM scheme.

The reference is the saved quadratic tetrahedral 6/64/2 component. A repeat
with the same CPU, two BLAS threads, MMD_AT_PLUS_A sparse ordering and all
six frequencies reproduced the saved voltages to roundoff. The original
tetrahedral mesh is not an exact or demonstrably converged solution.

## Measurements

The clearest Q2 match uses 16 cells per square-core side, two annular radial
cells and one thickness cell. Each hex has a 3x3x3 Q2 nodal grid. These cell
counts have different meanings from the tetrahedral rings/sectors/layers.

| Metric | P2 tetrahedral 6/64/2 | Q2 hexahedral candidate |
|---|---:|---:|
| Nodes | 7365 | 4803 |
| Elements | 4224 tetrahedra | 384 hexahedra |
| CPU time, six frequencies / both modes | 25.89 s | 3.72 s |
| Sampled peak process RSS | 434 MiB | 237 MiB |
| GPU-products/CPU-factor stiffness action | 3.25 ms | 2.06 ms |
| CPU-factor consistent mass inverse, GPU vectors | 3.04 ms | 1.25 ms |
| Sum of the last two action costs | 6.29 ms | 3.30 ms |

The serial repeat was approximately 6.96x faster in the CPU harmonic
component benchmark. The fixed-face operator proxy was approximately 1.90x
faster on the local RTX 3060 Ti, with GPU/CPU action differences below
3e-16. The GPU proxy has no host plate or contact interpolation; it cannot
predict the whole LTS simulation speed or H100 performance. Tiny component
CPU/GPU/library costs and shared system activity affect timings. These are
small benchmarks, not a statistical performance study.

Q2 candidate voltage differences from the saved P2 reference:

| kHz | S0 | A0 |
|---:|---:|---:|
| 50 | 0.0894% | 0.0438% |
| 100 | 0.0826% | 0.0453% |
| 150 | 0.0720% | 0.4870% |
| 200 | 0.0590% | 0.7334% |
| 250 | 0.0456% | 0.7290% |
| 300 | 0.0346% | 0.9133% |

Six-frequency aggregate relative L2 differences are 0.0528% for S0 and
0.6480% for A0. All individual candidate differences are below 1%, but
several exceed 0.1%.

## Refinement limits the conclusion

The Q2 radial study deliberately continued after finding a close match.
Here all cases have the same 64-sided boundary, 16 core cells per side,
one thickness cell and Gauss order 6. Times in this table include overlapping
local diagnostic work; use the serial-repeat table above for timing claims.

| Annular radial cells | Nodes | Seconds | S0 aggregate difference from P2 | A0 aggregate difference from P2 | Worst individual A0 difference |
|---:|---:|---:|---:|---:|---:|
| 1 | 4035 | 2.07 | 1.1712% | 5.2375% | 16.5574% |
| 2 | 4803 | 4.17 | 0.0528% | 0.6480% | 0.9133% |
| 3 | 5571 | 10.83 | 0.3719% | 1.6919% | 8.5276% |
| 4 | 6339 | 33.33 | 0.5904% | 2.2824% | 12.5348% |
| 5 | 7107 | 11.45 | 0.7173% | 2.6237% | 14.7792% |

Refinement moves Q2 **away** from the P2 reference after the two-ring case.
Successive aggregate A0 changes decrease (5.63%, 1.36%, 0.66%, 0.37%), but
the last worst individual-frequency change remains about 1.99%. The close
two-ring match may therefore be error cancellation between two spatial
approximations. It is not proof of true accuracy or full convergence. A0's
low net voltage around 150 kHz amplifies relative changes. Sparse LU fill
and mesh adjacency also make runtime/memory nonmonotonic with node count.

Q2 appears promising as a lower-cost representation, but it is not promoted
to the plate solver or declared an optimal mesh.

## Q3 follow-up

Q3 was attempted only after Q2 was found inexpensive but refinement-sensitive.
Keeping the exact same polygon requires the same 16 cells per core side.
The one-annular-cell / one-thickness-cell Q3 mesh has 11,908 nodes and far
more coupled harmonic equations. The six-frequency attempt exceeded the
180 s diagnostic budget after reaching the 150 kHz stage, with approximately
1.84 GiB peak RSS. It was stopped locally; no pod or plate calculation was
stopped. Its attempt metadata and progress log are preserved.

A separate bounded two-frequency check completed in 128.53 s, with about
1.88 GiB peak RSS:

| kHz | Q3 S0 difference from P2 | Q3 A0 difference from P2 |
|---:|---:|---:|
| 150 | 0.4691% | 4.7395% |
| 300 | 0.2808% | 1.2897% |

It did not provide closer agreement with the P2 reference than the cheaper
Q2 candidate at those frequencies. This does not prove Q3 is less accurate:
P2 is an approximation too. Given the cost, no Q4/higher mesh trials were
launched. No six-frequency Q3 result is claimed.

## Verification and remaining work

Ten spectral/quadratic tests passed, including Q2 and Q3 volume, electrode
capacitance and constant strain charge, bending energy, full mass, symmetry,
rigid translation, GLL basis completeness and shared nodes. Bending energy
uses a tolerance appropriate for cancellation of large membrane terms in
FP64. Q3's full matrix/response integration accuracy is not yet qualified.

For the Q2 candidate, increasing Gauss order from 6 to 8 changed voltages
by at most about 1e-10 percent, so its observed response differences are not
caused by that integration choice. Representative annular element matrix
changes for Gauss 6 to 8 were up to 7.4e-7 relative for Q2 and 1.4e-5 for
Q3; consistent mass changed only at roundoff. The Q3 voltage comparison
remains provisional because a full paired quadrature study was not done.

Before production use, the promising Q2 formulation needs a converged
reference, thickness/angular checks, attached-plate mechanical coupling,
electrical backreaction, time-step/stability qualification and an actual
matched LTS benchmark. No plate time-integrator adapter or simulation
schedule has been changed by this experiment.

## Reproduction

From this worktree in the project Python environment:

```powershell
python -m unittest tests.test_spectral_piezoelectric tests.test_quadratic_piezoelectric
python -m examples.spectral_pzt_experiment --order 2 --side 16 --radial 2 --layers 1
python -m examples.spectral_pzt_experiment --order 3 --side 16 --radial 1 --layers 1 --max-equations 40000 --frequencies 150000 300000
python -m examples.spectral_pzt_action_check --local-cuda-root ../.. --output outputs/spectral-pzt-experiment/actions
python -m examples.spectral_pzt_report
```

Use fresh output folders for repeats; completed receipts are reused, while
unfinished receipts are preserved and rejected for overwrite. The Q3 CLI
call is expensive even though it is a component test; the actual trial was
supervised with a separate 150 s / 2.5 GiB process budget. The initial full
Q3 attempt used 180 s. New and inherited raw results, timings, exceptions,
checksums and plots are retained in `docs/evidence/spectral-pzt/`.

## Research basis

- [Mathematical and numerical modelling of piezoelectric sensors](https://www.numdam.org/item/10.1051/m2an/2011070.pdf): spectral hexahedral piezoelectric formulation and electrical coupling; its use of diagonal mass is distinguished from this consistent-mass control.
- [MFEM tensor-product basis functions](https://mfem.org/basis-functions/): tensor-product hexahedral H1 basis construction.
- [MFEM numerical integration](https://mfem.org/integration/): physical-element integration and mapped gradients.
