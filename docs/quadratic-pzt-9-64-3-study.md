# Quadratic 9/64/3 PZT plate comparison

On research/pzt-quadratic-tetrahedra, the active PZT mesh protocol now schedules one combined configuration: 9 radial, 64 angular, 3 thickness subdivisions, with straight-sided 10-node tetrahedra for both displacement and electrical potential. All three attached PZT bodies use this configuration through attach_quadratic_disk; the original linear adapter remains available. Original LTS integration, mass merging and physical material files are unchanged (SHA-256 evidence in docs/evidence/quadratic-pzt-9-64-3).

The prepared study is outputs/quadratic-pzt-study-preparation/study. It uses one LTS broadband impulse kernel, reconstructs 50/100/150/200/250/300 kHz, and reports raw, envelope, wiggle and arrival metrics across all four channels using the existing plotting and resumable result pipeline. It retains the requested exact fine 5.340576171875 ns / coarse 10.68115234375 ns pair, primary only, with stability screening disabled. These steps have not been qualified for the new quadratic mesh.

The saved completed baseline was verified and imported from the external drive without rerunning it. It is LINEAR 6/32/3, so this experiment changes both element order and mesh. The isolated quadratic 6/64/2 receiver measurements are not full-plate LTS baseline waveforms and cannot replace that baseline. A single quadratic configuration does not establish quadratic spatial convergence.

## Resource limitation

15,687 nodes and 9,792 tetrahedra per body; 121,014 appended mechanical DOFs across three bodies; 28,291,032 total mechanical DOFs. The current dense attachment/coupling backend forecasts approximately 586 GiB host RAM and 280 GiB GPU memory. This exceeds the current 64 GiB component/device budgets and the known local and cloud hardware. These are estimates, not measured peaks, and do not establish a runnable configuration. Preflight correctly returns configuration_valid=true, ready=false, resource_blocked=[01-quadratic-9-64-3]. No full solve or cloud operation was performed. Memory-efficient coupling will be needed before execution; no precision, mesh or timestep was reduced to fit.

The quadratic adapter accepts an explicit node budget so the requested mesh can be described, with its existing allocation forecasts checked before dense matrices are built. Default limits remain available. No allocation checks were removed.

## Verification

21 lightweight tests passed: quadratic elements and two-sided consistent mass coupling, geometry/DOF forecasts including midside interface nodes, requested schedule, six-frequency/four-channel reporting, checkpoint/recovery and original time-step policy. Baseline verification used the archived assembly sources on the external drive. Protected source hashes match before/after. Historical configuration JSONs are retained but are not scheduled by cases().

Preflight (from this worktree with its Python environment):

```powershell
python -m examples.pzt_mesh_study --preflight --root outputs/quadratic-pzt-study-preparation/study
```

No launch command is recommended until the coupling memory requirement is addressed.
