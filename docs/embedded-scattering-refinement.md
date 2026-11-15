# Defect-induced mechanical waveform refinement

The material-loss signal is the recessed response minus its matched intact
response. A large incident wave can hide a poorly resolved scattering signal
when only the total waveform is compared. This additional screen checks the
raw difference separately, while retaining the total-waveform requirements.

`configs/embedded-scattering-protocol.json` is fixed before inspecting the
full recessed Q5 result. It inherits the existing complete-window protocol:
Q3/Q4/Q5 at 196,608 steps for space; Q4 at 131,072/196,608/294,912 steps for
time. Each paired intact/recessed case must match in every case field except
the material-loss flag. The source, receive footprints, plate dimensions,
material, pulse, sample times and observation window stay fixed.

The checker validates all six original windows, their archived-source
reviews, runtime bindings and independent CPU receipts through the existing
total-waveform assessor. It subtracts complete traces at each degree/time
level, then compares the original integer sample indices. It applies no
phase alignment, interpolation or amplitude fit.

Each receiver needs an identifiable scattering trace, decreasing successive
differences and a finest relative change below 1% for space or 0.1% for time.
The temporal rate must also remain in [1.7,2.3]. Both intact and recessed
total-waveform screens must pass. A zero scattering trace does not establish
an identifiable rate. A failed screen remains a failed result; it does not
change the original total-waveform result or authorize research data.

Four manufactured-control tests verify paired subtraction, second-order
temporal behavior, zero-signal handling and rejection of changed cases/grids.
One control deliberately passes the total-waveform spatial screen while
failing the scattering screen. These are software checks rather than new
forward-physics evidence. Together with the existing comparison tests, the
final guarded run passes eight tests in 0.442 seconds.

Run `python -m examples.embedded_scattering_study --help` for the paired-run
and CPU-receipt arguments. Use `spatial` or `time`, a fresh project output
directory, and the CUDA source-transition receipt when original integrator
versions differ. The output retains both total-waveform reassessments,
paired input hashes, the raw difference archive and a sealed scattering
report. Resource limits remain 89.9% system RAM and 768 MiB minimum available.

This covers the declared 500 by 500 by 3.175 mm mechanical control at 100 kHz.
PZT voltage response, other defect/frequency cases and physical agreement
remain separate requirements. No full training or data acceptance follows
from adding this checker.
