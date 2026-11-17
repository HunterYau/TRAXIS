# Five experimental timing comparisons

The user authorized exactly five tests on November 16, 2026. The older full
research goal and its unfinished refinement queue remain paused. This work
uses five frequencies on one experimental specimen, not five independent
plates: 50, 100, 150, 200 and 250 kHz. No additional physical comparison is
authorized by this batch.

Source: Wandowski, Fiborek, Ahamad, Soman and Malinowski, *Elastic wave mode
identification and separation based on piezoelectric and fibre Bragg grating
sensors*, Smart Materials and Structures 34 (2025), 105015,
https://doi.org/10.1088/1361-665X/ae0d1d.
Author manuscript: https://zenodo.org/records/18937842.
Experimental dataset: https://zenodo.org/records/18955122,
`Alu_1_2FBG_2PZT.mat`. Its MD5 was checked against the repository metadata.
The dataset description distinguishes measured signals from separately
supplied numerical waveforms; only measured specimen-1 signals are references.

The specimen is 1000 x 1000 x 1 mm aluminum. The source does not identify its
alloy or temper, so these results cannot be called 6061-T6 validation. The
paper uses E=71 GPa, density=2700 kg/m3 and Poisson ratio=0.33. Figure 1 and
the measurement description specify a central actuator, PZT sensors 250 mm
right and FBG sensors 250 mm left. Top and bottom sensors share the same
in-plane coordinates. Actuator/sensor PZT discs are 10 mm diameter, 0.5 mm
thick, NCE51, bonded with cyanoacrylate. FBG gauge length is 10 mm. Recorded
generator signals have five Hann-windowed sine cycles; sampling is 5 MHz,
with 50 acquisitions averaged. The FBG axis is the horizontal propagation
direction shown in Figure 1. Supports and placement/dimension uncertainty
are not specified.

A 6061-T6 paper was also inspected: Miranda et al., *Lamb wave-based mapping
of plate structures via frontier exploration*, Ultrasonics (2021), author
copy https://josh.vanderhook.info/media/pdf/ultrasonics2020lamb.pdf.
It describes a 609.6 x 609.6 x 3.175 mm plate and +/-1 mm transducer placement.
The inspected copy does not provide downloadable raw signals. Its quoted
group velocities are theoretical values confirmed in experiments, not raw
measured waveforms. They were not substituted for experimental data.

## Comparison scope fixed before propagation

Run the existing 3D float64 GLL tensor elasticity operator with Q5 elements,
250 x 250 x 1 cells (28,170,018 displacement DOFs), free boundaries, and
65,536 steps spanning 350 microseconds. Existing default 12-million-DOF
guards remain; the new experiment explicitly opts into a 32-million ceiling
and byte budgets, with the existing free-VRAM checks unchanged.

Use ideal radial traction around the 5 mm actuator rim, driven by the
recorded generator waveform. The traction amplitude is a fixed 1 N/m peak;
it is not fitted to the receiver data. This ideal bonded-wafer approximation
follows the interfacial shear-traction approach of Giurgiutiu (2005),
https://doi.org/10.1177/1045389X05050106. It omits PZT/glue/fibre mass,
stiffness and electrical transfer functions. It is more relevant to a bonded
extensional PZT than the normal-force patch used in the older mechanical
controls, but it is still an approximation.

Observe mean axial strain over the 10 mm FBG gauge at x=0.25, y=0.5 on both
surfaces. Extract the antisymmetric component as (top-bottom)/2 for both
predicted strain and measured FBG voltage. Compare their A0 packet envelope
peak times using the same theory-selected window, 5-cycle pulse duration,
5 MHz sampling grid and zero padding. No fitted shift, time warp, voltage
amplitude comparison or normalization-based accuracy claim is allowed.
Following the paper, subtract half the pulse duration to report flight time.
Raw signal peaks were inspected during screening; no tolerance is fitted.
There is no predeclared real-world pass threshold. Report errors, limitations
and any window-edge failure explicitly. Save complete waveforms and final
states, independently recompute final force on CPU, and check post-burst
modified-energy conservation. These are numerical checks, not extra
experimental trials or proof of mesh/time convergence.

The independent Bloch eigenproblem screens phase and group speed errors
along the measurement direction before running. Initially selecting the
lowest Bloch eigenvalue gave wrong answers beyond the first Brillouin zone;
the implementation was corrected to select antisymmetric parity and the
unfolded branch near the requested frequency, with a regression check.
This small eigenproblem is not a substitute for full spatial convergence.
The source/search/download/reader inspection stage did not run physical
comparison trials. PDF/archive readers were installed only under ignored
project outputs; the simulation interpreter's installed packages were not
changed.

Results and resource logs are saved under
`outputs/experimental-validation-20261116/`. Each comparison is reported to
the user on completion. These five tests do not authorize training-data
acceptance, model training, the older paused queue or more experiments.
