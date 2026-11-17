# Possible explanation for the 100/150 kHz discrepancies

Post-processing only; no additional plate simulations, fitting or changes
to the five saved comparisons were performed.

The current receiver is mean axial strain over a 10 mm gauge. For a plane
wave its spatial averaging factor is sin(kL/2)/(kL/2). With the frozen
plate properties and A0 dispersion, its first cancellation occurs at
92.246 kHz. The ideal 5 mm-radius circular actuator has a J1(ka) tuning
factor whose first cancellation occurs at 133.383 kHz. These are ideal
model estimates, not measured transfer functions of the real equipment.

A five-cycle Hann pulse spans a band of frequencies. Its main-lobe zeros
are approximately 0.6 and 1.4 times the carrier: 60-140 kHz for the
100 kHz pulse and 90-210 kHz for the 150 kHz pulse. Both contain the
cancellation regions. The corresponding regions are outside the 50 kHz
main lobe and farther from the centres of the 200/250 kHz bands. This is
a plausible frequency-dependent source of spectral weighting, packet
distortion and envelope-peak timing shifts in a dispersive A0 wave.

The saved 100 kHz simulation has envelope maxima at 156.4 and 182.0 us;
the latter is 95.49% of the earlier maximum. The measured maximum is
170.2 us. Selecting a different lobe could change the reported delay
substantially, but the original fixed peak rule is retained. At 150 kHz,
the selected model peak is 130.8 us versus measured 141.8 us. The real
experiment uses full bonded PZT dynamics and optical FBG sensing, while
our approximations omit their transfer functions. The source paper uses
an optical transfer-matrix model rather than plain mean strain.

Carrier-frequency Bloch group-speed discrepancies are only 0.00468% at
100 kHz and 0.02505% at 150 kHz; these do not establish full transient
convergence, but are much smaller than the packet timing discrepancies.
The 0.2 us timing grid is also much smaller than 11-13.8 us. Neither
observation alone rules out a numerical or processing contribution.

Conclusion: actuator/sensor tuning and the packet-peak metric are leading
hypotheses. A causal attribution still requires independent transfer-model
checks and full numerical convergence. Mode overlap, imperfect real
top/bottom cancellation, bonding, damping and unreported setup details
remain alternatives. Do not change the plate properties or select a more
favourable peak to force agreement.

Evidence: [diagnostic numbers](evidence/experimental-frequency-mismatch-diagnostic.json).
Sources: [experimental author manuscript](https://zenodo.org/records/18937842),
[circular actuator tuning study](https://research.cec.sc.edu/files/lamss/files/j123_jimss_2018_mei_giurgiutiu.pdf).
