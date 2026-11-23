# Saved-run phase investigation

Run 010 was examined without rerunning the solver or replacing any saved validation
scores. `examples/diagnose_pzt_phase.py` reproduces the diagnostic from saved voltages:

```powershell
.\.venv\portable-runtime\python.exe -m examples.diagnose_pzt_phase --root outputs/simulation-runs/run-010-wandowski-restored-ideal-properties --output outputs/result-analyses/run-010-phase-diagnostic
```

The script verifies waveform hashes and compares full records and the union of the
independently selected adaptive packet intervals. It fits two alternative diagnostic
hypotheses: a constant quadrature phase rotation (with gain), or a time delay bounded
by half a carrier period. Neither is a physical calibration or corrected accuracy
score. No stored voltages, arrival times, or original comparison scores are changed.
The same packet bounds can contain reflected or residual modes. A0 above the reported
detection range remains unconfirmed. Cropped Hilbert quadrature has boundary effects.

## Findings

Packet normalized wiggle differences, original -> fitted constant phase (%):

| kHz | S0-related | A0-related |
| --- | --- | --- |
| 50 | 117.69 -> 53.49 | 18.56 -> 15.08 |
| 100 | 105.29 -> 25.38 | 62.07 -> 15.30 |
| 150 | 78.06 -> 21.44 | 109.18 -> 30.90 |
| 200 | 96.15 -> 26.79 | 160.15 -> 33.55 |
| 250 | 115.93 -> 39.51 | 152.90 -> 29.02 |

Phase explains a large part of the disagreement, with substantial residual shape
differences. S0 and A0 require different phase/delay adjustments, including opposite
delay signs. A single global clock correction is therefore not supported.

## Independent author simulation cross-check

Downloaded the researchers' public `PZT_200kHz_numerical.mat` from
https://zenodo.org/records/18955122 and verified the published MD5
`0f09d0cc3e8daeaeddc3517bdb7dbff2`.
The `timePZT` vector runs 0..0.6. Interpreting this as milliseconds matches the
paper's sub-millisecond record and 200 kHz excitation. MAT metadata does not explicitly
state the unit, so this cross-check is conditional on that interpretation.

For the top receiver over a fixed illustrative 40..85 microsecond S0 interval:

| Pair | Original normalized wiggle difference | Fitted phase |
| --- | --- | --- |
| Our simulation vs experiment | 94.07% | -54.7 degrees |
| Authors' simulation vs experiment | 92.80% | -52.9 degrees |
| Our simulation vs authors' simulation | 8.40% | -1.6 degrees |

This is evidence against a phase error unique to our solver in this packet. It does
not validate all modes, frequencies, meshes, or the experimental reconstruction.
The illustrative later 100..140 microsecond packet differs more (37.58% between
simulators). Those intervals are not guaranteed isolated modes.

## Sources and limits on a physical fix

- Original paper and public preprint: https://zenodo.org/records/18937842,
  DOI https://doi.org/10.1088/1361-665X/ae0d1d. Local extracted paper reports sensor
  differences potentially caused by solder/wire mass, asymmetric wrap-around
  electrodes and glue. It explicitly leaves that investigation beyond its scope.
- Manufacturer NCE51: https://www.ctscorp.com/Product-Series/NCE51.htm.
  Its d31=-208, d33=443 pC/N and relative permittivity 1900 agree with our nominal
  profile. Qm=80 and dielectric loss=0.015 are already recorded in the profile;
  they do not supply a measured broadband complex transfer function for the bonded disk.
- NI instrument response documentation:
  https://www.ni.com/en/support/documentation/supplemental/17/specifications-explained--ni-oscilloscopes-and-digitizers.html.
  Actual acquisition settings and amplifier/PZT transfer are not established by
  equipment model names alone.
- Analytic signal phase/envelope method:
  https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.hilbert.html.

No incorrect sample timestamp or half-step offset was found in the inspected
integration/probe path: displacement is sampled at its corresponding integer time.
This is a source inspection, not a complete independent solver audit.

No physical phase correction is justified yet. Fitting each experimental packet and
reporting that as accuracy would hide the mismatch. The next discriminating evidence
would be measured actuator-terminal voltage and bonded-sensor/electronics transfer,
or a controlled numerical PZT/plate convergence study. Catalog values alone cannot
identify which unknown caused this phase difference. No invented PZT values were added.

Machine-readable diagnostics, source hashes, raw plots and the conditional author
cross-check are saved under `outputs/result-analyses/run-010-phase-diagnostic/`.
