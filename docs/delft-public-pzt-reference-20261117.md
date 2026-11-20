# Delft public PZT reference

## Finding

Ewald, Groves and Benedictus, [published experiment](https://doi.org/10.1177/1475921720933172).
This adds a major missing material detail relative to our current Wandowski
reference: the experimental plate's alloy **and temper** are identified.

- **7075-T6 aluminum**, 1000 x 500 x 2 mm.
- **APC850** discs, diameter 9.52 mm, thickness 1 mm; reported resonance 207 kHz.
- PZT actuation and reception, with sensor coordinates and hole geometry.
- Agilent 33500B generator and PicoScope 6402A oscilloscope.
- Physical excitation: five-cycle Hann burst at **200 kHz**.
- Superglue attachment discussed; bond thickness and full mechanical properties
  are not established. Waveform amplitude calibration/units need further checking.

This improves material identification, not every reconstruction requirement.
It is a separate single-frequency candidate, not a complete replacement for
the current five-frequency reference. No simulation or queue change was made.

## Verified public access

[4TU dataset](https://doi.org/10.4121/uuid:71354fa6-a5e2-49f3-80ab-ccc7ed1ec327),
[author GitHub repository](https://github.com/vewald/heuristicsensor).

Downloaded the university archive anonymously without contacting anyone:
`outputs/reference-screening-20261117/delft-ewald2020/data.zip`.
Size 21,944,147 bytes. MD5 matches the repository's published value:
`f94a591f08a683866a5f3c6060d187d3`.

The archive contains eight measurement workbooks, an explicit sensor-to-column
mapping and MATLAB analysis scripts. Scripts were read, never executed.
Baseline3S has time, actuator and four receiver columns; Baseline5S has time,
actuator and six receiver columns. Both contain 39,066 numeric records.
With the supplied analysis script's microsecond time convention, spacing is
0.0256 microseconds (39.0625 MHz inferred) and duration about 1 millisecond,
including about 100 microseconds before time zero. Do not assign volts or
millivolts from the bare numeric amplitudes without resolving their units.

Machine-readable inspection and archive inventory are saved beside the ZIP.
Original workbooks are unchanged and remain inside the archive. No code from
the archive was installed or run. Raw outputs are locally stored, not assumed
to be included in a GitHub clone.

## Other new leads and exclusions

[Yeum et al. author repository](https://github.com/chulminy/Lamb_Wave_Mode_Decomposition)
contains separated `data/exp` and `data/fem` trees and an experimental loader
for 180 kHz dual ring/disc PZT recordings. Complete experimental alloy/bond
details were not verified, so it is not being claimed as a fuller reconstruction.

[SYPIN benchmark](https://www.sypin.agh.edu.pl/Data/SYPIN_dataset_report.pdf)
is indexed, but direct access currently returns 403. NASA's current
[repository page](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/)
says its aluminum lap-joint download is unavailable and requires contacting
staff. Neither meets verified public-access acceptance in this search.

The Cardiff [Marks dataset](https://doi.org/10.17035/d.2017.0043078392)
was located through DataCite, correcting the earlier uncertainty about whether
any data deposit exists. Its receiver is still laser vibrometry, so it does
not meet the requested PZT receiver requirement.
