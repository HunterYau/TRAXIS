# New public multi-frequency aluminum/PZT recordings: Cano

## Finding

Author repository: https://github.com/WagnerCano/LambWave

The README identifies the data as supporting chapter 4 of the author's
master's thesis and the conference paper "Autoregressive Models for Damage
Prognosis in Smart Structures", DOI 10.20906/CPS/CON-2016-0209.
It explicitly lists physical excitation frequencies 100, 150, 200, 250 and
300 kHz, five cycles, with only path A3-B3 supplied. These are separate
frequency-labelled recordings, not slices presented as independent tests
from one chirp.

This is a newly verified public multi-frequency recording source. It is
not yet a fully specified replacement for the current validation experiment.

## Download verification

Downloaded anonymously, without author contact:
https://raw.githubusercontent.com/WagnerCano/LambWave/master/PL0402_Ref.zip

- 22,463,441 bytes.
- SHA256: 7266cfd153b4dcf9b62aab8f5bdbc6068fe100f4211a8253ffeceacf45fc8af9
- ZIP CRC check passed for all entries.
- 150 CSV recordings: 30 for each of the five frequencies.
- Names encode reference repeat, measurement number, frequency and B3_A3 path.
- Each inspected frequency example has 15,000 rows and three numeric columns.
- First column spans 0 to 0.0014999 seconds with 0.1 microsecond spacing,
  corresponding to 10 MHz sampling and a nominal 1.5 ms record.
- Inspected examples contain finite values. The two voltage-like columns
  have an early large burst and a later smaller response, respectively;
  exact channel labels and calibration must be verified before use.

Saved archive, inventory and per-frequency inspection:
`outputs/reference-screening-20261117/cano2016/`.

The repository also lists seven damaged-state archives (Dn1-Dn7); these
were not downloaded. Do not assign damage depths from the paper's shorter
four-damage-condition example to all seven repository conditions.

## Primary experiment descriptions inspected

Conference publisher index:
https://www.abcm.org.br/anais/conem/2016/

The index's entry and PDF link-generation code identify the paper URL:
https://www.abcm.org.br/anais/conem/2016/PDF/CON-2016-0209.pdf

Retrieved PDF: 1,838,728 bytes, saved as `conference-paper.pdf`.
This public publisher copy works despite the DOI's original SWGE host and
the newer Conferentia host failing to connect during this check.

Author's thesis, from the UNESP institutional repository:
https://repositorio.unesp.br/server/api/core/bitstreams/b2050c7e-b999-4b8a-a94f-c7c7798b8011/content

Retrieved PDF: 1,093,508 bytes, 69 pages, saved as `thesis.pdf`.

Section 2 of the conference paper and section 4.1 of the thesis describe a
1.6 mm aluminum plate with six PZT patches, room temperature 23 +/-3 C,
clamped plate ends, NI PXI/LabVIEW acquisition, 10 MHz sampling and 1.5 ms
recording. The paper's analyzed example is a five-peak Gaussian-windowed
150 kHz burst at 10 Vpp, with PZT5 as actuator and PZT2 as receiver.
The thesis photograph caption identifies Piezo Systems as sensor supplier.

## Outstanding reconstruction gaps

- Exact aluminum alloy/temper was not established.
- Exact PZT model, dimensions and full material properties were not established.
- Glue type, thickness and mechanical properties were not established.
- Exact plate in-plane dimensions, all sensor coordinates and support mechanics
  require verification; photographs are not precision coordinate tables.
- The repository path is B3_A3, while the paper example is numbered 5-to-2.
  Do not equate these paths without a mapping.
- The archive has three reference repeats while the paper describes two sets
  of ten signals for its example. Do not assume identical acquisition campaigns.
- Paper results use normalized response signals. Archive examples are not
  unit-variance signals, but raw instrument calibration still needs confirmation.

It meets the aluminum, actual PZT measurement, separate multiple-frequency,
and anonymous recording-access criteria. It does not yet meet the requested
high-fidelity reconstruction criterion, nor prove a metadata improvement over
Wandowski. No simulator settings, scheduler or simulations were changed.

## Other newly checked leads

- jbharley/WISP contains public physical aluminum/PZT MAT recordings.
  A 17,132,453-byte baseline file was downloaded and inspected (10 MHz,
  10,000 points, numerical Tx/Rx coordinates, drive and response arrays).
  This is a broadband chirp reference, not a verified set of independent
  tone-burst frequencies; alloy and attachment properties remain unresolved.
- dibgerge/gwshm-machine-learning contains 40/60/80/100/120 kHz HDF5
  recordings, but its primary paper describes a glass-fiber composite plate.
  It must not be presented as an aluminum experiment.
- SYPIN's official datasets page explicitly requires direct contact to obtain
  raw signals. Its public processed database is not a substitute for those.
- The IEEE focusing-guided-wave dataset page requires a subscription/login
  and a reviewer password; no access restriction was bypassed.

Expanded DataCite catalog search responses were saved separately as
`outputs/reference-screening-20261117/expanded-plate-catalog.json`.
