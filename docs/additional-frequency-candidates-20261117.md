# Additional physical multi-frequency PZT experiments

The requested frequencies need not be 50/100/150/200/250/300 kHz.
The following are physical aluminum/PZT experiments, not accepted new
raw-waveform validation datasets. No simulations were started.

## Liu et al. (2019): 30 to 70 kHz

Primary source: https://www.mdpi.com/1424-8220/19/19/4166

Section 4 reports physical tests from 30 through 70 kHz in 5 kHz steps.
For a separated subset, 30/50/70 kHz would be possible if recordings become
available. Specimen: 2024-T3 aluminum, 1200 x 1200 x 2 mm; eleven PZT-5A
receivers, diameter 8 mm and thickness 0.48 mm, spaced 18 mm; three PZT
actuators with polar coordinates reported relative to the array center.
Five-cycle sine bursts, +/-70 V; 10 MHz sampling; 10,000 samples including
1,000 pretrigger samples. Published experimental wavenumbers are available
in Table 3, permitting a limited comparison of wave propagation. Raw
waveforms were not located. The later composite experiment in the same
paper must not be confused with this aluminum specimen. Adhesive mechanics
and a full PZT property tensor are not established by these dimensions.

## Chen et al. (2023): 150/250/350 kHz

Primary source:
https://www.dpi-proceedings.com/index.php/shm2023/article/download/37079/35654

Stanford AA257 class-project conference paper, "Damage Localization in an
Aluminum Plate Using Built-In Piezo-electric Sensor Induced Ultrasonic
Signals." Reports an 18 x 18 inch aluminum plate, three PZT transmitters,
three PZT receivers, nine paths, five-peak bursts and actual measurements at
150/250/350 kHz (Table 1). Sensor layout is shown. A public raw recording
download, alloy, full transducer specification and adhesive properties
were not established. A repository search for AA257 did not locate usable
class data. Institutional affiliation does not fill these metadata gaps.

## Ahmed and Kopsaftopoulos (2021): 200 to 700 kHz

Primary source: https://www.mdpi.com/1424-8220/21/16/5672

Previously screened physical multi-frequency lead: 6061 aluminum, PZT-5A
Acellent discs, Hysol EA9394 adhesive, plate and sensor geometry and
instrument details reported. Physical excitation spans 200-700 kHz; the
paper's main time-series analysis uses 250 kHz. No public raw waveform
download was located. Temper and measured bond thickness are not established.
Do not silently resolve the paper's dimensional inconsistencies by guessing.

## New special-transducer lead: 60 to 300 kHz

Primary source: https://doi.org/10.1177/14759217261456985

Marquez Reyes et al. (2026), "Directional damage detection through
unidirectional FSAT in pulse-echo mode," reports actual 60-300 kHz tests in
10 kHz steps, on a 1000 x 1000 x 1 mm aluminum plate, with 7/10/13-cycle
bursts and repeated baseline/damage recordings. The directional FSAT is a
special transducer requiring its own electrode/body representation; this
is not a standard-disc replacement. Alloy and a public raw download were
not verified. Keep as a lead, not a configured validation reference.

## Decision

Several well-separated physical frequency examples exist. None of these
has yet been verified to meet both the public raw-waveform access and
complete reconstruction requirements. Published plots or processed
wave-propagation tables support a narrower comparison than raw voltage
waveform validation. The current Wandowski queue remains unchanged.

## Follow-up: search for the three original recordings

Searched the publisher pages, paper titles and DOI identifiers, author
publication resources, GitHub repositories and code, and the public
DataCite catalog. No verified raw recording download was found for Ahmed
2021, Chen 2023 or Liu 2019. This is a search outcome, not proof that the
recordings do not exist or could never be shared.

- Ahmed: the paper's Data Availability Statement says "Not applicable."
  The author's publication entry links the paper PDF but supplies no
  active code or dataset link:
  https://github.com/me-shabbir-ahmed/me-shabbir-ahmed.github.io/blob/main/content/publication/Uncertainty-journal/index.md
  The recursive main-branch tree returned 221 entries, was not truncated,
  and contained no matching wave-data archive or numeric recording file.
  The only CSV found was an unrelated 33-byte example visualization file.
  The author paper PDF is present; it is not a recording archive.
- Stanford: the conference paper states that records were supplied to the
  students for a blind class project. No public numerical-data attachment,
  repository or class-data download was located. Public course metadata
  displays additional resources as login-only; that alone does not prove
  those resources contain this experiment's recordings.
- Liu: no supplementary raw-recording attachment or matching data deposit
  was located. The published received-signal figures and experimental
  wavenumber table remain accessible. These are not the original recordings.

Exact DOI searches and distinctive title searches in DataCite returned
zero matching records. Broader author searches yielded unrelated records
or research articles/preprints rather than a matched recording deposit.
Catalog responses are saved in
`outputs/reference-screening-20261117/three-experiment-recording-search.json`.
The generic Ahmed author search was broad and included unrelated authors;
it must not be interpreted as an exhaustive author-identity search.

No author was contacted, no guessed waveform was reconstructed from a plot,
and no simulation was launched or changed.
