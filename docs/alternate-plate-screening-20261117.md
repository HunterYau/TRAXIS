# Additional plate reference screening

Research only. No simulation, configuration change, download or author contact.
The prepared Wandowski queue remains the current reference.

## New known-alloy candidate: Humer and colleagues

[2022 experimental paper](https://www.mdpi.com/1424-8220/22/17/6403)
and [2025 reuse](https://www.mdpi.com/1424-8220/25/6/1681).

- EN AW-5754 aluminum, 500 x 500 x 2 mm; temper unverified.
- Two PIC255 actuators, diameter 10 mm, thickness 0.2 mm; opposite faces,
  x = -100 mm, inverted drive, five-cycle Hann bursts, 18 Vpp.
- Physical center frequencies: 130 and 200 kHz. Structural epoxy attachment.
- Receiver is a laser vibrometer, not a PZT voltage channel.
- Raw data: 2025 paper says available on request; no public download found.
- Glue thickness/material constants used for the bonded damage parameter study
  are assumptions, not measured PZT attachment properties.

Useful candidate for separate plate-motion/scattering validation. Not accepted
as a replacement for direct PZT-to-PZT raw-waveform comparison. Neither the
requested 50/100/150/200/250/300 kHz schedule nor a complete measured bond model
is established by this source.

## Further leads checked

### Public PZT-to-PZT alternative: Utah long-term monitoring

[Primary paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC12162875/),
[author repository and replacement download link](https://github.com/SmartDATA-Lab/Long_Term_Guided_Waves),
[current public Dropbox folder](https://www.dropbox.com/scl/fo/cigpra8cde187g7321etz/ANJMc309bWNDdNDe9NdJCQ8?dl=0&rlkey=bdv8xwkgxybo4umz4mbyc1mnr).

Aluminum plate 530 x 530 x 3 mm; STEMINC SM412 discs, 7 mm diameter,
0.2 mm thick, nominal radial resonance 300 kHz. Table 1 identifies the part
as SMD-07T02R412WL, not the exact proposed user part. Six of eight installed
PZTs provide two transmitting and four receiving channels, with coordinates
in Figure 1. Drive is a 1 ms chirp spanning 5-350 kHz, not six separate
tone-burst tests. Alloy/temper and complete bond-layer properties remain
unverified; protective epoxy must not be mistaken for a fully specified bond.

Access checked anonymously: current Dropbox download with dl=1 returned
HTTP 200, application/zip, attachment data.zip, and a valid PK ZIP header.
Only 128 bytes were read and the response was closed; the full archive was
not downloaded or validated. No contact or sign-in was needed. The author
README warns that the original dataset link is unavailable, so use its current
replacement rather than assuming the original Figshare URL still works.
This meets public PZT-recording access requirements, but does not establish
a more completely specified plate than the current reference. No queue change.

[Fan and Kopsaftopoulos multi-fidelity study](https://arxiv.org/html/2505.01666v1):
6061 plate, 152.4 x 254 x 2.36 mm; six sensors, 3.175 mm diameter x 0.2 mm;
material table and 24 MHz acquisition. No public raw recording link or explicit
test-frequency list located in the inspected text. Hole diameter is inconsistent
between prose (12 mm) and table (12.7 mm). Not accepted as a reconstruction.

[Ding et al. experimental preprint](https://doi.org/10.21203/rs.3.rs-7848563/v1)
and [public dataset](https://data.mendeley.com/datasets/9rgx3856gr/1):
real pencil-lead-break scans, including aluminum; public Data.rar listed.
Aluminum alloy and complete PZT/bond model not verified. Broadband impacts
must not be described as separate PZT tone-burst tests. Archive not inspected.

[Cai et al. paper](https://doi.org/10.34133/research.1455):
2024-T3 benchmark, 200 x 180 x 0.75 mm, 0-1 MHz chirp, 100 microseconds,
approximately 20 Vpp, laser measurements. Public raw-wavefield access and
complete actuator/bond properties not established. Candidate only.

[Time-frequency classification study](https://doi.org/10.1007/s10921-024-01101-9):
data link exists, but experiments use 1 MHz wedge-coupled contact transducers,
not bonded thin PZT discs. Its theoretical calculation uses 6061 properties;
that alone does not certify the experimental specimen's alloy/temper.

## Practical conclusion

Known-alloy studies with detailed apparatus exist, but none newly screened
combines verified plate/PZT/bond properties, public measured voltage waveforms,
and the full requested frequency set. Keep missing values explicit. Do not fill
them from an unrelated specimen or promote numerical parameter sweeps to
physical experiments. Existing earlier candidates and access gaps are recorded
in multifrequency-reference-shortlist-20261117.md and related source-search notes.
