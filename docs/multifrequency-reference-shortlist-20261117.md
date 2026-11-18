# Multiple-frequency, detailed-property references

The user specifically requested multiple physical test frequencies and extensive
reconstruction details. No simulation ran and the prepared queue was not changed.
The ranking below is for further investigation, not validation acceptance.

## Best plate/PZT metadata lead: Yang et al. (2022)

[Publisher article](https://www.mdpi.com/1424-8220/22/17/6647),
[indexed full text](https://pmc.ncbi.nlm.nih.gov/articles/PMC9459707/).

Physical experiments at **150 and 200 kHz**, with temperature variation.
2024 aluminum, 500 x 500 x 2 mm; nine PZTs, 8 mm diameter x 0.48 mm,
150 mm spacing. The reported adhesive geometry is 8 mm diameter x 0.08 mm.
The paper supplies plate, adhesive and PZT material models and temperature
dependence. It describes five-cycle Hann excitation, +/-70 V, 10 MS/s
acquisition and chamber equipment. Some property inputs come from prior
literature/manuals rather than measurements of the actual specimen.
Temper is not established here. The experimental charge-amplifier response
requires normalization for the comparisons shown in the paper.

**Access gap:** Data Availability says “Not applicable”; no raw-waveform
download was located. This is the strongest directly relevant detailed
PZT-to-PZT lead in this follow-up, but it cannot yet replace the current raw
reference. Bond thickness being specified is not proof every glue property
was measured. Thermal prestress and adhesive dynamics are also beyond the
current perfect-bond, constant-temperature body implementation.

## Three-frequency lead: Marks et al. (2016)

[Publisher article](https://www.mdpi.com/2076-3417/6/1/12),
[Cardiff author copy](https://orca.cardiff.ac.uk/id/eprint/84448/1/applsci-06-00012%20%281%29.pdf).

Physical tests at **100, 250 and 300 kHz**. Known 6082-T6 aluminum with
3 mm plate thickness and a bonded angle stiffener; panel dimensions are
drawn in Figure 2. Stiffener adhesive is Araldite 420 with thickness controlled
using 0.1 mm wire gauges. Actuator is a packaged PANCOM Pico-Z with a 5 mm
face, attached using cyanoacrylate. Ten-cycle excitation, 160 Vpp; laser
velocity measurement at 2.56 MHz and 200 averages. No raw numerical download
was established. The controlled stiffener bond is distinct from the actuator
bond; its 0.1 mm value must not be assigned to the actuator glue.

**Fit gap:** laser velocity validates motion, not PZT receiver voltage.
The stiffener, packaged actuator and attachment need additional modeling.
This is useful for separate propagation/bond validation, not a drop-in
replacement for the flat-plate PZT-to-PZT experiment.

## Broad-frequency adhesive-specific lead

[Ha et al., publisher](https://journals.sagepub.com/doi/10.1177/1475921710365267):
physical adhesive-layer studies at frequencies up to 500 kHz, with two reported
bond thicknesses (40 and 120 micrometers), and experimental/numerical comparison.
The accessible abstract does not establish the exact frequency list, complete
specimen properties or a public raw-data download. Keep as an adhesive-specific
lead, not an accepted frequency schedule.

## Screening exclusions that matter

[ONERA durability paper](https://www.mdpi.com/1424-8220/24/2/450) has detailed
PZT/glue discussion, but its 10–350 kHz tuning sweep is numerical; physical
LDV tests described are at 25 kHz. It must not be counted as measured
multi-frequency voltage ground truth. Data are available by request.

[Islam dissertation repository](https://mavmatrix.uta.edu/mechaerospace_dissertations/343/)
and the associated adhesive-thickness studies are further leads. The primary
full text was not retrieved in this follow-up; values visible only on third-party
indexes were not promoted to verified configuration inputs. Distinguish model
frequency sweeps and fitted adhesive parameters from physical tone-burst data.

## Decision

Yang is the first candidate to pursue for detailed multi-frequency PZT voltage
validation, if its recordings can be obtained. Marks is a separate three-frequency
motion-validation option. Neither is established as a complete, downloadable
replacement. No raw dataset, property profile or new executable queue was created.
No author was contacted. A useful request would seek raw drive/receiver records,
channel/time metadata, material temper/measurements, bond construction and
instrument transfer/load information. A paper figure is insufficient for the
high-precision raw-waveform comparison requested by the user.
