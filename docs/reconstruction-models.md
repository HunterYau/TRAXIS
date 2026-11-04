# Reconstruction candidates and bounded checks

The current candidates implement the two neural pipelines from the research
plan. They have passed software checks, not research training or physical
evaluation. Eight one-step debug updates were run on an analytic point-scattering
fixture. No full training cycle, epoch sweep or GPU computation was performed.

## Model A: direct waveform reconstruction

Each acquired record supplies raw current and baseline-subtracted traces to a
shared three-block 1D CNN. Eight temporal bins retain arrival-time structure.
The feature vector is combined with explicit amplitude features and context:
TX/RX coordinates and IDs, frequency, sampling interval, record start time,
and plate dimensions/thickness. Sixty-four learned spatial queries attend to
the acquired record tokens. The resulting 8 x 8 spatial seed is decoded with
bilinear upsampling and residual 2D convolutional blocks.

## Model B: DAS-assisted residual U-Net

A residual encoder/decoder with U-Net skip connections consumes a two-channel
image: residual-envelope DAS intensity and valid-arrival timing coverage.
The same waveform/context encoder supplies attention-pooled information to
the spatial bottleneck. The residual encoder trains from scratch; no pretrained
ResNet weights or ImageNet transfer performance is claimed.

Both return a depth map in mm bounded between zero and the configured nominal
thickness. `ModelConfig` supports 128/256 outputs and 0–3 extra final decoder
blocks. The current factorial debug comparison covers both architectures,
both sizes and zero/one extra block. Default width is 16, embedding size 64,
with four attention heads. These are provisional values for controlled
experiments, not selected hyperparameters.

## Missing records, units and normalization

`observation_batch` receives only public replay states. Missing waveforms and
context are masked before convolution and attention. A null token handles
the zero-observation state without an all-masked attention softmax. Tests
include changing hidden signals/context to NaN, record permutations, batches
with empty observations and zero-valued acquired records.

`SignalScale.fit` requires simulation training banks and one observable unit.
It fits a dataset RMS without inspecting labels. The same scale must be
saved with the model and reused on validation, test and physical adaptation
data in matching units. No per-record peak normalization removes relative
amplitude. The RMS feature branch preserves explicit amplitude evidence even
though the temporal CNN uses normalization layers. Changing mechanical
displacement into voltage requires a calibrated forward/measurement mapping;
the unit check refuses to treat them as interchangeable.

`reconstruction_loss` weights true damaged pixels by a provisional factor of
10. This addresses area imbalance but does not establish the best objective.
Training and validation compare each native prediction with the bank's shared
reference-cell label (normally 512 x 512). Integer upsampling repeats prediction
cells and coarsening averages them; reference labels and their damaged masks
are not reduced to the model output grid. This keeps the target and region
weights common across the 128/256 decoder ablation. Loss remains differentiable
with respect to native output cells. Parameter gradients and a one-cell-defect
example are checked analytically. Older native-label loss values are historical
debug results and must not be combined with this objective for comparisons.
The final training/evaluation harness must freeze this choice on training and
validation splits, train on varied observation subsets, and report regional
errors alongside whole-plate error.

## Reproduce the short check

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -r requirements-learning.txt
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m traxis verify-models --output outputs/model-verification
```

`verify-models` uses the default widths and exactly one SGD debug update per
configuration. It records parameter counts, finite gradients, before/after
loss, single-call inference time, source hashes and resource samples. It
enforces a cooperative 60-second / 1,024 MiB process-memory limit plus system
RAM and temperature limits. Eight updates are not a training schedule. The
timings are one CPU invocation after its debug update, not statistically
stable throughput estimates. The fixture is not an elastic or physical
dataset, and its loss values must not rank architectures.

Compact evidence is in `docs/evidence/reconstruction-debug.json`. The checked
configurations have about 103k–182k parameters. All eight checks passed. The
first monitored run used roughly eight seconds, sampled at most 423 MiB RSS
and 77.4% system memory, and observed a GPU temperature of 47 C. CPU temperature
was unavailable. No GPU arrays were allocated.

Checkpoint helpers save state dictionaries, configuration, signal scale and
plain provenance. Loading uses CPU placement and `weights_only=True`, with
strict parameter matching. Tests verify identical roundtrip predictions.
Optimizer/RNG resume, ensemble inference, frozen experiment selection and
physical adaptation roles are implemented in the experiment workflow. Research
training and measured physical adaptation have not been run.
