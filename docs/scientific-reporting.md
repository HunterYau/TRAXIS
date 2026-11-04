# Scientific figures from frozen replay

`report` reads completed, fingerprinted evaluation results. It performs no
training, threshold fitting, model selection or reranking of cases. Install
the optional plotting package inside the project virtual environment:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-plots.txt
.\.venv\Scripts\python.exe -m traxis report --evaluations outputs/evaluation --labels selected-model --output outputs/figures
```

The output directory must be new or empty. Open `index.html` to review the
PNG figures and download their standalone SVG versions and CSV tables.
Matplotlib runs with an Agg canvas, and its initial font cache is placed
inside the report directory. Reading and plotting completed results needs
the core and plotting dependencies; it does not import PyTorch.

## Recorded evidence and matching

New evaluations save schema 2: plate dimensions, signal/data roles, sensor
coordinates, record catalog size, predictor configuration, parameter counts,
artifact hashes, execution platform, CPU thread count and per-budget timing.
Each map archive has a checksum bound into the fingerprinted evaluation
report. The report command checks those hashes for selected map cases and
recomputes their errors to detect metrics from a different saved prediction.
It also rechecks the aggregate summary against the episode rows.

Older evaluations lack this information. Freeze a new plan and rerun their
replay with the existing selected checkpoints to obtain schema 2; training
does not need to be repeated. Inference-source changes invalidate a frozen
plan, so preserve the recorded source revision or freeze a new plan explicitly.

For several evaluation directories, pass one distinct label per run:

```powershell
.\.venv\Scripts\python.exe -m traxis report --evaluations outputs/architecture-test outputs/ensemble-test --labels ABC ensemble-A --cases 0:9 1:3 --case-records 10 40 --output outputs/comparison-figures
```

Combined runs must have the same dataset fingerprint, held-out split,
requested budgets, initial actions and fixed defect threshold. Fixed/random
observations must match across predictors for the same specimen, policy and
budget. This permits controlled architecture and sensing comparisons; it
does not establish that the training schedules were matched. Check each
training configuration before interpreting an architectural advantage.

`--cases run:episode ...` selects exact zero-based run and saved episode
indices. The example selects episode 9 from run 0 and episode 3 from run 1;
those indices and budgets must exist. Otherwise, `--case-count` uses the first
one episode per run by default (0–4 permitted). Default map budgets are the
first saved budget and the dense reference. Use `--case-count 0` for curves
and tables only. Plan display cases before examining final holdout errors;
the tool never chooses cases by prediction quality.

## Figure interpretation

The curve figure shows whole-plate MAE, defect-region MAE, footprint Dice,
and absolute MAE difference from each model's dense prediction. A dense
prediction is a model reference, not defect truth. Negative excess error is
retained. Relative error is undefined for near-zero dense error; its CSV
cell stays empty. Defect-region error is undefined on intact controls and
does not become zero.

Random seeds average within scans, then scans average within parent plates,
then independent parents receive equal weight. Whiskers show between-parent
sample standard deviation when at least two parents have a defined metric;
they are not confidence intervals. CSVs give the actual contributing-parent
count for each metric. Different RL artifacts remain separate series, as do
different uncertainty-path sample counts. Random seeds remain pooled repeats.

2D maps use physical x/y axes in millimeters, lower-left origin and nearest
cell display. Known/predicted loss depth and absolute error use fixed scales
from zero to nominal plate thickness. The estimate remains at its native
resolution; upsampling for comparison does not create finer information.
Remaining thickness is nominal thickness minus loss depth. Ensemble standard
deviation is shown separately and labeled **uncalibrated**; its upper limit
is at least nominal thickness and expands only if spread exceeds that bound.

2.5D surfaces area-average to at most 64 cells per axis to bound rendering
cost. Their vertical scale is exaggerated and their cell-center surfaces
connect adjacent displayed samples. Use the 2D maps and native arrays for
footprint edges and quantitative analysis. Plotting uses Matplotlib's
[explicit image origin/extent](https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.imshow.html)
and [3D surface API](https://matplotlib.org/stable/api/toolkits/mplot3d/axes3d.html).

`summary.csv` and `per-parent.csv` retain aggregate and independent-plate
metrics. `computation.csv` records neural parameter count, device, measured
inference time at each saved budget, cumulative calls/time, selector time
and unique transmitter/frequency request counts. Inference timing includes
input preparation, DAS when applicable, and all ensemble members. These are
single replay measurements without a benchmark warm-up, not controlled
latency measurements. No hardware acquisition duration is inferred. Debug
models/fixtures retain conspicuous debug labels in figures and the index.

The command guards wall time, process/system memory and available telemetry
between render operations; it cannot preempt a blocked native render. Reports
are limited to eight source runs, 24 predictor/policy curves, eight explicitly
selected case pairs, and eight map budgets per case. Hashes of exported files,
source results and plotting code are written to `report.json` with resource
samples. No physical accuracy follows from successful report rendering.
