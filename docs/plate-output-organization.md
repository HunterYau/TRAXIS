# Simulation run outputs

The rest of `outputs/` is grouped by purpose: `solver-development/`,
`gpu-and-performance/`, `pzt-development/`, `experimental-references/`,
`result-analyses/`, `test-infrastructure/`, `science-log/`, and `supporting-files/`.
Each original folder keeps its name and contents inside its group. Solver studies
also have subfolders by method, such as `solver-development/embedded/`.
`outputs/README.txt` lists every moved entry; `outputs/organization.json` records
the path mapping and verified file inventories. The current simulation resolves
its historical reference-data path through this map without modifying the saved
experiment receipts. Older development tools with hardcoded output paths can use
the index to locate their archived results.

## Where future checks write

Most development commands require an explicit `--output`; they do not choose a
folder automatically. Use a path inside the relevant group, for example:

```powershell
.\.venv\portable-runtime\python.exe -m examples.embedded_wave --output outputs/solver-development/embedded/my-next-check --help
```

The example above shows help only. Keep passing all inputs and run options required
by the selected check. Any explicit output path is honored, including custom
locations. Worker outputs remain within the parent check's chosen folder.

Commands with automatic locations now use the groups directly: PZT readiness
records/cache use `pzt-development/`; saved-strength analyses use `result-analyses/`;
the CLI verification pilots use `test-infrastructure/`. Simulation queues continue
to use their selected folder in `simulation-runs/`.

Two PZT evidence scripts changed only output directory strings. The original
numerical readiness record remains preserved. The active inherited record is
`docs/evidence/resolved-pzt-organized-output-readiness.json`, with a separately
hashed AST comparison receipt proving those edits do not change computations.
No numerical/CUDA simulation was rerun for this path update.

To preview grouping for newly created development output folders, run
`python -m examples.organize_outputs`; use `--apply` to group them. Existing
groups and simulation run folders are retained. This command does not simulate.

Each complete experiment/simulation run has its own folder directly under
`outputs/simulation-runs/`, including repeat runs on the same plate.
The selected next experiment is `run-010-wandowski-restored-ideal-properties/`,
prepared but not started; see [restored settings](restored-ideal-experiment.md).
The latest completed experiment is `run-009-wandowski-realistic-bond-damping/`:

```text
run-009-wandowski-realistic-bond-damping/
  all-results.txt           all frequency results, times and uncertainty labels
  all-results.json          combined machine-readable comparisons
  summaries/               one text summary per frequency
  50khz/                   result.json, comparison.json, waveforms.npz,
                           final-state.npz, worker.log, launcher.json, status.json
  100khz/ ... 250khz/       same structure
  logs/                    scheduler and terminal transcripts
  provenance/              original snapshots, receipts and monitoring records
  cache/                   disposable GPU compilation cache
  prepared.json            frozen source/configuration/reference receipt
```

Older models, partial runs and preparations each have their own folder alongside
the current run; short numerical setup checks are under `setup-checks/`. The earlier
50/100 kHz ideal-bond results are in run 006 and its 150/200/250 kHz continuation
is run 007. Model versions are not merged into one scientific comparison.
Older archived frequency folders retain their original `native/` subfolder.
New runs use the flat frequency structure shown above. Each prepared run also
has a combined text report and a `summaries/` folder.

The `simulation-runs/README.txt` lists the old and new names. Move inventories and saved
result/comparison/waveform hashes are in `outputs/simulation-runs/relocations.json`.
Original stored receipts, logs and JSON results retain their historical paths;
the maintained Python analysis tools resolve the moved run locations.

## Starting and resuming

For the current experiment, use `examples/run_resolved_pzt.cmd`, or:

```powershell
.\.venv\portable-runtime\python.exe -m examples.resolved_pzt_queue --run
```

The scheduler verifies completed frequencies and skips them. It retries the first
unfinished frequency, then proceeds serially. Incomplete output remains in the
same experiment folder, under `<frequency>khz/previous-attempts/attempt-001-unfinished/`.
Old logs, partial data, failure markers and result files are preserved there.
The retried frequency starts from the beginning; there are no mid-test integration
checkpoints. `INTERRUPTED.txt` or `FAILED.txt` makes an unfinished attempt visible.
The combined report updates after each experiment and on normal interruption/error.

If Windows or the process terminates abruptly, the next export or display detects
that the scheduler and worker are gone and reports `INTERRUPTED / UNFINISHED`.
Export also saves that status marker. The OS queue lock releases automatically
on process exit. A worker that outlives its scheduler prevents a duplicate launch.
Remove `STOP_AFTER_CURRENT` before resuming if you intentionally used that marker.

The five tests in run 009 are already complete. Run 010 is a separate batch with
restored ideal properties and has not started. Changed source, configuration or input data require a new
experiment folder, preventing mixed model versions. To prepare a new batch with
the selected configuration (preparation does not simulate):

```powershell
.\.venv\portable-runtime\python.exe -m examples.resolved_pzt_queue --root outputs/simulation-runs/run-010-next-comparison --prepare
```

Then pass that same folder to `--run`, or as the first argument to
`examples/run_resolved_pzt.cmd`. Reuse that folder for subsequent resumes.
For another experiment, choose another run folder under `simulation-runs/`.
For another physical plate/dataset, also select the matching simulation
configuration; a folder name alone does not change the physics.

View progress or regenerate the collective text without launching simulations:

```powershell
.\.venv\portable-runtime\python.exe -m examples.pzt_terminal --resources --debug
.\.venv\portable-runtime\python.exe -m examples.pzt_terminal --export-results
```

Use `--root <experiment-folder>` with either command for another run. Outputs
remain local and ignored by Git; back up these folders separately to transfer
recordings and results to another computer.
