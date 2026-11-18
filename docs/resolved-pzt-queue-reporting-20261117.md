# Prepared serial queue and automatic reports

The latest instruction is **prepare only; do not run any simulations**.
No physics simulation, including a short GPU check, was run for this update.
Thirteen scheduler, comparison and display tests passed using synthetic
signals, mocked workers and temporary test files.

The prepared order is **50 -> 100 -> 150 -> 200 -> 250 kHz**, using the
existing Wandowski plate-1 PZT measurements for a controlled before/after
comparison. 300 kHz has no matching recording on this specimen and is omitted.
No more complete replacement experiment was established by the source search.

The current frozen receipt is
`outputs/wandowski-resolved-pzt-v3/prepared.json`. Earlier v1/v2 receipts and
the completed old runs are preserved. The `launch_ready=false` gate remains:
the PZT-body mesh sensitivity near resonance still needs resolution before
a future long run. Preparing this scheduler does not establish that the
new body model has passed that check.

## Per-experiment sequence

1. Run one worker and write its progress/debug events to `worker.log`.
2. Save `native/waveforms.npz` and `native/final-state.npz` as soon as time
   integration finishes, before analysis. If later analysis fails, these survive.
3. Check final forces and available unforced energy samples; calculate the
   comparison against the corresponding measured PZT recording.
4. Save `native/comparison.json`, `native/result.json`, and `native/summary.txt`.
   Print the human-readable results into the worker and queue text logs.
5. Verify completion, the comparison/text files and waveform/state checksums,
   then start the next frequency. A failure stops the queue.

After all five, save `all-results.json` and print the completion message.
Each experiment also has a launcher receipt. Files are not overwritten.
There is no intermediate-state resume mechanism; `STOP_AFTER_CURRENT`
stops between experiments and preserves completed output.

## Display and persistent text

`examples/watch_resolved_pzt.ps1` opens no simulation; it runs only the viewer.
The viewer reports the queue order, current frequency/test, completed and total
steps, percentage, elapsed time, approximate ETA, current phase, worker-log
location, machine RAM and GPU utilization/device memory/temperature. Displayed
progress and summaries are appended as ordinary lines to `terminal-progress.txt`.
Raw worker events remain in `worker.log`; overall order/results are in `queue.log`.
Before any launch it clearly says **NOT STARTED** and lists the mesh hold.

Closing the viewer does not stop an independently launched queue. The scheduler
and worker text files still record results without the viewer. This is a
text progress/debug-log screen, not a breakpoint debugger.

## Meaning of the comparisons

The previously used percentages remain the primary numbers: absolute
fast/slow **flight-time error**, where lower is better. The same paired
top+bottom / top-bottom procedure and predeclared timing windows are used.
Uncertain arrivals are printed as unresolved, not as zero error. The published
PZT slow-mode detection limit remains respected at the higher frequencies.

A separate whole-record **envelope-shape difference** is now reported for
each sum/difference trace. Both envelopes are individually normalized to
unit L2 norm, then their L2 difference is multiplied by 100. This compares
shape on the common 0–349.8 microsecond grid, including reflections. No time
shift or polarity fit is performed. It deliberately removes overall amplitude
scale because the drive/instrument gain is uncalibrated. It is not an overall
accuracy percentage, and it does not establish that a weak residual is a
detected slow wave. A zero signal yields unresolved output.

No new experimental percentages exist yet; the numbers will be calculated
and printed only after future simulations finish. Raw run files stay local
under ignored `outputs/`; GitHub contains the code and compact evidence.
