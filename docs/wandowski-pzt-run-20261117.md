# PZT experiment queue launched November 17, 2026

The user authorized launching the prepared five-frequency queue and asked
Codex to observe only its first 15 minutes. After that observation, Codex
must end its turn while the independent program continues through all five
frequencies. The user explicitly clarified that the program must not be
stopped when Codex leaves. The older broad research goal remains paused.

Started at 2026-11-17 16:06:51 UTC (08:06:51 Pacific). Queue PID 4340;
launch metadata is `outputs/wandowski-pzt-plate1/launch-20261117.json`.
The monitoring window ends at approximately 16:21:52 UTC.

The frozen preparation matched all current source/configuration/data hashes
before launch. The source-bound driver was not changed. Five frequencies
are scheduled serially on the same literature plate: 50, 100, 150, 200 and
250 kHz. There is no 300 kHz run or training job.

The independent queue was launched using the portable Python runtime,
`-u -m examples.wandowski_pzt_queue --run`, with logs redirected to
`queue.stdout.log` and `queue.stderr.log` in the run directory. Each worker
has its own `FREQUENCYkhz/worker.log`; finished raw traces, final state and
results go under `FREQUENCYkhz/native/`. `queue-owner.json` prevents a
duplicate launch. Do not delete it or restart the queue while it is active.

A read-only Windows progress display is provided by
`examples/watch_wandowski_pzt.ps1`. Its title is
**TRAXIS | Live Plate Simulation**. It shows the plate, frequency/test number,
percentage, completed/requested steps, elapsed time and saved test count.
Closing the display does not stop the calculation. The display refreshes
every two seconds from worker events printed approximately every 15 seconds.
Its initial hidden-window launch was replaced with a visible-window launch;
only the viewer was restarted. The simulation was never restarted. The
viewer retains the last step count through automatic cooling events.

Initial checks: preflight passed, the first worker reached propagation, GPU
utilization reached 100%, and steps increased continuously without a reported
exception. At approximately six minutes, 8992/65536 steps (13.7%) were done,
system RAM was 72.9%, GPU allocation about 3974 MiB and GPU temperature 71 C.
These are operational observations, not experimental accuracy results.

The standalone queue retains its existing resource/temperature guards and
stops on a worker failure. The computer must remain awake for it to finish.
Results save after each frequency; the middle of a frequency is not a
restartable checkpoint. Closing Codex does not terminate this independent
process. End-of-monitor observations are saved in
`outputs/wandowski-pzt-plate1/codex-monitor-20261117.jsonl` and the final
handoff receipt. Physics limitations remain as documented in
[the prepared protocol](wandowski-pzt-prepared-queue.md).
