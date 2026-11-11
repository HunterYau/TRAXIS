# Progress publication on Windows

The Q5 full-plate simulation completed, but its optional cooling monitor
stopped when Windows denied replacement of `progress.json`. The simulation's
separate resource guards continued. An unchanged monitor attached to the same
PID and creation identity and completed a second segment. Both the failed
segment and recovery remain recorded; their separate clocks do not establish
continuous sidecar coverage. See [the wave record](tensor-wave-windows.md).

## Publication and observation

`storage.atomic_write` now retries Windows permission errors 5, 32 and 33
when replacing an existing file. Eight backoff delays total 0.455 seconds,
with at most nine replacement attempts. The completed temporary file is
reused and the writer callback runs once. A persistent denial still raises,
preserving the old target and cleaning up the temporary file. Unrelated
errors and create-only hard-link publication retain their previous behavior.

`examples.read_live_progress` makes brief bounded JSON observations. On
Windows it explicitly requests read/write/delete sharing and owns the native
handle through a non-inheritable binary descriptor. Missing, malformed or
oversized observations are errors, not proof that a worker has finished.

```powershell
.\.venv\Scripts\python.exe -m examples.read_live_progress outputs/run/progress.json
```

Sharing flags alone did not pass an actual held-open replacement test on this
machine. The writer's bounded retry handles brief overlap; neither mechanism
guarantees replacement while another process holds an indefinite lock.
The failed reader-only test and its source snapshots remain in
`outputs/live-progress-tests-20261110`. An unsigned PowerShell prototype was
blocked by the existing script policy and removed; no policy bypass or
machine setting change was used.

## Verification

The seven reader/storage tests pass without skips in 0.079 test seconds.
The final 13-test integration group passes without skips in 8.677 test /
10.344 monitored seconds. It includes the existing experiment workflow and
tensor protocol checks. The experiment fixture performs two one-step CPU
debug updates, checkpoint/resume and bounded calibration/evaluation/replay;
no full or research training cycle ran. These overlapping groups do not
increase the last complete integrated count of 268 tests.

A fresh 40-by-30-by-3 mm GPU control completes 32,768 steps over 500
microseconds with 1,215 coefficients. Only `storage.py` changes in the bound
core inventory. Mass, disk coefficients and time grids match the original
control exactly. Receiver repeat differences are at most 1.27e-13 relative;
final displacement/velocity differences are 7.20e-14 / 6.36e-13. Independent
CPU mass, energies and probes agree within 2.53e-16. Propagation takes
16.286 seconds; RSS peaks at 422.72 MiB and GPU observations at 994 MiB /
45 C. CPU temperature is unavailable. This is a small workflow check, not
a full-size rerun or physics/data acceptance.

After this source change, both historical full-plate assessments reproduce
their exact original comparison arrays using the archived numerical source.
Replay checks inventory, paths and source hashes; it does not override
identity checks or rerun forward physics. The spatial/time replay monitors
complete in 7.821 / 9.864 seconds.

Raw test, smoke and replay directories are indexed in
`evidence/progress-log-recovery-debug.json`, fingerprint
`2d5b3f66bd2031a1eec5b8ca66033548f6b195f75b29eba6c53af351e5362f3e`.
The receipt preserves the failed first idea, source identities, successful
checks and resource observations. No research bank is accepted by this fix.
