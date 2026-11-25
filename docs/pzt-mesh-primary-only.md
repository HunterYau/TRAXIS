# Requested primary-only continuation

Run `pzt-mesh-convergence-003` starts at combined configuration 2 (10/48/5),
then configuration 3 (12/64/6). Configuration 1 is skipped. Each scheduled
mesh has exactly one broadband kernel and six frequency reconstructions,
50 through 300 kHz. No half-step accuracy runs are scheduled. Temporal
accuracy is reported as unchecked.

The fixed requested pair is coarse 10.68115234375 ns and fine
5.340576171875 ns. A rejected fixed step is never silently halved.
At the user's explicit request, the extra 20% fine-step margin is removed
in opt-in copies `traxis/gpu_local_time_requested.py` and
`traxis/lts_stability_requested.py`. The strict full-system fine-step bound,
effective-operator screen, FP64 calculations and energy check remain.
Configuration 3 may therefore be reported unsuccessful if it fails the
remaining checks. Passing the screen is not a temporal convergence result.

The original five protected solver/operator files retain their recorded
SHA-256 hashes. An AST comparison confirmed the copied `gpu_lts_simulate`
function is unchanged; only the fine-step admission guard differs in the
copied solver module. Original solver and screening implementations remain
available as fallbacks.

The interrupted run 002 was downloaded and verified at
`E:\TRAXIS Project Outputs\PZT Mesh Convergence\Combined Only\pzt-mesh-20261125T082317410031Z`.
Run 003 downloads to
`E:\TRAXIS Project Outputs\PZT Mesh Convergence\Primary Only` while this PC
is awake and connected. The pod remains running after completion.
