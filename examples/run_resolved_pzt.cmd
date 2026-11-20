@echo off
setlocal
cd /d "%~dp0.."
title TRAXIS - experiment scheduler
set "TRAXIS_RUN=outputs\simulation-runs\run-010-wandowski-restored-ideal-properties"
if not "%~1"=="" set "TRAXIS_RUN=%~1"
echo Starting or resuming experiments in %TRAXIS_RUN%.
echo Keep this scheduler window open until it finishes.
echo A separate progress window will display and record each experiment's results.
start "TRAXIS progress" powershell.exe -NoExit -File ".\examples\watch_resolved_pzt.ps1" -RunRoot "%TRAXIS_RUN%"
".\.venv\portable-runtime\python.exe" -m examples.resolved_pzt_queue --root "%TRAXIS_RUN%" --run
if errorlevel 1 (
  echo The queue stopped with an error. Check the message above and saved logs.
) else (
  echo The scheduler finished. Results are saved in %TRAXIS_RUN%.
)
pause
endlocal
