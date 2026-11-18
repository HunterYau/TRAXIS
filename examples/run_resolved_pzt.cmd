@echo off
setlocal
cd /d "%~dp0.."
title TRAXIS - experiment scheduler
".\.venv\portable-runtime\python.exe" -c "from examples.resolved_pzt_queue import preflight; preflight()"
if errorlevel 1 (
  echo Preparation check failed. No experiment started.
  pause
  exit /b 1
)
echo Starting the prepared five-frequency exploratory comparison.
echo Keep this scheduler window open until it finishes.
echo A separate progress window will display and record each experiment's results.
start "TRAXIS progress" powershell.exe -NoExit -Command "& '.\.venv\portable-runtime\python.exe' -m examples.pzt_terminal --root outputs/wandowski-resolved-pzt-v5 --resources --debug --transcript outputs/wandowski-resolved-pzt-v5/terminal-progress.txt"
".\.venv\portable-runtime\python.exe" -m examples.resolved_pzt_queue --run
if errorlevel 1 (
  echo The queue stopped with an error. Check the message above and saved logs.
) else (
  echo The scheduler finished. Completed results are saved in outputs\wandowski-resolved-pzt-v5.
)
pause
endlocal
