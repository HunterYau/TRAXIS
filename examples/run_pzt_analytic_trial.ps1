param([string]$RunRoot = 'outputs/pzt-development/analytic-drive-ab-20261122')
Set-Location (Split-Path $PSScriptRoot -Parent)
$Host.UI.RawUI.WindowTitle = 'TRAXIS - two analytic pulse tests'
& '.\.venv\portable-runtime\python.exe' -u -m examples.pzt_analytic_trial --root $RunRoot --run 2>&1 |
    Tee-Object -FilePath (Join-Path $RunRoot 'console.log') -Append
Write-Host 'Scheduler finished or stopped. Results: ' (Join-Path $RunRoot 'findings.txt')
