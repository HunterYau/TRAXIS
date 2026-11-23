param([string]$RunRoot = 'outputs/pzt-development/full-plate-convergence-20261121')
Set-Location (Split-Path $PSScriptRoot -Parent)
$Host.UI.RawUI.WindowTitle = 'TRAXIS - full plate convergence audit'
& '.\.venv\portable-runtime\python.exe' -u -m examples.pzt_convergence --root $RunRoot --run 2>&1 |
    Tee-Object -FilePath (Join-Path $RunRoot 'console.log') -Append
Write-Host 'Audit stopped or completed. See findings.txt and status.json in the run folder.'
