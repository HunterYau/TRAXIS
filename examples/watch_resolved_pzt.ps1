# Display only. This script never starts or stops a simulation.
param([string]$RunRoot = 'outputs\simulation-runs\run-010-wandowski-restored-ideal-properties')
$projectPath = Split-Path -Parent $PSScriptRoot
$pythonPath = Join-Path $projectPath '.venv\portable-runtime\python.exe'
$queuePath = if ([IO.Path]::IsPathRooted($RunRoot)) { $RunRoot } else { Join-Path $projectPath $RunRoot }
$Host.UI.RawUI.WindowTitle = 'TRAXIS - experiment queue (text progress)'
Push-Location -LiteralPath $projectPath
try {
    New-Item -ItemType Directory -Force -Path (Join-Path $queuePath 'logs') | Out-Null
    & $pythonPath -m examples.pzt_terminal --root $queuePath --resources --debug --follow-retries --transcript (Join-Path $queuePath 'logs\terminal-progress.txt')
} finally {
    Pop-Location
}
