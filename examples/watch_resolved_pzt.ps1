# Display only. This script never starts or stops a simulation.
$projectPath = Split-Path -Parent $PSScriptRoot
$pythonPath = Join-Path $projectPath '.venv\portable-runtime\python.exe'
$queuePath = Join-Path $projectPath 'outputs\wandowski-resolved-pzt-v5'
$Host.UI.RawUI.WindowTitle = 'TRAXIS - experiment queue (text progress)'
Push-Location -LiteralPath $projectPath
try {
    & $pythonPath -m examples.pzt_terminal --root $queuePath --resources --debug --transcript (Join-Path $queuePath 'terminal-progress.txt')
} finally {
    Pop-Location
}
