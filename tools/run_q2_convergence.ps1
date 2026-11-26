param([string]$OutputDirectory = '', [ValidateSet('cpu','gpu')][string]$Backend = 'cpu')
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path $PSScriptRoot -Parent
$repositoryRoot = Split-Path (Split-Path $projectRoot -Parent) -Parent
$pythonPath = Join-Path $repositoryRoot '.venv/portable-runtime/python.exe'
$runOutput = if ($OutputDirectory) { $OutputDirectory } else { Join-Path $projectRoot ('outputs/q2-convergence-' + (Get-Date -Format 'yyyyMMddTHHmmss')) }
New-Item -ItemType Directory -Path $runOutput -Force | Out-Null
$logSuffix = if ($OutputDirectory) { '-resume-' + (Get-Date -Format 'yyyyMMddTHHmmss') } else { '' }
$progressLog = Join-Path $runOutput ('progress' + $logSuffix + '.txt')
$errorLog = Join-Path $runOutput ('errors' + $logSuffix + '.txt')
$pythonArguments = '-u -c "import sys,runpy; sys.path.insert(0,''.''); runpy.run_module(''examples.spectral_pzt_convergence_schedule'',run_name=''__main__'')" --output "' + $runOutput + '" --backend ' + $Backend + ' --local-cuda-root "' + $repositoryRoot + '"'
$worker = Start-Process -FilePath $pythonPath -ArgumentList $pythonArguments -WorkingDirectory $projectRoot -WindowStyle Hidden -RedirectStandardOutput $progressLog -RedirectStandardError $errorLog -PassThru
@{ pid = $worker.Id; output = $runOutput; progress_log = $progressLog; error_log = $errorLog } | ConvertTo-Json | Set-Content (Join-Path $runOutput 'launcher.json')
$consoleCommand = '$host.UI.RawUI.WindowTitle = ''Q2 convergence - 6 and 8 rings''; Write-Host ''Q2 convergence: 6 rings then 8 rings. Results save automatically.''; Get-Content -LiteralPath ''' + $progressLog + ''' -Wait'
$console = Start-Process powershell.exe -ArgumentList @('-NoProfile','-NoExit','-Command',$consoleCommand) -WorkingDirectory $projectRoot -WindowStyle Normal -PassThru
Write-Output ('Worker PID: ' + $worker.Id)
Write-Output ('Console PID: ' + $console.Id)
Write-Output ('Results: ' + $runOutput)
