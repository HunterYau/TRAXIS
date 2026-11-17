# Read-only live display. Closing this window does not stop the simulations.
param([string]$RunRoot = (Join-Path $PSScriptRoot '..\outputs\wandowski-pzt-plate1'))
Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing
$RunRoot = [IO.Path]::GetFullPath($RunRoot)
$launch = Get-Content -LiteralPath (Join-Path $RunRoot 'launch-20261117.json') -Raw | ConvertFrom-Json
$queueStart = [DateTime]::Parse($launch.started_utc).ToUniversalTime()
$frequencies = @(50,100,150,200,250)

$form = New-Object Windows.Forms.Form
$form.Text = 'TRAXIS | Live Plate Simulation'
$form.ClientSize = New-Object Drawing.Size(850,540)
$form.StartPosition = 'CenterScreen'
$form.BackColor = [Drawing.Color]::FromArgb(20,26,37)
$form.ForeColor = [Drawing.Color]::WhiteSmoke
$form.Font = New-Object Drawing.Font('Segoe UI',12)
function Label([int]$top,[int]$height,[int]$size) {
    $item = New-Object Windows.Forms.Label
    $item.Location = New-Object Drawing.Point(26,$top)
    $item.Size = New-Object Drawing.Size(798,$height)
    $item.Font = New-Object Drawing.Font('Segoe UI',$size)
    $form.Controls.Add($item)
    return $item
}
$title = Label 20 40 22
$title.Text = 'Plate 1 | Published aluminum / NCE51 PZT'
$subtitle = Label 67 30 12
$subtitle.Text = 'Same plate, five frequencies: 50, 100, 150, 200, 250 kHz'
$subtitle.ForeColor = [Drawing.Color]::LightSteelBlue
$case = Label 116 36 18
$bar = New-Object Windows.Forms.ProgressBar
$bar.Location = New-Object Drawing.Point(26,164)
$bar.Size = New-Object Drawing.Size(798,28)
$form.Controls.Add($bar)
$stepsLabel = Label 205 32 15
$timeLabel = Label 249 30 13
$statusLabel = Label 293 48 12
$statusLabel.ForeColor = [Drawing.Color]::LightGreen
$savedLabel = Label 353 28 12
$footer = Label 398 48 11
$footer.Text = 'Closing this display does not stop the simulation. Results save automatically.'
$footer.ForeColor = [Drawing.Color]::LightSteelBlue
$pathLabel = Label 453 70 10
$pathLabel.Text = "Logs: $RunRoot"
$pathLabel.ForeColor = [Drawing.Color]::LightSteelBlue
$timer = New-Object Windows.Forms.Timer
$timer.Interval = 2000
$timer.Add_Tick({
    try {
        $completed = 0; $current = $null; $currentNumber = 0
        foreach ($khz in $frequencies) {
            $directory = Join-Path $RunRoot ($khz.ToString() + 'khz')
            $resultPath = Join-Path $directory 'native\result.json'
            if (Test-Path -LiteralPath $resultPath) {
                $result = Get-Content -LiteralPath $resultPath -Raw | ConvertFrom-Json
                if ($result.completed) { $completed++; continue }
            }
            if ($null -eq $current) { $current = $directory; $currentNumber = $khz }
        }
        $queueElapsed = [DateTime]::UtcNow - $queueStart
        $case.Text = "Test $([Math]::Min($completed+1,5)) / 5 | $currentNumber kHz"
        $savedLabel.Text = "Completed and saved: $completed / 5 frequencies"
        $statusLabel.ForeColor = [Drawing.Color]::LightGreen
        if ($completed -eq 5) {
            $case.Text = 'All five frequencies completed'
            $bar.Value = 100; $stepsLabel.Text = '100% | All results saved'
            $statusLabel.Text = 'Queue finished.'
            $timeLabel.Text = "Total elapsed: $($queueElapsed.ToString('hh\:mm\:ss'))"
            return
        }
        $logPath = Join-Path $current 'worker.log'
        $lines = @()
        if (Test-Path -LiteralPath $logPath) { $lines = @(Get-Content -LiteralPath $logPath -Tail 15) }
        $event = $null; $lastProgress = $null
        foreach ($line in $lines) {
            if ($line.StartsWith('{')) {
                try {
                    $event = $line | ConvertFrom-Json
                    if ($event.phase -eq 'propagating') { $lastProgress = $event }
                } catch {}
            }
        }
        $statusLabel.Text = 'Initializing GPU operator and PZT ports...'
        $percent = 0; $done = 0; $total = 65536
        $phaseElapsed = 0
        if ($null -ne $event) {
            $statusLabel.Text = "Status: $($event.phase) | Live update every 15 seconds"
        }
        if ($null -ne $lastProgress) {
            $phaseElapsed = [double]$lastProgress.elapsed_seconds
            $done = [int]$lastProgress.steps_completed; $total = [int]$lastProgress.requested_steps
            $percent = 100.0*$done/$total
        }
        $bar.Value = [Math]::Min(100,[Math]::Max(0,[int]$percent))
        $stepsLabel.Text = ('{0:N1}% complete | Steps: {1:N0} / {2:N0}' -f $percent,$done,$total)
        $timeLabel.Text = "Test elapsed: $([TimeSpan]::FromSeconds($phaseElapsed).ToString('hh\:mm\:ss')) | Queue elapsed: $($queueElapsed.ToString('hh\:mm\:ss'))"
        $logAge = 0
        if (Test-Path -LiteralPath $logPath) { $logAge = ([DateTime]::UtcNow - (Get-Item -LiteralPath $logPath).LastWriteTimeUtc).TotalSeconds }
        if ($logAge -gt 60) {
            $statusLabel.Text = "Last worker update $([int]$logAge) seconds ago; may be initializing, cooling or checking results."
            $statusLabel.ForeColor = [Drawing.Color]::Khaki
        }
        if ($lines -match 'Traceback|Error:') {
            $statusLabel.Text = 'Worker reported an error. Inspect worker.log; queue stops on failure.'
            $statusLabel.ForeColor = [Drawing.Color]::Salmon
        }
        if ($null -eq (Get-Process -Id $launch.pid -ErrorAction SilentlyContinue)) {
            $statusLabel.Text = 'Queue process has stopped. Inspect logs and saved results.'
            $statusLabel.ForeColor = [Drawing.Color]::Khaki
        }
    } catch {
        $statusLabel.Text = "Reading live status: $($_.Exception.Message)"
        $statusLabel.ForeColor = [Drawing.Color]::Khaki
    }
})
$form.Add_Shown({$timer.Start()})
$form.Add_FormClosed({$timer.Stop(); $timer.Dispose()})
[Windows.Forms.Application]::Run($form)
