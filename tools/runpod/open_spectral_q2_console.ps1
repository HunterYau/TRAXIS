$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$keyPath = Join-Path $env:USERPROFILE '.ssh\traxis_runpod_ed25519'
$knownHosts = Join-Path $repoRoot 'outputs\runpod-transfer\spectral-q2-transfer-001\known_hosts'
$remote = @'
cd /workspace/TRAXIS
while true; do
  clear
  echo 'Spectral Q2 PZT mesh study — RunPod H100 NVL'
  date -u
  cat outputs/simulation-runs/spectral-q2-convergence-001/status.json
  for f in outputs/simulation-runs/spectral-q2-convergence-001/cases/*/worker.log; do
    [ -f "$f" ] || continue
    echo "--- $f (latest progress) ---"
    tail -n 8 "$f"
  done
  sleep 5
done
'@
ssh -i $keyPath -p 11427 -o "UserKnownHostsFile=$knownHosts" -o StrictHostKeyChecking=yes root@205.196.17.154 $remote
Read-Host 'SSH console ended. Press Enter to close'
