"""Loopback-only browser console for the active RunPod spectral Q2 study."""
from __future__ import annotations

import json
import os
import subprocess
import shlex
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[2]
KEY = Path(os.environ.get("TRAXIS_SSH_KEY", str(Path.home() / ".ssh" / "traxis_runpod_ed25519")))
KNOWN_HOSTS = Path(os.environ.get("TRAXIS_KNOWN_HOSTS", str(ROOT / "outputs" / "runpod-transfer" / "spectral-q2-transfer-001" / "known_hosts")))
REMOTE = os.environ.get("TRAXIS_SSH_REMOTE", "root@205.196.17.154")
PORT = os.environ.get("TRAXIS_SSH_PORT", "11427")
REMOTE_PROJECT = os.environ.get("TRAXIS_REMOTE_PROJECT", "/workspace/traxis-local-aluminum-001/TRAXIS")
STUDY_ROOT = os.environ.get("TRAXIS_STUDY_ROOT", "outputs/simulation-runs/spectral-q2-local-aluminum-001")
REMOTE_COMMAND = r'''cd REMOTE_PROJECT || exit 1
echo '===STATUS==='
cat STUDY_ROOT/status.json
echo '===WORKERS==='
tail -n 24 STUDY_ROOT/console.log 2>/dev/null || true
for f in STUDY_ROOT/cases/*/worker.log; do
  [ -f "$f" ] || continue
  echo "--- $f ---"
  tail -n 14 "$f"
done
echo '===GPU==='
nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader'''

REMOTE_COMMAND = REMOTE_COMMAND.replace("STUDY_ROOT", shlex.quote(STUDY_ROOT)).replace("REMOTE_PROJECT", shlex.quote(REMOTE_PROJECT))

PAGE = r'''<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Spectral Q2 RunPod Progress</title>
<style>
:root{color-scheme:dark;--bg:#10151e;--panel:#1a2330;--muted:#9caabd;--accent:#72d7ad;--line:#2a3748}
body{margin:0;background:var(--bg);color:#eef4fb;font:15px/1.5 system-ui,Segoe UI,sans-serif}
main{max-width:1100px;margin:32px auto;padding:0 20px}.top{display:flex;justify-content:space-between;align-items:center;gap:16px}
h1{font-size:23px;margin:0}.muted{color:var(--muted)}.pill{color:var(--accent);border:1px solid #356b58;border-radius:99px;padding:4px 11px;font-size:13px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px;margin:22px 0}
.card,section{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:16px}.label{color:var(--muted);font-size:12px;text-transform:uppercase;letter-spacing:.06em}.value{font-size:20px;font-weight:650;margin-top:5px}
.bar{height:12px;background:#293546;border-radius:99px;overflow:hidden;margin:10px 0 18px}.bar span{display:block;height:100%;width:0;background:linear-gradient(90deg,#58b8ff,var(--accent));transition:width .4s}
h2{font-size:16px;margin:0 0 12px}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#0d1219;border-radius:8px;padding:14px;color:#c7d4e4;max-height:520px;overflow:auto;font:12px/1.5 ui-monospace,Consolas,monospace}
button{background:#243447;color:#eef4fb;border:1px solid #3b5068;border-radius:8px;padding:7px 12px;cursor:pointer}
</style></head><body><main>
<div class="top"><div><h1>Spectral Q2 PZT mesh study</h1><div class="muted">Live status from the H100 NVL RunPod</div></div><span id="connection" class="pill">Connectingâ€¦</span></div>
<div class="grid"><div class="card"><div class="label">Study status</div><div id="status" class="value">â€”</div></div>
<div class="card"><div class="label">Configuration</div><div id="case" class="value">â€”</div></div>
<div class="card"><div class="label">Current step</div><div id="steps" class="value">â€”</div></div>
<div class="card"><div class="label">Simulation stopwatch</div><div id="stopwatch" class="value">â€”</div></div>
<div class="card"><div class="label">Total run stopwatch</div><div id="total-stopwatch" class="value">--</div></div>
<div class="card"><div class="label">Current simulation ETA</div><div id="eta" class="value">--</div></div>
<div class="card"><div class="label">GPU</div><div id="gpu" class="value">â€”</div></div></div>
<div class="label">Configuration progress</div><div class="bar"><span id="progress"></span></div>
<section><div class="top"><h2>Live debug output</h2><button onclick="refresh()">Refresh</button></div><div id="updated" class="muted"></div><pre id="logs">Waiting for pod statusâ€¦</pre></section>
</main><script>
let timerState=null;
function clock(seconds){seconds=Math.max(0,Math.floor(seconds));return [Math.floor(seconds/3600),Math.floor(seconds/60)%60,seconds%60].map(x=>String(x).padStart(2,'0')).join(':')}
function updateTimers(){if(!timerState)return;const s=timerState;const delta=s.timer_updated_unix_s?Math.max(0,Date.now()/1000-s.timer_updated_unix_s):0;document.getElementById('total-stopwatch').textContent=clock((s.total_elapsed_seconds||0)+(s.run_timer_running?delta:0));document.getElementById('stopwatch').textContent=clock((s.simulation_seconds||0)+(s.simulation_timer_running?delta:0));document.getElementById('eta').textContent=s.eta_remaining_seconds!=null?clock(Math.max(0,s.eta_remaining_seconds-(s.simulation_timer_running?delta:0))):(s.eta_status==='calculating'?'Calculating...':'--')}
setInterval(updateTimers,1000);
async function refresh(){const c=document.getElementById('connection');try{const r=await fetch('/api/status',{cache:'no-store'});const d=await r.json();if(!r.ok)throw Error(d.error||r.statusText);const t=d.text||'';const sm=t.split('===STATUS===\n')[1]?.split('\n===WORKERS===')[0];const s=JSON.parse(sm);const w=t.split('===WORKERS===\n')[1]?.split('\n===GPU===')[0]||'';const g=t.split('===GPU===\n')[1]?.trim()||'';const p=Number(s.percent||0);document.getElementById('status').textContent=s.status||'â€”';document.getElementById('case').textContent=(s.case||'â€”')+' Â· '+(s.phase||'');document.getElementById('steps').textContent=s.steps_completed!=null?`${s.steps_completed.toLocaleString()} / ${(s.requested_steps||0).toLocaleString()} (${p.toFixed(2)}%)`:'â€”';timerState=s;updateTimers();document.getElementById('stopwatch').textContent=s.simulation_stopwatch||'â€”';document.getElementById('progress').style.width=Math.max(0,Math.min(100,p))+'%';document.getElementById('gpu').textContent=g.split(',').slice(0,2).join(',').trim()||'â€”';document.getElementById('logs').textContent='STATUS\n'+JSON.stringify(s,null,2)+'\n\nWORKER LOGS\n'+w+'\n\nGPU\n'+g;c.textContent='Connected';c.style.color='var(--accent)';document.getElementById('updated').textContent='Updated '+new Date(d.fetched_at).toLocaleTimeString()}catch(e){c.textContent='Connection issue';c.style.color='#ffb4a8';document.getElementById('logs').textContent='Could not read the live RunPod status.\n'+e}}
refresh();setInterval(refresh,5000);
</script></body></html>'''


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/":
            body = PAGE.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if path == "/api/status":
            args = ["ssh", "-T", "-i", str(KEY), "-p", PORT, "-o", "BatchMode=yes",
                    "-o", "ConnectTimeout=5", "-o", f"UserKnownHostsFile={KNOWN_HOSTS}",
                    "-o", "StrictHostKeyChecking=yes", REMOTE, REMOTE_COMMAND]
            try:
                result = subprocess.run(args, capture_output=True, text=True, timeout=12, check=False)
                payload = {"text": result.stdout, "error": result.stderr if result.returncode else "",
                           "returncode": result.returncode,
                           "fetched_at": datetime.now(timezone.utc).isoformat()}
                code = 200 if result.returncode == 0 else 502
            except Exception as exc:
                payload = {"error": str(exc), "fetched_at": datetime.now(timezone.utc).isoformat()}
                code = 502
            body = json.dumps(payload).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_error(404)

    def log_message(self, fmt, *args):
        pass


if __name__ == "__main__":
    ThreadingHTTPServer(("127.0.0.1", 8765), Handler).serve_forever()
