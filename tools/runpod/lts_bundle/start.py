"""Explicitly start the prepared simulation detached from SSH; never controls pods."""
import json
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))

if __name__=='__main__':
    if not (ROOT/'ready.json').exists(): raise RuntimeError('Run setup.sh first')
    output=ROOT/'outputs/simulation-runs/lts-005'
    pidfile=output/'launcher.json'
    if pidfile.exists():
        raise RuntimeError('Existing launcher receipt: inspect process and queue lock before restarting')
    with (output/'console.log').open('ab',buffering=0) as log:
        process=subprocess.Popen(['bash',str(ROOT/'tools/runpod/lts_bundle/run.sh')],cwd=ROOT,
            stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    receipt=dict(pid=process.pid,console=str(output/'console.log'),status=str(output/'status.json'),
        automatic_pod_stop=False)
    pidfile.write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt))
