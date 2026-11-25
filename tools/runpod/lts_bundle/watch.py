"""Run on the user's PC: back up a completed LTS run, verify it, then stop its pod.

Never creates, starts or resumes a pod. --launch starts only the simulation on
an already running pod. Keep this PC awake and online until backup finishes.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def verified_completion(folder):
    """Recheck saved bytes and required outputs before allowing pod shutdown."""
    manifest = json.loads((folder/'results-sha256.json').read_text())
    run = folder/'lts-005'
    for name, digest in manifest.items():
        path = run/name
        path.resolve().relative_to(run.resolve())
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError('Local backup checksum mismatch: '+name)
    required = ['status.json', 'plan.json', 'comparisons.json', 'performance.json',
                'summary.txt', 'all-results.txt', 'console.log']
    plan = json.loads((run/'plan.json').read_text())
    if plan['tasks'] != ['control-kernel', 'lts-kernel']:
        raise ValueError('Unexpected benchmark tasks')
    if json.loads((run/'status.json').read_text())['status'] != 'completed':
        raise ValueError('Run has not completed')
    for task in plan['tasks']:
        required += [task+'/result.json', task+'/trace.npz']
        receipt = json.loads((run/task/'result.json').read_text())
        if not receipt.get('completed') or receipt['sha256'] != manifest[task+'/trace.npz']:
            raise ValueError('Missing or invalid completed solve: '+task)
    for frequency in plan['frequencies_hz']:
        for method in ('normal-broadband', 'lts-broadband'):
            required += [f'{method}/{frequency//1000}khz/waveforms.npz']
        required += [f'lts-broadband/{frequency//1000}khz/comparison.json']
    if not set(required).issubset(manifest):
        raise ValueError('Backup is missing required result files')
    return run


def credential_env(dpapi_path):
    env = os.environ.copy()
    if dpapi_path:
        # Existing Windows user-encrypted credential; only the local CLI sees it.
        import ctypes
        from ctypes import wintypes
        class Blob(ctypes.Structure):
            _fields_ = [('cbData', wintypes.DWORD), ('pbData', ctypes.POINTER(ctypes.c_ubyte))]
        raw = bytes.fromhex(dpapi_path.read_text().strip())
        memory = ctypes.create_string_buffer(raw)
        source = Blob(len(raw), ctypes.cast(memory, ctypes.POINTER(ctypes.c_ubyte)))
        result = Blob()
        if not ctypes.windll.crypt32.CryptUnprotectData(ctypes.byref(source), None, None,
                                                      None, None, 1, ctypes.byref(result)):
            raise ctypes.WinError()
        try:
            env['RUNPOD_API_KEY'] = ctypes.string_at(result.pbData, result.cbData).decode('utf-16-le')
        finally:
            ctypes.windll.kernel32.LocalFree(result.pbData)
    return env


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--pod-id', required=True)
    parser.add_argument('--host', required=True)
    parser.add_argument('--port', type=int, required=True)
    parser.add_argument('--key', type=Path, required=True)
    parser.add_argument('--known-hosts', type=Path, required=True)
    parser.add_argument('--remote-root', default='/workspace/traxis-lts-six-frequency-003')
    parser.add_argument('--cli', type=Path, required=True)
    parser.add_argument('--dpapi-key', type=Path)
    parser.add_argument('--receipt', type=Path, required=True,
                        help='Local state file; reuse this file to resume failed downloads')
    parser.add_argument('--output-root', type=Path, default=ROOT/'outputs/runpod-results')
    parser.add_argument('--launch', action='store_true')
    parser.add_argument('--detach', action='store_true')
    parser.add_argument('--download-only', action='store_true',
                        help='Download and verify results, then exit without stopping the pod')
    args = parser.parse_args()
    if not re.fullmatch(r'[a-z0-9]+', args.pod_id):
        raise ValueError('Invalid pod id')
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9.-]*', args.host) or not 1 <= args.port <= 65535:
        raise ValueError('Invalid SSH endpoint')
    if not re.fullmatch(r'/workspace/traxis-lts-[A-Za-z0-9_-]+', args.remote_root):
        raise ValueError('Unexpected remote root')
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    if args.detach:
        log = args.receipt.with_suffix('.log')
        with log.open('ab', buffering=0) as out:
            options = dict(creationflags=subprocess.CREATE_NO_WINDOW) if os.name == 'nt' else dict(start_new_session=True)
            process = subprocess.Popen([sys.executable, '-u', str(__file__),
                *[a for a in sys.argv[1:] if a != '--detach']], stdin=subprocess.DEVNULL,
                stdout=out, stderr=subprocess.STDOUT, **options)
        print(f'Backup/stop watcher PID {process.pid}; log: {log}', flush=True)
        return
    env = credential_env(args.dpapi_key)
    def cli(*words):
        p = subprocess.run([str(args.cli.resolve()), *words], env=env, capture_output=True,
                           text=True, timeout=60)
        if p.returncode:
            raise RuntimeError('Runpod CLI request failed; pod kept running')
        return json.loads(p.stdout)
    # Validate account access before optionally launching any simulation.
    cli('pod', 'get', args.pod_id)
    options = ['-i', str(args.key.resolve(strict=True)), '-o', 'BatchMode=yes',
               '-o', 'StrictHostKeyChecking=yes', '-o', 'ConnectTimeout=15',
               '-o', 'UserKnownHostsFile='+str(args.known_hosts.resolve(strict=True)),
               '-p', str(args.port), 'root@'+args.host]
    def ssh(code):
        p = subprocess.run(['ssh', *options, 'python3 -c '+shlex.quote(code)],
                           capture_output=True, text=True, timeout=60)
        if p.returncode:
            raise RuntimeError('SSH unavailable; backup pending and pod kept running')
        return json.loads(p.stdout)
    project = args.remote_root+'/TRAXIS'
    state = json.loads(args.receipt.read_text()) if args.receipt.exists() else dict(
        pod_id=args.pod_id, remote_root=args.remote_root, stage='waiting')
    if (state['pod_id'], state['remote_root']) != (args.pod_id, args.remote_root):
        raise ValueError('Receipt belongs to another pod/run')
    def save():
        tmp = args.receipt.with_suffix('.tmp')
        tmp.write_text(json.dumps(state, indent=2))
        tmp.replace(args.receipt)
    save()
    if args.launch and not state.get('launch_requested') and state['stage'] == 'waiting':
        # Save intent first: a disconnect must never cause a duplicate launch.
        state['launch_requested'] = True
        save()
        ssh('import subprocess,json; subprocess.run(['+repr(project+'/.venv-lts/bin/python')+
            ','+repr(project+'/tools/runpod/lts_bundle/start.py')+'],check=True,stdout=subprocess.DEVNULL); print(json.dumps(True))')
    while state['stage'] != 'stopped':
        try:
            if state['stage'] == 'waiting':
                data = ssh('import json,os; from pathlib import Path; r=Path('+repr(project+'/outputs/simulation-runs/lts-005')+'); '
                    's=json.loads((r/"status.json").read_text()); '
                    'pid=json.loads((r/"launcher.json").read_text())["pid"]; '
                    'p=Path("/proc")/str(pid)/"stat"; '
                    's["process_alive"]=p.exists() and p.read_text().split(") ",1)[1].split()[0]!="Z"; print(json.dumps(s))')
                print(json.dumps(data), flush=True)
                if data.get('status') == 'completed' and not data['process_alive']:
                    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
                    dest = args.output_root/('lts-six-frequency-'+stamp)
                    # Each retry uses a fresh directory; failed partial copies are preserved.
                    subprocess.run([sys.executable, str(HERE/'transfer.py'), 'download',
                        '--host', args.host, '--port', str(args.port), '--key', str(args.key),
                        '--known-hosts', str(args.known_hosts), '--remote-root', args.remote_root,
                        '--destination', str(dest)], check=True, timeout=3600)
                    verified_completion(dest)
                    state.update(stage='download_verified', destination=str(dest.resolve()))
                    save()
                    print((dest/'lts-005/summary.txt').read_text(), flush=True)
            if state['stage'] == 'download_verified':
                verified_completion(Path(state['destination']))
                if args.download_only:
                    print('Backup verified on this computer; pod left running.', flush=True)
                    return
                cli('pod', 'stop', args.pod_id)
                state['stage'] = 'stop_requested'
                save()
            if state['stage'] == 'stop_requested':
                pod = cli('pod', 'get', args.pod_id)
                if pod.get('desiredStatus') == 'EXITED':
                    state['stage'] = 'stopped'
                    save()
                    print('Backup verified on this computer; pod stopped.', flush=True)
        except (OSError, ValueError, RuntimeError, KeyError, subprocess.SubprocessError) as exc:
            # Do not print CLI/SSH output or environment: they may contain credentials.
            print(f'{type(exc).__name__}: backup/stop pending; retrying in 30 seconds.', flush=True)
        if state['stage'] != 'stopped':
            time.sleep(30)


if __name__ == '__main__':
    main()
