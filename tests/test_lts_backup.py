"""Exercise backup verification and shutdown ordering without a real pod."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from tools.runpod.lts_bundle import watch


def backup(folder):
    run = folder/'lts-005'
    run.mkdir(parents=True)
    files = dict(status=json.dumps({'status':'completed'}))
    (run/'status.json').write_text(files['status'])
    plan = {'tasks':['control-kernel','lts-kernel'], 'frequencies_hz':[50000]}
    (run/'plan.json').write_text(json.dumps(plan))
    for name in ('comparisons.json','performance.json','summary.txt','all-results.txt','console.log'):
        (run/name).write_text('{}')
    for task in plan['tasks']:
        d = run/task; d.mkdir(); (d/'trace.npz').write_bytes(b'saved waveform')
        (d/'result.json').write_text(json.dumps({'completed':True,
            'sha256':hashlib.sha256((d/'trace.npz').read_bytes()).hexdigest()}))
    for method in ('normal-broadband','lts-broadband'):
        d = run/method/'50khz'; d.mkdir(parents=True)
        (d/'waveforms.npz').write_bytes(b'waveform')
    (run/'lts-broadband/50khz/comparison.json').write_text('{}')
    manifest = {p.relative_to(run).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
                for p in run.rglob('*') if p.is_file()}
    (folder/'results-sha256.json').write_text(json.dumps(manifest))


class BackupTests(unittest.TestCase):
    def test_manifest_cannot_omit_required_outputs(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp); backup(folder)
            manifest_path = folder/'results-sha256.json'
            manifest = json.loads(manifest_path.read_text())
            del manifest['lts-broadband/50khz/waveforms.npz']
            manifest_path.write_text(json.dumps(manifest))
            with self.assertRaises(ValueError): watch.verified_completion(folder)

    def test_corrupt_or_missing_waveform_rejects_shutdown(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp); backup(folder)
            watch.verified_completion(folder)
            trace = folder/'lts-005/lts-kernel/trace.npz'
            trace.write_bytes(b'corrupt')
            with self.assertRaises(ValueError): watch.verified_completion(folder)
            trace.unlink()
            with self.assertRaises(OSError): watch.verified_completion(folder)

    def test_failed_download_retries_before_stop_and_records_backup(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for name in ('key','hosts','cli'): (root/name).touch()
            receipt = root/'watch.json'; calls=[]; attempts=0
            def command(args, **kwargs):
                nonlocal attempts
                if 'transfer.py' in str(args[1]):
                    attempts += 1; calls.append('download')
                    if attempts == 1: raise subprocess.CalledProcessError(1,args)
                    backup(Path(args[args.index('--destination')+1]))
                    return subprocess.CompletedProcess(args,0)
                if args[0] == 'ssh':
                    return subprocess.CompletedProcess(args,0,json.dumps({'status':'completed','process_alive':False}))
                if 'stop' in args:
                    calls.append('stop')
                    state=json.loads(receipt.read_text())
                    self.assertEqual(state['stage'],'download_verified')
                    watch.verified_completion(Path(state['destination']))
                return subprocess.CompletedProcess(args,0,json.dumps({'desiredStatus':'EXITED'}))
            argv=['watch','--pod-id','abc123','--host','test.example','--port','22',
                  '--key',str(root/'key'),'--known-hosts',str(root/'hosts'),
                  '--cli',str(root/'cli'),'--receipt',str(receipt),'--output-root',str(root/'results')]
            with patch.object(sys,'argv',argv), patch.object(watch.subprocess,'run',side_effect=command), \
                 patch.object(watch.time,'sleep'), patch.object(watch,'credential_env',return_value={}):
                watch.main()
            self.assertEqual(calls,['download','download','stop'])
            self.assertEqual(json.loads(receipt.read_text())['stage'],'stopped')


if __name__ == '__main__': unittest.main()
