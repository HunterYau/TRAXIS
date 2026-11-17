"""Save five fixed comparisons serially; allow a stop between completed tests."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import psutil

from examples.experimental_five import ROOT,PROJECT,FREQUENCIES,protocol
from examples.report_experimental_five import report
from traxis.elmer_reference import _stop_tree
from traxis.storage import sha256_file,write_json


def active_run(frequency):
    found=[]
    for process in psutil.process_iter(['name']):
        try:
            if (process.info['name'] or '').lower()!='python.exe':continue
            command=process.cmdline()
            if command[1:]==['-m','examples.experimental_five','run',str(frequency)]:
                found.append((process.pid,process.create_time()))
        except (psutil.NoSuchProcess,psutil.AccessDenied):continue
    if len(found)>1:raise RuntimeError('multiple launchers for one trial')
    return found[0] if found else None


def main():
    if json.loads((ROOT/'protocol.json').read_text())!=protocol():
        raise ValueError('frozen source/protocol check failed')
    owner=ROOT/'queue-owner.json'
    if owner.exists():
        previous=json.loads(owner.read_text())
        try:
            process=psutil.Process(previous['pid'])
            if process.create_time()==previous['created']:raise RuntimeError('queue is already running')
        except psutil.NoSuchProcess:pass
        raise RuntimeError('preserve previous queue owner; review before restarting queue')
    write_json(owner,dict(pid=os.getpid(),created=psutil.Process().create_time(),
                         queue_source_sha256=sha256_file(Path(__file__))))
    child=None
    with (ROOT/'queue-events.jsonl').open('x',encoding='utf-8') as events:
        def event(phase,**extra):
            value=dict(phase=phase,unix_time=time.time(),**extra)
            events.write(json.dumps(value)+'\n');events.flush()
            print(json.dumps(value),flush=True)
        try:
            event('queue_started',frequencies_hz=list(FREQUENCIES))
            for index,frequency in enumerate(FREQUENCIES,1):
                directory=ROOT/f'trial-{index}-{frequency//1000}khz'
                launcher=directory/'launcher.json'
                running=active_run(frequency)
                if not launcher.exists():
                    if running:
                        event('watching_existing_launcher',trial=index,pid=running[0],created=running[1])
                        while not launcher.exists():
                            try:
                                process=psutil.Process(running[0])
                                if process.create_time()!=running[1]:raise RuntimeError('launcher PID was reused')
                            except psutil.NoSuchProcess:
                                raise RuntimeError('launcher exited without its final record')
                            time.sleep(2)
                    else:
                        if (ROOT/'STOP_AFTER_CURRENT').exists():
                            event('stopped_between_tests',completed=index-1);return
                        event('starting_trial',trial=index,frequency_hz=frequency)
                        with (ROOT/f'trial-{index}-queue-console.txt').open('x',encoding='utf-8') as log:
                            child=subprocess.Popen([sys.executable,'-m','examples.experimental_five','run',str(frequency)],
                                cwd=PROJECT,stdout=log,stderr=subprocess.STDOUT,
                                creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                            child.wait()
                            if child.returncode:raise RuntimeError(f'launcher exited {child.returncode}')
                        child=None
                result=report(frequency)
                event('trial_saved',trial=index,frequency_hz=frequency,
                      arrival_error_us=result['arrival_error_us'],flight_relative_error=result['flight_relative_error'],
                      evidence_fingerprint=result['fingerprint'])
                if (ROOT/'STOP_AFTER_CURRENT').exists():
                    event('stopped_between_tests',completed=index);return
            event('all_five_saved',completed=5)
        except BaseException as exc:
            if child is not None and child.poll() is None:_stop_tree(child)
            event('queue_stopped_on_error',error=f'{type(exc).__name__}: {exc}')
            raise


if __name__=='__main__':main()
