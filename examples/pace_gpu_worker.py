"""Cool a verified project worker by pausing host submission, without changing its state."""

import argparse
import json
from pathlib import Path
import time

import psutil

from traxis.resources import snapshot
from traxis.storage import canonical_hash,sha256_file,write_json


class CoolingPacer:
    def __init__(self,process,upper=78.,lower=72.):
        self.process,self.upper,self.lower = process,upper,lower
        self.paused = False

    def update(self,temperature):
        if not self.paused and temperature>=self.upper:
            self.process.suspend()
            self.paused = True
            return 'suspended'
        if self.paused and temperature<=self.lower:
            self.process.resume()
            self.paused = False
            return 'resumed'
        return None

    def close(self):
        if self.paused:
            try:
                self.process.resume()
            except psutil.NoSuchProcess:
                pass
            self.paused = False


def find_worker(plan):
    candidates = []
    for process in psutil.process_iter(['pid','cmdline']):
        try:
            args = process.info['cmdline'] or []
            if ('examples.enriched_gpu_wave' in args and '--worker' in args and '--plan' in args
                    and Path(args[args.index('--plan')+1]).resolve()==plan):
                candidates.append(process)
        except (psutil.NoSuchProcess,psutil.AccessDenied,ValueError,IndexError):
            continue
    ids = {p.pid for p in candidates}
    leaves = [p for p in candidates if not any(c.pid in ids for c in p.children())]
    if len(leaves)!=1:
        raise ValueError('expected exactly one live leaf worker for the explicit plan')
    return leaves[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    plan,output = args.plan.resolve(),args.output.resolve()
    if not plan.is_relative_to(project/'outputs') or not output.is_relative_to(project/'outputs'):
        raise ValueError('pacing is limited to project output plans and receipts')
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('GPU pacing needs a fresh output directory')
    process = find_worker(plan)
    created = process.create_time()
    output.mkdir(parents=True,exist_ok=True)
    pacer = CoolingPacer(process)
    started,last_pause = time.monotonic(),None
    report = dict(completed=False,worker_pid=process.pid,worker_created=created,
        plan_sha256=sha256_file(plan),pacer_sha256=sha256_file(Path(__file__)),
        suspend_at_c=78.,resume_at_c=72.,maximum_wall_seconds=2400.,pause_seconds=0.,events=[],samples=[],
        scope='Scheduling only for this exact project worker; original resource guards and numerical state remain active.')
    try:
        while process.is_running() and process.create_time()==created:
            if time.monotonic()-started>2400:
                raise TimeoutError('GPU pacer reached its wall-time cap')
            reading = snapshot()
            if reading['gpu_status']!='available' or len(reading['gpus'])!=1:
                raise RuntimeError('GPU pacing requires the observed single-GPU sensor')
            temperature = reading['gpus'][0]['temperature_c']
            elapsed = time.monotonic()-started
            report['samples'].append(dict(elapsed_seconds=elapsed,temperature_c=temperature,
                system_memory_percent=reading['system_memory_percent'],gpu_used_mib=reading['gpus'][0]['used_mib']))
            event = pacer.update(temperature)
            if event:
                report['events'].append(dict(event=event,elapsed_seconds=elapsed,temperature_c=temperature))
                if event=='suspended':
                    last_pause = time.monotonic()
                elif last_pause is not None:
                    report['pause_seconds'] += time.monotonic()-last_pause
                    last_pause = None
            write_json(output/'progress.json',report,overwrite=True)
            time.sleep(1)
        report['completed'] = True
    except psutil.NoSuchProcess:
        report['completed'] = True
    except BaseException as error:
        report['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        pacer.close()
        if last_pause is not None:
            report['pause_seconds'] += time.monotonic()-last_pause
        report['wall_seconds'] = time.monotonic()-started
        report['fingerprint'] = canonical_hash(report)
        write_json(output/'pacing-report.json',report)
    print(json.dumps({k:v for k,v in report.items() if k not in ('events','samples')}))
    return 0 if report['completed'] else 1


if __name__=='__main__':
    raise SystemExit(main())
