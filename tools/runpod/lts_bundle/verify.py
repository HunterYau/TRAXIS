"""Verify the unpacked snapshot and optionally run tiny CUDA checks only."""
import argparse
import contextlib
import hashlib
import json
from pathlib import Path
import platform
import sys
import unittest

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))


def verify(gpu=False):
    manifest=json.loads((ROOT/'bundle-manifest.json').read_text())
    for row in manifest['files']:
        p=ROOT/row['path'];p.resolve().relative_to(ROOT.resolve())
        if p.is_symlink() or not p.is_file() or p.stat().st_size!=row['bytes'] or hashlib.sha256(p.read_bytes()).hexdigest()!=row['sha256']:
            raise ValueError('File verification failed: '+row['path'])
    from examples import pzt_local_time_study as study
    plan=json.loads((ROOT/manifest['plan']/'plan.json').read_text())
    study.validate_plan(plan)
    if plan['identity']!=study.identity(): raise ValueError('Scientific source fingerprint differs')
    if plan['tasks']!=['control-kernel','lts-kernel'] or plan['frequencies_hz']!=study.FREQUENCIES:
        raise ValueError('Unexpected schedule')
    print('All file hashes and the two-method, six-frequency plan match.',flush=True)
    if not gpu: return
    if platform.system()!='Linux' or platform.machine()!='x86_64' or sys.version_info[:2]!=(3,12):
        raise RuntimeError('Deployment requires Linux x86_64 and Python 3.12')
    import cupy as cp
    import psutil
    from cupy.cuda import nvrtc
    props=cp.cuda.runtime.getDeviceProperties(0)
    name=props['name'].decode() if isinstance(props['name'],bytes) else props['name']
    free,total=cp.cuda.runtime.memGetInfo()
    hardware=dict(gpu=name,free_gpu_bytes=free,total_gpu_bytes=total,
        available_host_bytes=psutil.virtual_memory().available,python=sys.version,
        cuda_runtime=cp.cuda.runtime.runtimeGetVersion(),cuda_driver=cp.cuda.runtime.driverGetVersion(),
        nvrtc=nvrtc.getVersion())
    output=ROOT/manifest['plan'];(output/'hardware.json').write_text(json.dumps(hardware,indent=2))
    if hardware['cuda_runtime']!=12090 or tuple(hardware['nvrtc'])!=(12,9) or hardware['cuda_driver']<12090:
        raise RuntimeError('Pinned CUDA 12.9 runtime/compiler and compatible driver required')
    if free<24*1024**3 or hardware['available_host_bytes']<12*1024**3:
        raise MemoryError('Readiness screen requires 24 GiB free GPU and 12 GiB free host RAM')
    suite=unittest.defaultTestLoader.loadTestsFromNames(['tests.test_local_time',
        'tests.test_local_time_schedule','tests.test_gpu_local_time','tests.test_pzt_broadband.BroadbandTests'])
    with (output/'readiness-checks.txt').open('w') as log,contextlib.redirect_stdout(log):
        result=unittest.TextTestRunner(stream=log,verbosity=2).run(suite)
    if not result.wasSuccessful() or result.skipped or result.testsRun!=31:
        raise RuntimeError('Readiness checks failed or skipped; see readiness-checks.txt')
    (ROOT/'ready.json').write_text(json.dumps(dict(manifest_sha256=hashlib.sha256(
        (ROOT/'bundle-manifest.json').read_bytes()).hexdigest(),tests=31,hardware=hardware),indent=2))
    print('31 tiny checks passed. Full plate has NOT been simulated or stability-screened.',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--gpu',action='store_true')
    verify(p.parse_args().gpu)
