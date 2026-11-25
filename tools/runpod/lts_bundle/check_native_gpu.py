"""Run only small stability-screen checks with optional isolated CUDA DLLs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import unittest

ROOT=Path(__file__).resolve().parents[3]


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--cuda-libs',type=Path,help='Isolated Windows pip --target folder')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    os.environ.setdefault('CUPY_CACHE_DIR',str(args.output/'cupy-cache'))
    os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
    os.environ.setdefault('OMP_NUM_THREADS','1')
    dll_dirs=[];loaded=[]
    if args.cuda_libs:
        if os.name!='nt':raise ValueError('--cuda-libs is for the Windows DLL installation')
        import ctypes
        libs=args.cuda_libs.resolve(strict=True)
        dll_dirs=[os.add_dll_directory(str(d)) for d in libs.glob('nvidia/*/bin')]
        # Explicit paths let CUDA Pathfinder discover the already loaded DLLs
        # without modifying Windows PATH or the simulation's .venv packages.
        for pattern in ('nvJitLink*.dll','cublasLt*.dll','cublas64*.dll','cusparse64*.dll'):
            paths=list(libs.glob('nvidia/*/bin/'+pattern))
            if len(paths)!=1:raise RuntimeError('Missing/ambiguous isolated CUDA DLL: '+pattern)
            loaded.append(ctypes.WinDLL(str(paths[0])))
    sys.path.insert(0,str(ROOT))
    import cupy as cp
    from traxis.lts_stability import _gpu_available
    _gpu_available(cp)
    properties=cp.cuda.runtime.getDeviceProperties(0)
    gpu=properties['name'].decode() if isinstance(properties['name'],bytes) else properties['name']
    free,total=cp.cuda.runtime.memGetInfo()
    print('Native GPU eigensolver loaded on '+gpu,flush=True)
    tick=time.perf_counter()
    suite=unittest.defaultTestLoader.loadTestsFromNames([
        'tests.test_lts_stability','tests.test_lts_stability_native_gpu.NativeGpuScreenTests',
        'tests.test_local_time_schedule'])
    with (args.output/'checks.txt').open('w') as log:
        result=unittest.TextTestRunner(stream=log,verbosity=2).run(suite)
    print((args.output/'checks.txt').read_text(),flush=True)
    from tests.test_lts_stability_native_gpu import NATIVE_RESULTS
    audit=ROOT/'outputs/stability-screen-audit'
    before=json.loads((audit/'solver-before.json').read_text())
    hashes={n:dict(before=expected,after=hashlib.sha256((ROOT/n).read_bytes()).hexdigest())
            for n,expected in before.items()}
    cmd=['git','-c','safe.directory='+ROOT.parents[1].as_posix(),'-c','safe.directory='+ROOT.as_posix(),
         '-C',str(ROOT),'diff','--',*before]
    diff=subprocess.run(cmd,capture_output=True,text=True,check=True).stdout
    unchanged=all(row['before']==row['after'] for row in hashes.values())
    same_diff=diff==(audit/'solver-git-diff-before.patch').read_text()
    report=dict(tests=result.testsRun,success=result.wasSuccessful(),skipped=len(result.skipped),
        seconds=time.perf_counter()-tick,gpu=gpu,cupy_version=cp.__version__,
        cuda_runtime=cp.cuda.runtime.runtimeGetVersion(),cuda_driver=cp.cuda.runtime.driverGetVersion(),
        free_gpu_before_bytes=free,total_gpu_bytes=total,
        gpu_pool_reserved_bytes=cp.get_default_memory_pool().total_bytes(),
        native_cupy_eigensolver_tested=True,full_simulations_executed=0,
        native_results=NATIVE_RESULTS,
        solver_hashes=hashes,solver_unchanged=unchanged,solver_git_diff_unchanged=same_diff)
    (args.output/'report.json').write_text(json.dumps(report,indent=2))
    (args.output/'solver-git-diff-after.patch').write_text(diff)
    print(json.dumps(report,indent=2),flush=True)
    if not (result.wasSuccessful() and not result.skipped and unchanged and same_diff):raise SystemExit(1)


if __name__=='__main__':main()
