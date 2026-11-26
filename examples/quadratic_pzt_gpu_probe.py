"""Small FP64 GPU sparse-QR feasibility probe; no plate or scheduler changes."""
import os
os.environ['OPENBLAS_NUM_THREADS']='2'
os.environ['OMP_NUM_THREADS']='2'
import argparse,ctypes,json,time
from pathlib import Path
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu
from traxis.quadratic_piezoelectric import quadratic_disk,assemble,electrical_reduction
from traxis.piezoelectric import PiezoMaterial,read_profile
from traxis.pzt_receiver import DiskGeometry

ROOT=Path(__file__).resolve().parents[1]

def main():
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--shared-libs',type=Path,required=True)
    parser.add_argument('--solver-libs',type=Path,required=True);parser.add_argument('--runtime-libs',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    os.environ['CUPY_CACHE_DIR']=str((args.output/'cupy-cache').resolve())
    runtime=args.runtime_libs.resolve();os.environ['CUDA_PATH']=str(runtime/'nvidia/cuda_runtime')
    handles=[];loaded=[]
    for folder in [args.shared_libs,args.solver_libs,runtime]:
        handles.extend(os.add_dll_directory(str(p.resolve())) for p in folder.glob('nvidia/*/bin'))
    for folder,pattern in [(args.shared_libs,'nvJitLink*.dll'),(args.shared_libs,'cublasLt*.dll'),
                           (args.shared_libs,'cublas64*.dll'),(args.shared_libs,'cusparse64*.dll'),(args.solver_libs,'cusolver64*.dll')]:
        files=list(folder.glob('nvidia/*/bin/'+pattern))
        if len(files)!=1:raise RuntimeError('Missing isolated DLL '+pattern)
        loaded.append(ctypes.WinDLL(str(files[0].resolve())))
    import cupy as cp
    from cupyx.scipy import sparse as gpu_sparse
    from cupyx.scipy.sparse.linalg import spsolve
    material=PiezoMaterial.from_profile(read_profile(ROOT/'configs/pzt/nce51-noliac-ver1404.json'))
    device=cp.cuda.runtime.getDeviceProperties(0);records=[]
    for rings,sectors,layers in [(1,16,1),(3,32,2)]:
        mesh=quadratic_disk(DiskGeometry((.5,.5),.005,.0005,'top',1),rings=rings,sectors=sectors,layers=layers)
        n=len(mesh.points_m);a=assemble(mesh,material);B,D,lower,_=electrical_reduction(a,mesh.points_m)
        fixed=np.concatenate([lower+i*n for i in range(3)]);free=np.setdiff1d(np.arange(3*n),fixed)
        K,M=a.stiffness[free][:,free],a.mass[free][:,free];B=B[free]
        scale=np.sqrt(np.median(K.diagonal())/np.median(D.diagonal()))
        S=sparse.bmat([[K-(2*np.pi*150000)**2*M,B*scale],[B.T*scale,-D*scale**2]],format='csr')
        # The physical solve uses four real RHS columns (S0/A0 real/imag).
        rhs=np.random.default_rng(71).normal(size=(S.shape[0],4))
        tick=time.perf_counter();lu=splu(S.tocsc(),permc_spec='MMD_AT_PLUS_A');cpu=lu.solve(rhs);cpu_seconds=time.perf_counter()-tick
        free_bytes,total=cp.cuda.runtime.memGetInfo();dense_upper=10*S.shape[0]**2*8
        if dense_upper>.7*free_bytes:raise MemoryError('Probe GPU workspace forecast exceeds free device memory')
        sg=gpu_sparse.csr_matrix(S);bg=cp.asarray(rhs);cp.cuda.get_current_stream().synchronize()
        # Warm up on a tiny independent matrix, not the timed physical system.
        spsolve(gpu_sparse.eye(4,format='csr',dtype=cp.float64),cp.ones(4,dtype=cp.float64));cp.cuda.get_current_stream().synchronize()
        tick=time.perf_counter();gpu=spsolve(sg,bg);cp.cuda.get_current_stream().synchronize();gpu_seconds=time.perf_counter()-tick
        actual=cp.asnumpy(gpu);error=float(np.linalg.norm(actual-cpu)/np.linalg.norm(cpu))
        residual=float(np.linalg.norm(S@actual-rhs)/np.linalg.norm(rhs))
        record=dict(mesh=[rings,sectors,layers],nodes=n,equations=S.shape[0],cpu_factor_and_four_rhs_seconds=cpu_seconds,
                    gpu_qr_four_rhs_seconds=gpu_seconds,gpu_speedup=cpu_seconds/gpu_seconds,
                    relative_solution_difference=error,relative_residual=residual,precision='FP64',gpu_pool_bytes=cp.get_default_memory_pool().total_bytes())
        records.append(record);print(json.dumps(record),flush=True)
        if error>1e-9 or residual>1e-9:raise ArithmeticError('GPU parity failed')
        del sg,bg,gpu;cp.get_default_memory_pool().free_all_blocks()
    result=dict(device=device['name'].decode() if isinstance(device['name'],bytes) else device['name'],cases=records,
        note='Small component probes only. GPU QR factor reuse is unavailable in this interface; four RHS cause repeated QR solves. No production switch implied.')
    (args.output/'probe.json').write_text(json.dumps(result,indent=2))

if __name__=='__main__':main()
