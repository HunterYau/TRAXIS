"""Compare fixed-face PZT operator costs, with GPU products/CPU FP64 solves.

Component proxy only: no attached plate, source waveform, integration or pod.
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='2'
os.environ['OMP_NUM_THREADS']='2'
import argparse,json,time
from pathlib import Path
import numpy as np
from scipy.sparse.linalg import splu
from traxis.spectral_piezoelectric import disk_mesh as hex_mesh,assemble as assemble_hex
from traxis.quadratic_piezoelectric import quadratic_disk,assemble as assemble_tet,electrical_reduction
from traxis.pzt_receiver import DiskGeometry
from traxis.piezoelectric import PiezoMaterial,read_profile

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--local-cuda-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    from examples.quadratic_pzt_sparse_check import cuda_paths
    handles=cuda_paths(a.local_cuda_root,a.output)
    import cupy as cp
    from cupyx.scipy import sparse
    mat=PiezoMaterial.from_profile(read_profile(ROOT/'configs/pzt/nce51-noliac-ver1404.json'))
    geo=DiskGeometry((.5,.5),.005,.0005,'top',1);records=[]
    for name in ['P2-6-64-2','Q2-s16-r2-z1']:
        mesh=quadratic_disk(geo,rings=6,sectors=64,layers=2) if name.startswith('P2') else hex_mesh(geo,radial_cells=2)
        data=assemble_tet(mesh,mat,max_nodes=12000) if name.startswith('P2') else assemble_hex(mesh,mat,quadrature_order=6)
        n=len(mesh.points_m);B,D,bottom,_=electrical_reduction(data,mesh.points_m)
        scalar_free=np.setdiff1d(np.arange(n),bottom);ids=np.concatenate([scalar_free+c*n for c in range(3)])
        K=data.stiffness[ids][:,ids].tocsr();B=B[ids].tocsr();M=data.mass[scalar_free][:,scalar_free].tocsc()
        dl=splu(D,permc_spec='MMD_AT_PLUS_A');ml=splu(M,permc_spec='MMD_AT_PLUS_A')
        kg=sparse.csr_matrix(K);bg=sparse.csr_matrix(B);bt=bg.T.tocsr()
        x=np.random.default_rng(188).normal(size=len(ids));gx=cp.asarray(x)
        def stiffness(v):return kg@v+bg@cp.asarray(dl.solve(cp.asnumpy(bt@v)))
        def inverse(v):return cp.asarray(ml.solve(cp.asnumpy(v).reshape(3,-1).T).T.ravel())
        expected_k=K@x+B@dl.solve(B.T@x);expected_m=ml.solve(x.reshape(3,-1).T).T.ravel()
        actual_k=cp.asnumpy(stiffness(gx));actual_m=cp.asnumpy(inverse(gx))
        errors=dict(stiffness=float(np.linalg.norm(actual_k-expected_k)/np.linalg.norm(expected_k)),
                    mass_inverse=float(np.linalg.norm(actual_m-expected_m)/np.linalg.norm(expected_m)))
        if max(errors.values())>1e-10:raise ArithmeticError('Spectral action GPU parity failed')
        times={}
        for label,fn in [('stiffness',stiffness),('mass_inverse',inverse)]:
            for _ in range(3):fn(gx)
            cp.cuda.get_current_stream().synchronize();tick=time.perf_counter()
            for _ in range(30):fn(gx)
            cp.cuda.get_current_stream().synchronize();times[label+'_ms']=(time.perf_counter()-tick)*1000/30
        record=dict(id=name,free_mechanical_dofs=len(ids),electrical_dofs=D.shape[0],timings=times,errors=errors,
            gpu_pool_bytes=cp.get_default_memory_pool().total_bytes(),precision='FP64',mass='consistent')
        print(json.dumps(record),flush=True);records.append(record)
        del kg,bg,bt,gx;cp.get_default_memory_pool().free_all_blocks()
    (a.output/'action-check.json').write_text(json.dumps(dict(cases=records,
        device=cp.cuda.runtime.getDeviceProperties(0)['name'].decode(),
        limitation='Fixed-face component proxy only; no plate contact/interior coupling or full simulation speed claim.'),indent=2))


if __name__=='__main__':main()
