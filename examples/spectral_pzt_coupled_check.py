"""Tiny dense/sparse spectral coupling and original LTS parity; no plate study."""
import os
os.environ['OPENBLAS_NUM_THREADS']='2'
os.environ['OMP_NUM_THREADS']='2'
import argparse,json
from pathlib import Path
from types import SimpleNamespace
import numpy as np


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--local-cuda-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    from examples.quadratic_pzt_sparse_check import cuda_paths
    handles=cuda_paths(args.local_cuda_root,args.output)
    import cupy as cp
    from scipy import sparse
    from traxis.config import Plate,Material
    from traxis.gpu_tensor import make_layout,build_gpu_tensor
    from traxis.piezoelectric import PiezoMaterial,read_profile
    from traxis.pzt_receiver import DiskGeometry
    from traxis.spectral_pzt_sparse import component
    from traxis.quadratic_pzt_sparse import SparseAttachment,ConsistentMass
    from traxis.gpu_quadratic_pzt_sparse import couple_sparse
    from traxis.gpu_pzt_body import couple_gpu_bodies
    from traxis.gpu_local_time import LocalActions
    from traxis.local_time import fine_partition,split_lf2_step
    root=Path(__file__).resolve().parents[1]
    mat=PiezoMaterial.from_profile(read_profile(root/'configs/pzt/nce51-noliac-ver1404.json'))
    layout=make_layout(Plate(.04,.04,.001),(2,2,1),2)
    data=component(DiskGeometry((.02,.02),.003,.0005,'top',1),mat,rings=1,sectors=8,layers=1)
    rng=np.random.default_rng(188);errors={}
    for surface in ('top','bottom'):
        for electrical in ('voltage','open'):
            geo=DiskGeometry((.02,.02),.003,.0005,surface,1)
            base=build_gpu_tensor(layout,Material(young_pa=71e9,poisson=.33,density_kg_m3=2700))
            b=SparseAttachment(data,layout,geo,electrical=electrical)
            mass=ConsistentMass(b,base.mass_diagonal)
            T=np.zeros((3*data.n,len(b.indices)))
            for axis in range(3):T[axis*data.n:(axis+1)*data.n,b.positions[:,axis]]=b.t.toarray()
            K=data.stiffness.toarray()+data.bi@data.electrical_lu.solve(data.bi.T.toarray())
            K=T.T@K@T
            if electrical=='open':K+=np.outer(b.coupling,b.coupling)/b.total_capacitance_f
            M=T.T@sparse.block_diag([data.scalar_mass]*3).toarray()@T
            dense=SimpleNamespace(**b.__dict__,stiffness=K,mass=M)
            ref=couple_gpu_bodies(base,[dense]);candidate=couple_sparse(base,[b],[mass],factor_backend='gpu')
            load=np.zeros(len(ref.mass_diagonal));load[b.indices]=-b.coupling
            fine=fine_partition(ref,load);actions=[LocalActions(s,fine) for s in (ref,candidate)]
            x=cp.asarray(rng.normal(size=len(load)))
            def relative(a,b):return float(cp.linalg.norm(a-b)/cp.linalg.norm(a))
            key=surface+'/'+electrical
            for op in ('full','local','mass_action'):
                errors[key+'/'+op]=relative(*[getattr(a,op)(x) for a in actions])
            states=[]
            prev=cp.asarray(rng.normal(size=len(load))*1e-10);cur=cp.asarray(rng.normal(size=len(load))*1e-10)
            for action in actions:
                a,c=prev.copy(),cur.copy()
                for j in range(16):
                    nxt=split_lf2_step(a,c,1e-10,2,action.full,action.local,action.fine,cp.zeros_like(c),lambda t:0.,j*1e-10)
                    a,c=c,nxt
                states.append(c)
            errors[key+'/16_LTS_steps']=relative(*states)
            print(key,'checked',flush=True)
    if not all(np.isfinite(v) and v<1e-10 for v in errors.values()):raise ArithmeticError(errors)
    receipt=dict(status='passed',errors=errors,precision='FP64',mass='full consistent',scope='Tiny dense vs sparse spectral operator, both faces and ports, 16 original LTS steps. No full simulations.')
    (args.output/'verification.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt,indent=2))


if __name__=='__main__':main()
