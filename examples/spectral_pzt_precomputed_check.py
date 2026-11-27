"""Tiny real-GPU spectral sparse vs precomputed/graph execution check only."""
import os
os.environ['OPENBLAS_NUM_THREADS']='2'
os.environ['OMP_NUM_THREADS']='2'
import argparse
import json
from pathlib import Path
import numpy as np


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--local-cuda-root',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    handles=None
    if args.local_cuda_root:
        from examples.quadratic_pzt_sparse_check import cuda_paths
        handles=cuda_paths(args.local_cuda_root,args.output)
    import cupy as cp
    from scipy.signal import hilbert
    from traxis.config import Plate,Material
    from traxis.gpu_tensor import make_layout,build_gpu_tensor
    from traxis.pzt_receiver import DiskGeometry
    from traxis.piezoelectric import PiezoMaterial,read_profile
    from traxis.spectral_pzt_sparse import component
    from traxis.quadratic_pzt_sparse import SparseAttachment,ConsistentMass
    from traxis.gpu_quadratic_pzt_sparse import couple_sparse
    from traxis.gpu_pzt_precomputed import couple_precomputed
    from traxis.gpu_local_time import LocalActions
    from traxis.gpu_local_time_unchecked import gpu_lts_simulate as original
    from traxis.gpu_lts_optimized import gpu_lts_simulate as optimized
    from traxis.gpu_enriched_verlet import sparse_probe_set
    from traxis.local_time import fine_partition
    root=Path(__file__).resolve().parents[1]
    layout=make_layout(Plate(.04,.04,.001),(4,2,1),2)
    base=build_gpu_tensor(layout,Material())
    mat=PiezoMaterial.from_profile(read_profile(root/'configs/pzt/nce51-noliac-ver1404.json'))
    def geo(center,surface):return DiskGeometry(center,.002,.0005,surface,1)
    data=component(geo((.008,.02),'top'),mat,rings=1,sectors=8,layers=1)
    bodies=[];masses=[];next_dof=len(base.mass_diagonal)
    for center,surface,port in [((.008,.02),'top','voltage'),((.032,.02),'top','open'),((.032,.02),'bottom','open')]:
        a=SparseAttachment(data,layout,geo(center,surface),first_dof=next_dof,electrical=port)
        bodies.append(a);masses.append(ConsistentMass(a,base.mass_diagonal));next_dof=a.next_dof
    sparse=couple_sparse(base,bodies,masses)
    dense=couple_precomputed(base,bodies,masses,max_workspace_mib=128,block_columns=13)
    load=np.zeros(next_dof);load[bodies[0].indices]=-bodies[0].coupling
    probes=sparse_probe_set([(b.indices,b.coupling/b.total_capacitance_f) for b in bodies[1:]],next_dof)
    fine=fine_partition(dense,load)
    def relative(a,b):return float(np.linalg.norm(a-b)/max(np.linalg.norm(b),1e-300))
    errors={};rng=np.random.default_rng(83)
    x=cp.asarray(rng.normal(size=next_dof))
    old=LocalActions(sparse,fine);new=LocalActions(dense,fine)
    for op in ('full','local','mass_action'):
        errors[op]=relative(cp.asnumpy(getattr(new,op)(x)),cp.asnumpy(getattr(old,op)(x)))
    H=min(1e-9,.01*dense.stable_dt_s)
    source=lambda t: float(abs(t/H-1)<.3)+.37*float(abs(t/H-4)<.3)
    kwargs=dict(fine=fine,force=source,source_sampling='coarse',energy_stride=11)
    t,reference,stats=original(sparse,65,H,probes,load,**kwargs)
    modes=[]
    for graphs in (False,True):
        tt,y,s=optimized(dense,65,H,probes,load,**kwargs,use_graphs=graphs,graph_steps=7,screen_stability=False)
        np.testing.assert_array_equal(t,tt)
        derived=lambda z:np.column_stack((z,z[:,0]+z[:,1],z[:,0]-z[:,1]))
        per_channel=[]
        for label,a,b in zip(('top','bottom','S0-related','A0-related'),derived(y).T,derived(reference).T):
            raw=relative(a,b);envelope=relative(abs(hilbert(a)),abs(hilbert(b)))
            errors[f'{graphs}/{label}/raw']=raw;errors[f'{graphs}/{label}/envelope']=envelope
            per_channel.append(dict(channel=label,raw_relative_l2=raw,envelope_relative_l2=envelope))
        energy=relative(np.array(s['modified_energy_j']),np.array(stats['modified_energy_j']))
        errors[f'{graphs}/energy']=energy
        if s['energy_time_s']!=stats['energy_time_s']:raise ArithmeticError('Energy sampling changed')
        modes.append(dict(graphs=graphs,solver_seconds=s['solver_seconds'],graph_launches=s['graph_launches'],channels=per_channel,energy_relative_l2=energy))
    passed=all(np.isfinite(v) and v<1e-9 for v in errors.values())
    report=dict(passed=passed,scope='Tiny three-PZT operator/65-step checks only; no full plate simulation or production speed claim',
        timing_note='Single short runs include different first-use and graph-capture costs; timings cannot establish a full-study speedup.',
        dofs=next_dof,steps=65,gpu=cp.cuda.runtime.getDeviceProperties(0)['name'].decode(),
        sparse_seconds=stats['solver_seconds'],modes=modes,errors=errors,
        precomputed_verification=dense.screen['precomputed_verification'])
    (args.output/'gpu-check.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True)
    if not passed:raise ArithmeticError('Precomputed graph equivalence failed')


if __name__=='__main__':main()
