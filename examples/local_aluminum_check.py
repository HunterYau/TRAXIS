"""Tiny CPU/GPU equivalence and unchanged LTS executor checks."""
import os
os.environ['OPENBLAS_NUM_THREADS']='2'
os.environ['OMP_NUM_THREADS']='2'
import argparse,json
from pathlib import Path
import numpy as np

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--local-cuda-root',type=Path)
    parser.add_argument('--host-order',type=int,choices=(2,5),default=2)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();args.output.mkdir(exist_ok=True,parents=True)
    handles=None
    if args.local_cuda_root:
        from examples.quadratic_pzt_sparse_check import cuda_paths
        handles=cuda_paths(args.local_cuda_root,args.output)
    import cupy as cp
    from types import SimpleNamespace
    from traxis.config import Plate,Material
    from traxis.gpu_tensor import make_layout,build_gpu_tensor
    from traxis.local_aluminum import LocalMesh,LocalAttachment
    from traxis.gpu_local_aluminum import build_system,HybridActions,gpu_lts_simulate,fine_partition
    from traxis.pzt_receiver import DiskGeometry
    from traxis.piezoelectric import PiezoMaterial,read_profile
    from traxis.spectral_partial import component
    from traxis.quadratic_pzt_sparse import ConsistentMass
    from traxis.gpu_enriched_verlet import sparse_probe_set
    if cp.cuda.runtime.memGetInfo()[0]<512*2**20:raise MemoryError('512 MiB free GPU reserve required')
    root=Path(__file__).resolve().parents[1];errors={};rng=np.random.default_rng(12)
    material=Material(71e9,.33,2700);layout=make_layout(Plate(.024,.024,.001),(6,6,1),args.host_order)
    mesh=LocalMesh(layout,material,[(.012,.012)],patch_width_m=.012,cells_per_patch_axis=4)
    base=build_gpu_tensor(layout,material)
    mat=PiezoMaterial.from_profile(read_profile(root/'configs/pzt/nce51-noliac-ver1404.json'))
    geo=DiskGeometry((.012,.012),.0007,.0005,'top',1)
    data=component(geo,mat,rings=1,sectors=8,layers=1,max_workspace_mib=256,max_factor_mib=256)
    attachment=LocalAttachment(data,mesh,geo,first_dof=3*mesh.n)
    mass=ConsistentMass(attachment,np.tile(mesh.diagonal,3),max_factor_mib=256)
    system=build_system(base,mesh,[attachment],[mass],native=True)
    n=len(system.mass_diagonal);host=3*mesh.n
    def relative(a,b):return float(np.linalg.norm(a-b)/max(np.linalg.norm(b),1e-300))
    x=rng.normal(size=n)
    expected=np.r_[mesh.stiffness(x[:host]),np.zeros(n-host)]
    expected[attachment.indices]+=attachment.stiffness_action(x[attachment.indices])
    actual=system.stiffness@x;errors['coupled_stiffness']=relative(actual,expected)
    fine=fine_partition(system,np.zeros(n));actions=HybridActions(system,fine)
    projected=x*fine
    reference=np.r_[mesh.stiffness(projected[:host]),np.zeros(n-host)]
    reference[attachment.indices]+=attachment.stiffness_action(projected[attachment.indices])
    def inverse(y):
        z=np.r_[mesh.mass(y[:host],inverse=True),np.zeros(n-host)]
        z[attachment.indices]=mass.solve(y[attachment.indices]);return z
    expected=inverse(reference)
    errors['selected_fine_stiffness_and_mass_inverse']=relative(cp.asnumpy(actions.local(cp.asarray(x))),expected)
    expected=np.r_[mesh.mass(x[:host]),np.zeros(n-host)]
    expected[attachment.indices]=mass.action(x[attachment.indices])
    actual=cp.asnumpy(actions.mass_action(cp.asarray(x)));errors['coupled_mass']=relative(actual,expected)
    y=cp.asarray(actual);system.inverse_mass_inplace(y);errors['mass_inverse_roundtrip']=relative(cp.asnumpy(y),x)
    load=np.zeros(n);load[attachment.indices]=-attachment.coupling
    fine=fine_partition(system,load)
    probes=sparse_probe_set([(attachment.indices,attachment.coupling/attachment.total_capacitance_f)],n)
    source=lambda t:np.sin(2*np.pi*100000*t) if t>=0 else 0.
    outputs=[]
    for graph in (False,True):
        t,v,stats=gpu_lts_simulate(system,16,1e-10,probes,load,fine=fine,force=source,
            source_sampling='coarse',energy_stride=16,screen_stability=False,use_graphs=graph,graph_steps=4)
        outputs.append(v)
    errors['cuda_graph_vs_ordinary_trace']=relative(outputs[1],outputs[0])
    # The CPU reference applies the SAME original split-LFC recurrence.
    from traxis.local_time import split_lf2_step
    def full(u):
        r=np.r_[mesh.stiffness(u[:host]),np.zeros(n-host)];r[attachment.indices]+=attachment.stiffness_action(u[attachment.indices]);return inverse(r)
    def local(u):return full(u*fine)
    previous=current=np.zeros(n);trace=[0.];drive=inverse(load)
    for j in range(16):
        following=split_lf2_step(previous,current,1e-10,2,full,local,fine,drive,source,j*1e-10)
        previous,current=current,following;trace.append(attachment.coupling@current[attachment.indices]/attachment.total_capacitance_f)
    errors['GPU_vs_CPU_original_recurrence']=relative(outputs[0][:,0],np.array(trace))
    system.close()
    if max(errors.values())>1e-9:raise ArithmeticError(errors)
    result=dict(passed=True,errors=errors,mesh=mesh.report,full_plate_simulations=0,
        device=base.device,checks='Exact CPU/GPU actions, projected mass, native factor solve, graph capture, 16-step toy LTS comparison')
    (args.output/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))

if __name__=='__main__':main()
