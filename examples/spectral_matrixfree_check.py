"""Small local tensor/cached-LU/coupled operator checks; no full simulations."""
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
    from scipy import sparse
    from scipy.sparse.linalg import splu
    from scipy.signal import hilbert
    from traxis.config import Plate,Material
    from traxis.gpu_tensor import make_layout,build_gpu_tensor
    from traxis.pzt_receiver import DiskGeometry
    from traxis.piezoelectric import PiezoMaterial,read_profile
    from traxis.spectral_piezoelectric import disk_mesh,assemble
    from traxis.spectral_pzt_sparse import component as old_component
    from traxis.spectral_partial import TensorReference,component
    from traxis.gpu_spectral_partial import DeviceTensor
    from traxis.gpu_cached_lu import CachedLU
    from traxis.quadratic_pzt_sparse import SparseAttachment,ConsistentMass
    from traxis.gpu_quadratic_pzt_sparse import couple_sparse
    from traxis.gpu_spectral_matrixfree import couple_matrixfree
    from traxis.gpu_local_time import LocalActions
    from traxis.gpu_local_time_unchecked import gpu_lts_simulate
    from traxis.gpu_enriched_verlet import sparse_probe_set
    from traxis.local_time import fine_partition
    free,total=cp.cuda.runtime.memGetInfo()
    if free<512*2**20:raise MemoryError('Small GPU checks need 512 MiB free reserve')
    root=Path(__file__).resolve().parents[1]
    mat=PiezoMaterial.from_profile(read_profile(root/'configs/pzt/nce51-noliac-ver1404.json'))
    rng=np.random.default_rng(514);errors={};timing={};plans=[]
    def relative(a,b):return float(np.linalg.norm(a-b)/max(np.linalg.norm(b),1e-300))
    print('Checking retained LU plans with changing RHS and nontrivial permutations',flush=True)
    a=sparse.random(23,23,density=.25,rng=rng,format='csc')+sparse.diags(np.linspace(1,4,23))
    factor=splu(a)
    for nrhs in (1,3):
        plan=CachedLU(factor,nrhs);plans.append(plan)
        for j in range(5):
            rhs=np.asfortranarray(rng.normal(size=(23,nrhs)))
            plan.rhs.set(rhs);answer=cp.asnumpy(plan.solve())
            errors[f'cached_lu/{nrhs}/{j}']=relative(answer,factor.solve(rhs))
        if plan.analysis_calls!=2:raise ArithmeticError('LU analysis repeated')
        plan.close()
    print('Checking fused tensor K, B, B.T, D and mass on two small curved meshes',flush=True)
    for rings,layers in ((1,1),(2,2)):
        mesh=disk_mesh(DiskGeometry((.02,.02),.003,.0005,'top',1),side_cells=2,radial_cells=rings,layers=layers)
        cpu=TensorReference(mesh,mat);device=DeviceTensor(cpu);ref=assemble(mesh,mat,quadrature_order=6)
        n=cpu.n;u=rng.normal(size=3*n);phi=rng.normal(size=n);x=rng.normal(size=n)
        mech=cp.empty(3*n,dtype=cp.float64);scalar=cp.empty(n,dtype=cp.float64)
        device.apply(cp.asarray(u),mech,scalar,0)
        errors[f'tensor/{rings}/{layers}/K']=relative(cp.asnumpy(mech),ref.stiffness@u)
        errors[f'tensor/{rings}/{layers}/BT']=relative(cp.asnumpy(scalar),ref.coupling.T@u)
        device.apply(cp.asarray(phi),mech,scalar,1)
        errors[f'tensor/{rings}/{layers}/B']=relative(cp.asnumpy(mech),ref.coupling@phi)
        errors[f'tensor/{rings}/{layers}/D']=relative(cp.asnumpy(scalar),ref.dielectric@phi)
        device.apply(cp.asarray(phi),mech,scalar,3)
        errors[f'tensor/{rings}/{layers}/B_only']=relative(cp.asnumpy(mech),ref.coupling@phi)
        xx,yy,zz=mesh.points_m.T
        bending=np.r_[-xx*(zz-.00025),np.zeros(n),.5*xx*xx]
        device.apply(cp.asarray(bending),mech,scalar,0)
        errors[f'tensor/{rings}/{layers}/bending']=relative(cp.asnumpy(mech),ref.stiffness@bending)
        translation=np.r_[np.ones(n),np.zeros(2*n)]
        device.apply(cp.asarray(translation),mech,scalar,0)
        errors[f'tensor/{rings}/{layers}/rigid_scaled_residual']=float(np.linalg.norm(cp.asnumpy(mech))/(np.linalg.norm(ref.stiffness.data)*np.linalg.norm(translation)))
        device.apply(cp.asarray(x),mech,scalar,2)
        errors[f'tensor/{rings}/{layers}/mass']=relative(cp.asnumpy(scalar),ref.mass[:n,:n]@x)
        # Tensor kernels support graph capture; pinned CuPy's cuSPARSE solve
        # bindings explicitly reject it. Verify only the supported tensor path.
        gu=cp.asarray(u);device.apply(gu,mech,scalar,0);cp.cuda.get_current_stream().synchronize()
        stream=cp.cuda.Stream(non_blocking=True)
        with stream:
            stream.begin_capture();device.apply(gu,mech,scalar,0);graph=stream.end_capture()
            graph.launch(stream=stream);graph.launch(stream=stream);stream.synchronize()
        errors[f'tensor/{rings}/{layers}/graph']=relative(cp.asnumpy(mech),ref.stiffness@u)
    layout=make_layout(Plate(.04,.04,.001),(4,2,1),2);base=build_gpu_tensor(layout,Material())
    def geo(center,surface):return DiskGeometry(center,.002,.0005,surface,1)
    old=old_component(geo((.008,.02),'top'),mat,rings=1,sectors=8,layers=1)
    new=component(geo((.008,.02),'top'),mat,rings=1,sectors=8,layers=1)
    # Both faces, driven/open ports, polarity and capacitance independently.
    print('Checking both faces, electrical ports and interface work',flush=True)
    for surface in ('top','bottom'):
        for port in ('voltage','open'):
            geometry=DiskGeometry((.008,.02),.002,.0005,surface,-1)
            a=SparseAttachment(old,layout,geometry,electrical=port,input_capacitance_f=.3*old.capacitance_f)
            b=SparseAttachment(new,layout,geometry,electrical=port,input_capacitance_f=.3*new.capacitance_f)
            mass=ConsistentMass(a,base.mass_diagonal);mass_new=ConsistentMass(b,base.mass_diagonal)
            ref=couple_sparse(base,[a],[mass]);fast=couple_matrixfree(base,[b],[mass_new])
            x=cp.asarray(rng.normal(size=len(ref.mass_diagonal)))
            load=np.zeros(len(x));load[a.indices]=-a.coupling
            fine=fine_partition(ref,load);oa=LocalActions(ref,fine);ob=LocalActions(fast,fine)
            key=surface+'/'+port
            for op in ('full','local','mass_action'):
                actual=cp.asnumpy(getattr(ob,op)(x));expected=cp.asnumpy(getattr(oa,op)(x))
                errors[key+'/'+op]=relative(actual,expected)
            xx=x.copy();yy=x.copy();fast.inverse_mass_inplace(xx);ref.inverse_mass_inplace(yy)
            errors[key+'/mass_inverse']=relative(cp.asnumpy(xx),cp.asnumpy(yy))
            patch=fast.stiffness.storage.device_patches[0]
            xx=cp.asarray(rng.normal(size=len(a.indices)));patch.stiffness(xx)
            cp.cuda.get_current_stream().synchronize()
            expected=cp.asnumpy(patch.stiffness(xx)).copy();cp.cuda.get_current_stream().synchronize()
            for _ in range(3):patch.stiffness(xx)
            errors[key+'/reused_action']=relative(cp.asnumpy(patch.stiff_out),expected)
            # Direct virtual-work identity checks the unchanged interface map.
            hx=rng.normal(size=len(a.indices));force=rng.normal(size=3*a.data.n)
            work_body=a.pull(hx)@force;work_plate=hx@a.push(force)
            errors[key+'/interface_work']=abs(work_body-work_plate)/max(abs(work_body),1e-300)
            if patch.factor.analysis_calls!=2 or patch.component.electrical.analysis_calls!=2:
                raise ArithmeticError('Patch analysis repeated')
            fast.close()
    print('Checking 65 unchanged LTS steps with three PZTs and all voltage channels',flush=True)
    bodies=[[],[]];masses=[[],[]]
    for index,data in enumerate((old,new)):
        next_dof=len(base.mass_diagonal)
        for center,surface,port in [((.008,.02),'top','voltage'),((.032,.02),'top','open'),((.032,.02),'bottom','open')]:
            a=SparseAttachment(data,layout,geo(center,surface),first_dof=next_dof,electrical=port)
            bodies[index].append(a);masses[index].append(ConsistentMass(a,base.mass_diagonal));next_dof=a.next_dof
    ref=couple_sparse(base,bodies[0],masses[0]);fast=couple_matrixfree(base,bodies[1],masses[1])
    load=np.zeros(next_dof);load[bodies[0][0].indices]=-bodies[0][0].coupling
    probes=sparse_probe_set([(b.indices,b.coupling/b.total_capacitance_f) for b in bodies[0][1:]],next_dof)
    fine=fine_partition(ref,load);H=min(1e-9,.01*ref.stable_dt_s)
    kwargs=dict(fine=fine,force=lambda t:float(abs(t/H-1)<.3)+.37*float(abs(t/H-4)<.3),source_sampling='coarse',energy_stride=11)
    ta,ya,sa=gpu_lts_simulate(ref,65,H,probes,load,**kwargs)
    tb,yb,sb=gpu_lts_simulate(fast,65,H,probes,load,**kwargs)
    np.testing.assert_array_equal(ta,tb)
    derived=lambda y:np.column_stack((y,y[:,0]+y[:,1],y[:,0]-y[:,1]))
    for label,a,b in zip(('top','bottom','S0-related','A0-related'),derived(ya).T,derived(yb).T):
        errors[label+'/raw']=relative(b,a)
        errors[label+'/envelope']=relative(abs(hilbert(b)),abs(hilbert(a)))
    errors['energy']=relative(np.array(sb['modified_energy_j']),np.array(sa['modified_energy_j']))
    # Warmed action-pair timing on this bounded component model only. Do not
    # extrapolate this ratio to the 28-million-DOF plate or cloud hardware.
    import time
    x=cp.asarray(rng.normal(size=next_dof))
    for label,system in (('reference',ref),('matrixfree',fast)):
        action=LocalActions(system,fine);action.full(x);action.local(x)
        cp.cuda.get_current_stream().synchronize();tick=time.perf_counter()
        for _ in range(5):action.full(x);action.local(x)
        cp.cuda.get_current_stream().synchronize()
        timing[label+'_action_pair_ms']=(time.perf_counter()-tick)*1000/5
    counts=dict(electrical=sum(c.electrical.analysis_calls for c in {id(p.component):p.component for p in fast.stiffness.storage.device_patches}.values()),
        mass=sum(p.factor.analysis_calls for p in fast.stiffness.storage.device_patches))
    passed=all(np.isfinite(x) and x<1e-9 for x in errors.values())
    report=dict(passed=passed,scope='Small operator, tensor-only graph and 65-step coupled checks only; no full plate simulation',
        coupled_cuda_graph_capture_supported=False,
        graph_limit='Pinned CuPy cuSPARSE binding rejects stream capture; cached analysis is reused with ordinary GPU launches.',
        gpu=cp.cuda.runtime.getDeviceProperties(0)['name'].decode(),cuda_runtime=cp.cuda.runtime.runtimeGetVersion(),
        errors=errors,largest_relative_difference=max(errors.values()),cached_analysis_calls=counts,
        retained_pzt_device_bytes=fast.screen['retained_pzt_device_bytes'],cartesian_mass_copies=1,
        timing=timing,timing_limit='Five warmed action pairs on a tiny local model; not a full-study speed prediction',
        matrixfree_k_and_coupling=True,assembled_mass='scalar sparse consistent',electrical='unchanged exact sparse LU')
    (args.output/'gpu-check.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)
    fast.close()
    if not passed:raise ArithmeticError('Numerical equivalence failed')


if __name__=='__main__':main()
