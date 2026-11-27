"""Small component equivalence/cost checks; --target assembles ONE PZT only."""
import os
os.environ['OPENBLAS_NUM_THREADS']='2'
os.environ['OMP_NUM_THREADS']='2'
import argparse
import json
import time
from pathlib import Path
import numpy as np
import psutil
from traxis.config import Plate
from traxis.gpu_tensor import make_layout
from traxis.pzt_receiver import DiskGeometry
from traxis.piezoelectric import PiezoMaterial,read_profile
from traxis.quadratic_piezoelectric import attach_quadratic_disk
from traxis.quadratic_pzt_sparse import component,SparseAttachment,ConsistentMass,receipt

ROOT=Path(__file__).resolve().parents[1]


def cuda_paths(root,out):
    """Windows-only optional local dependency paths, no installs/env edits."""
    import ctypes
    os.environ['CUPY_CACHE_DIR']=str((out/'cupy-cache').resolve())
    runtime=root/'.venv/Lib/site-packages';old=root/'outputs/stability-screen-audit/cuda-eigensolver-libs'
    solver=root/'outputs/quadratic-pzt-gpu-libs'
    os.environ['CUDA_PATH']=str(runtime/'nvidia/cuda_runtime')
    handles=[os.add_dll_directory(str(p.resolve())) for r in [runtime,old,solver] for p in r.glob('nvidia/*/bin')]
    dlls=[]
    for r,pat in [(old,'nvJitLink*.dll'),(old,'cublasLt*.dll'),(old,'cublas64*.dll'),(old,'cusparse64*.dll'),(solver,'cusolver64*.dll')]:
        matches=list(r.glob('nvidia/*/bin/'+pat))
        if len(matches)!=1:raise RuntimeError('Missing isolated dependency '+pat)
        dlls.append(ctypes.WinDLL(str(matches[0].resolve())))
    return handles,dlls


def relative(a,b):return float(np.linalg.norm(a-b)/max(np.linalg.norm(b),1e-300))


def gpu_check(out,factor_backend='gpu'):
    import cupy as cp
    from traxis.gpu_tensor import build_gpu_tensor
    from traxis.gpu_pzt_body import couple_gpu_bodies
    from traxis.gpu_quadratic_pzt_sparse import couple_sparse
    from traxis.gpu_local_time import LocalActions
    from traxis.local_time import split_lf2_step,fine_partition
    from traxis.config import Material
    layout=make_layout(Plate(.04,.04,.001),(2,2,1),2)
    geo=DiskGeometry((.02,.02),.003,.0005,'top',1)
    material=PiezoMaterial.from_profile(read_profile(ROOT/'configs/pzt/nce51-noliac-ver1404.json'))
    data=component(geo,material,rings=1,sectors=8,layers=1)
    attachment=SparseAttachment(data,layout,geo)
    base=build_gpu_tensor(layout,Material(young_pa=71e9,poisson=.33,density_kg_m3=2700))
    mass=ConsistentMass(attachment,base.mass_diagonal)
    dense=attach_quadratic_disk(layout,material,geo,rings=1,sectors=8,layers=1)
    ref=couple_gpu_bodies(base,[dense]);candidate=couple_sparse(base,[attachment],[mass],factor_backend=factor_backend)
    n=len(ref.mass_diagonal);load=np.zeros(n);load[attachment.indices]=-attachment.coupling
    fine=fine_partition(ref,load)
    actions=[LocalActions(s,fine) for s in [ref,candidate]]
    rng=np.random.default_rng(91);x=cp.asarray(rng.normal(size=n));errors={}
    for op in ['full','local','mass_action']:
        a,b=[cp.asnumpy(getattr(s,op)(x)) for s in actions];errors[op]=relative(b,a)
    for i,s in enumerate([ref,candidate]):
        y=x.copy();s.inverse_mass_inplace(y)
        if i==0:inverse=cp.asnumpy(y)
        else:errors['mass_inverse']=relative(cp.asnumpy(y),inverse)
    prev=cp.asarray(rng.normal(size=n)*1e-10);cur=cp.asarray(rng.normal(size=n)*1e-10)
    states=[];energy=[]
    for a in actions:
        p=prev.copy();c=cur.copy()
        e0=a.discrete_energy(p,c,1e-9,2)
        for j in range(16):
            nx=split_lf2_step(p,c,1e-9,2,a.full,a.local,a.fine,cp.zeros(n),lambda t:0.,j*1e-9)
            p,c=c,nx
        states.append(cp.asnumpy(c));energy.append([e0,a.discrete_energy(p,c,1e-9,2)])
    errors['sixteen_lts_steps']=relative(states[1],states[0]);errors['energy']=relative(np.array(energy[1]),np.array(energy[0]))
    errors['sparse_unforced_energy_relative_drift']=abs(energy[1][1]/energy[1][0]-1)
    timings={}
    for label,a in zip(['dense','sparse'],actions):
        a.full(x);a.local(x);cp.cuda.get_current_stream().synchronize()
        tick=time.perf_counter()
        for _ in range(10):a.full(x);a.local(x)
        cp.cuda.get_current_stream().synchronize();timings[label+'_full_plus_local_ms']=(time.perf_counter()-tick)*100
    record=dict(errors=errors,timing=timings,component=receipt(attachment,mass),gpu=cp.cuda.runtime.getDeviceProperties(0)['name'].decode(),
        gpu_pool_bytes=cp.get_default_memory_pool().total_bytes(),scope='Tiny operator/16 recurrence steps only, no full plate simulation',
        timing_limit='Ten warmed action pairs on tiny systems, not full-run speed predictions.')
    record['factor_backend']=factor_backend
    (out/f'gpu-check-{factor_backend}.json').write_text(json.dumps(record,indent=2));print(json.dumps(record,indent=2),flush=True)
    if max(errors.values())>1e-10:raise ArithmeticError('GPU sparse parity or energy check failed')


def target_check(out,device=False):
    geo=DiskGeometry((.08,.06),.005,.0005,'top',1)
    material=PiezoMaterial.from_profile(read_profile(ROOT/'configs/pzt/nce51-noliac-ver1404.json'))
    print('Target component: assembly and bound, no plate build/propagation',flush=True)
    data=component(geo,material,max_workspace_mib=1536,max_factor_mib=1536)
    print('Target component: factor consistent mass',flush=True)
    # Match saved plate layout exactly; make_layout creates topology metadata,
    # never plate matrices or device buffers. Extract real contact interpolation.
    # Exact baseline configuration is supplied by the caller's archived plan.
    if 'TRAXIS_TARGET_BASELINE_PLAN' not in os.environ:raise ValueError('Set TRAXIS_TARGET_BASELINE_PLAN to archived baseline plan.json')
    baseline=json.loads(Path(os.environ['TRAXIS_TARGET_BASELINE_PLAN']).read_text());config=baseline['configuration']
    layout=make_layout(Plate(*config['plate_m']),(250,250,1),config['mesh']['order'],max_dofs=32000000,max_device_mib=6144,max_host_mib=4096)
    geo=DiskGeometry(tuple(config['actuator_center_xy_m']),config['disk_radius_m'],config['disk_thickness_m'],'top',1)
    if geo.radius_m!=.005 or geo.thickness_m!=.0005:raise ValueError('Target geometry differs')
    from traxis.gpu_tensor import mass_diagonal
    from traxis.config import Material
    host=mass_diagonal(layout,Material(**config['host_material']))
    receipts=[];next_dof=layout.screen['vector_dofs'];indices=[]
    for j,(center,surface,electrical) in enumerate([(config['actuator_center_xy_m'],'top','voltage'),
        (config['receiver_center_xy_m'],'top','open'),(config['receiver_center_xy_m'],'bottom','open')]):
        geo=DiskGeometry(tuple(center),config['disk_radius_m'],config['disk_thickness_m'],surface,1)
        a=SparseAttachment(data,layout,geo,electrical=electrical,first_dof=next_dof)
        tick=time.perf_counter();mass=ConsistentMass(a,host,max_factor_mib=1536);elapsed=time.perf_counter()-tick
        x=np.random.default_rng(3).normal(size=len(a.indices));sol=mass.solve(x)
        check=relative(mass.action(sol),x)
        r=receipt(a,mass);r.update(mass_factor_seconds=elapsed,mass_solve_residual=check)
        if check>1e-10:raise ArithmeticError('Target consistent mass residual failed')
        receipts.append(r);indices.append(set(a.indices.tolist()));next_dof=a.next_dof
        if j==0 and device:
            import cupy as cp
            from traxis.gpu_quadratic_pzt_sparse import DeviceComponent,DevicePatch
            tick=time.perf_counter();dc=DeviceComponent(data);dp=DevicePatch(a,mass,dc)
            gx=cp.asarray(x);dp.stiffness(gx);dp.inverse(gx);cp.cuda.get_current_stream().synchronize()
            gpu_setup=time.perf_counter()-tick
            expected_k=a.stiffness_action(x);expected_m=mass.solve(x)
            errors=dict(stiffness=relative(cp.asnumpy(dp.stiffness(gx)),expected_k),mass_inverse=relative(cp.asnumpy(dp.inverse(gx)),expected_m))
            gpu_times={}
            for name,op in [('stiffness',dp.stiffness),('mass_inverse',dp.inverse)]:
                tick=time.perf_counter()
                for _ in range(5):op(gx)
                cp.cuda.get_current_stream().synchronize();gpu_times[name+'_ms']=(time.perf_counter()-tick)*200
            r.update(gpu_errors=errors,gpu_timings=gpu_times,gpu_upload_and_warmup_seconds=gpu_setup,
                gpu_pool_bytes=cp.get_default_memory_pool().total_bytes())
            if max(errors.values())>1e-10:raise ArithmeticError('Target GPU parity failed')
            dc_cpu=DeviceComponent(data,'cpu');dp_cpu=DevicePatch(a,mass,dc_cpu,'cpu')
            errors_cpu=dict(stiffness=relative(cp.asnumpy(dp_cpu.stiffness(gx)),expected_k),
                            mass_inverse=relative(cp.asnumpy(dp_cpu.inverse(gx)),expected_m))
            times_cpu={}
            for name,op in [('stiffness',dp_cpu.stiffness),('mass_inverse',dp_cpu.inverse)]:
                tick=time.perf_counter()
                for _ in range(5):op(gx)
                cp.cuda.get_current_stream().synchronize();times_cpu[name+'_ms']=(time.perf_counter()-tick)*200
            r['cpu_factor_hybrid']=dict(errors=errors_cpu,timing=times_cpu)
            if max(errors_cpu.values())>1e-10:raise ArithmeticError('Hybrid target parity failed')
            del dp_cpu,dc_cpu,dp,dc,gx;cp.get_default_memory_pool().free_all_blocks()
        del a,mass
    if any(indices[i].intersection(indices[j]) for i in range(3) for j in range(i)):
        raise ValueError('Target mass support overlaps')
    record=dict(components=receipts,conservative_gpu_payload_bytes=2*sum(r['sparse_arrays_and_factors_bytes'] for r in receipts),
        sampled_rss_bytes=psutil.Process().memory_info().rss,scope='Three full-size PZT mass factors in sequence, one shared component; no plate simulation',
        bound_s2=data.body_eigenvalue_bound,all_mass_supports_disjoint=True,
        timing_limit='Five warmed GPU component actions, not full plate timing prediction.')
    (out/'target-component.json').write_text(json.dumps(record,indent=2));print(json.dumps(record,indent=2),flush=True)


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--gpu',action='store_true');p.add_argument('--target',action='store_true')
    p.add_argument('--local-cuda-root',type=Path);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    handles=cuda_paths(a.local_cuda_root,a.output) if a.local_cuda_root else None
    if a.gpu:
        gpu_check(a.output,'gpu');gpu_check(a.output,'cpu')
    if a.target:target_check(a.output,device=a.gpu)


if __name__=='__main__':main()
