"""Small native SpSM/graph equivalence and warmed replay timing only."""
import os
os.environ['OPENBLAS_NUM_THREADS']='2'
os.environ['OMP_NUM_THREADS']='2'
import argparse
import json
from pathlib import Path
import time
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
    from traxis.gpu_cached_lu import CachedLU
    from traxis.gpu_native_cusparse import NativeLibrary,NativeCachedLU,SP_SM_ARGS
    from traxis import gpu_spectral_native_graph as graph_adapter
    from traxis import gpu_lts_optimized as original_executor
    from traxis.config import Plate,Material
    from traxis.pzt_receiver import DiskGeometry
    from traxis.piezoelectric import PiezoMaterial,read_profile
    from traxis.spectral_partial import component
    from traxis.quadratic_pzt_sparse import SparseAttachment,ConsistentMass
    from traxis.gpu_tensor import make_layout,build_gpu_tensor
    from traxis.gpu_enriched_verlet import sparse_probe_set
    from traxis.local_time import fine_partition
    from traxis.gpu_local_time import LocalActions
    from traxis.gpu_local_time_unchecked import gpu_lts_simulate as fallback_simulate
    if cp.cuda.runtime.memGetInfo()[0]<512*2**20:raise MemoryError('Small check requires 512 MiB GPU reserve')
    assert len(SP_SM_ARGS)==10
    assert graph_adapter._execute.__code__ is original_executor._execute.__code__
    assert graph_adapter._simulate.__code__ is original_executor.gpu_lts_simulate.__code__
    library=NativeLibrary();print('Native library:',json.dumps(library.identity),flush=True)
    errors={};records=[];rng=np.random.default_rng(723)
    def relative(a,b):return float(np.linalg.norm(a-b)/max(np.linalg.norm(b),1e-300))
    matrix=sparse.random(31,31,density=.2,rng=rng,format='csc')+sparse.diags(np.linspace(1,4,31))
    factor=splu(matrix)
    for nrhs in (1,3):
        print('Checking native/fallback and graph replay, RHS columns:',nrhs,flush=True)
        plan=NativeCachedLU(CachedLU(factor,nrhs),library)
        for j in range(3):
            rhs=np.asfortranarray(rng.normal(size=(31,nrhs)))
            plan.rhs.set(rhs);actual=cp.asnumpy(plan.solve())
            errors[f'lu/{nrhs}/native/{j}']=relative(actual,factor.solve(rhs))
            errors[f'lu/{nrhs}/fallback/{j}']=relative(cp.asnumpy(plan.fallback.solve()),factor.solve(rhs))
        cp.cuda.get_current_stream().synchronize()
        stream=cp.cuda.Stream(non_blocking=True);plan.bind_stream(stream)
        with stream:
            plan.solve();stream.synchronize()
            stream.begin_capture()
            try:plan.solve()
            finally:graph=stream.end_capture()
            for j in range(5):
                rhs=np.asfortranarray(rng.normal(size=(31,nrhs)))
                plan.rhs.set(rhs);graph.launch(stream=stream);stream.synchronize()
                errors[f'lu/{nrhs}/graph/{j}']=relative(cp.asnumpy(plan.result),factor.solve(rhs))
        records.append(dict(nrhs=nrhs,native_analysis_calls=plan.analysis_calls,
            fallback_analysis_calls=plan.fallback.analysis_calls,graph_replays=5))
        if plan.analysis_calls!=2 or plan.fallback.analysis_calls!=2:raise ArithmeticError('Repeated analysis')
        plan.close();plan.close()
    print('Building only the tiny 1,431-DOF three-PZT component model',flush=True)
    root=Path(__file__).resolve().parents[1]
    mat=PiezoMaterial.from_profile(read_profile(root/'configs/pzt/nce51-noliac-ver1404.json'))
    layout=make_layout(Plate(.04,.04,.001),(4,2,1),2);base=build_gpu_tensor(layout,Material())
    def geo(center,surface):return DiskGeometry(center,.002,.0005,surface,1)
    data=component(geo((.008,.02),'top'),mat,rings=1,sectors=8,layers=1)
    bodies=[];masses=[];next_dof=len(base.mass_diagonal)
    for center,surface,port in [((.008,.02),'top','voltage'),((.032,.02),'top','open'),((.032,.02),'bottom','open')]:
        a=SparseAttachment(data,layout,geo(center,surface),first_dof=next_dof,electrical=port)
        bodies.append(a);masses.append(ConsistentMass(a,base.mass_diagonal));next_dof=a.next_dof
    system=graph_adapter.couple_native(base,bodies,masses)
    load=np.zeros(next_dof);load[bodies[0].indices]=-bodies[0].coupling
    probes=sparse_probe_set([(b.indices,b.coupling/b.total_capacitance_f) for b in bodies[1:]],next_dof)
    fine=fine_partition(system,load)
    # Compare implicit actions and inverse in native and preserved fallback modes.
    x=cp.asarray(rng.normal(size=next_dof));action=LocalActions(system,fine)
    system.set_native_enabled(False)
    expected={name:cp.asnumpy(getattr(action,name)(x)).copy() for name in ('full','local','mass_action')}
    xx=x.copy();system.inverse_mass_inplace(xx);expected_inverse=cp.asnumpy(xx)
    system.set_native_enabled(True)
    for name in expected:errors['actions/'+name]=relative(cp.asnumpy(getattr(action,name)(x)),expected[name])
    xx=x.copy();system.inverse_mass_inplace(xx);errors['actions/inverse']=relative(cp.asnumpy(xx),expected_inverse)
    H=min(1e-9,.01*system.stable_dt_s)
    source=lambda t:float(abs(t/H-1)<.3)+.37*float(abs(t/H-4)<.3)
    kwargs=dict(fine=fine,force=source,source_sampling='coarse',energy_stride=11)
    system.set_native_enabled(False)
    t0,y0,s0=fallback_simulate(system,65,H,probes,load,**kwargs)
    system.set_native_enabled(True)
    stats=[]
    for graphs in (False,True):
        events=[]
        print('Checking unchanged executor, graph mode:',graphs,flush=True)
        tt,yy,ss=graph_adapter.gpu_lts_simulate(system,65,H,probes,load,**kwargs,
            screen_stability=False,use_graphs=graphs,graph_steps=7,progress=events.append)
        np.testing.assert_array_equal(tt,t0)
        derived=lambda y:np.column_stack((y,y[:,0]+y[:,1],y[:,0]-y[:,1]))
        for label,a,b in zip(('top','bottom','S0-related','A0-related'),derived(y0).T,derived(yy).T):
            errors[f'wave/{graphs}/{label}']=relative(b,a)
            errors[f'envelope/{graphs}/{label}']=relative(abs(hilbert(b)),abs(hilbert(a)))
        errors[f'energy/{graphs}']=relative(np.array(ss['modified_energy_j']),np.array(s0['modified_energy_j']))
        if ss['energy_time_s']!=s0['energy_time_s']:raise ArithmeticError('Energy sampling changed')
        if [e['steps_completed'] for e in events]!=[16,32,48,64,65]:raise ArithmeticError('Progress checkpoints changed')
        if graphs and ss['graph_launches']<2:raise ArithmeticError('Graph replay did not occur')
        stats.append(dict(graphs=graphs,solver_seconds=ss['solver_seconds'],graph_launches=ss['graph_launches'],graph_variants=ss['graph_variants']))
    # Also preserve the executor's alternate fine-source sampling algebra.
    fine_kwargs=dict(kwargs,source_sampling='fine')
    system.set_native_enabled(False)
    tf,yf,sf=fallback_simulate(system,33,H,probes,load,**fine_kwargs)
    system.set_native_enabled(True)
    tg,yg,sg=graph_adapter.gpu_lts_simulate(system,33,H,probes,load,**fine_kwargs,
        screen_stability=False,use_graphs=True,graph_steps=3)
    np.testing.assert_array_equal(tf,tg)
    errors['fine_source/wave']=relative(yg,yf)
    errors['fine_source/energy']=relative(np.array(sg['modified_energy_j']),np.array(sf['modified_energy_j']))
    # A missing/incompatible bridge must retain the original cached GPU path.
    from unittest.mock import patch
    with patch.object(graph_adapter,'NativeLibrary',side_effect=RuntimeError('injected unavailable native bridge')):
        other=graph_adapter.couple_native(base,bodies,masses,allow_fallback=True)
        if other.screen['native_bridge'] or other.screen['cuda_graph_capture_supported']:
            raise ArithmeticError('Unavailable native bridge was advertised as graph capable')
        actual=LocalActions(other,fine).full(x)
        errors['unavailable_bridge/fallback']=relative(cp.asnumpy(actual),expected['full'])
        other.close()
    # Warmed fixed action pair: same operators/work for ordinary vs graph.
    print('Timing warmed action-pair replay; no long simulation benchmark',flush=True)
    cp.cuda.get_current_stream().synchronize();stream=cp.cuda.Stream(non_blocking=True)
    with stream:
        actions=graph_adapter.MatrixFreeReusableActions(system,fine)
        a=cp.empty_like(x);b=cp.empty_like(x);projected=cp.where(actions.fine,x,0.)
        def pair():actions.full_into(x,a);actions.local_into(projected,b)
        pair();stream.synchronize()
        expected_a=cp.asnumpy(a);expected_b=cp.asnumpy(b)
        stream.begin_capture()
        try:pair()
        finally:replay=stream.end_capture()
        replay.launch(stream=stream);stream.synchronize()
        errors['replay/full']=relative(cp.asnumpy(a),expected_a)
        errors['replay/local']=relative(cp.asnumpy(b),expected_b)
        timings={}
        for label,call in (('ordinary',pair),('graph',lambda:replay.launch(stream=stream))):
            samples=[];device=[]
            for _ in range(3):
                start=cp.cuda.Event();stop=cp.cuda.Event();stream.synchronize()
                tick=time.perf_counter();start.record(stream)
                for _ in range(10):call()
                stop.record(stream);stop.synchronize()
                samples.append((time.perf_counter()-tick)*1000/10)
                device.append(cp.cuda.get_elapsed_time(start,stop)/10)
            timings[label]=dict(host_wall_ms=samples,gpu_event_ms=device,
                median_host_wall_ms=float(np.median(samples)),median_gpu_event_ms=float(np.median(device)))
        pool_before=cp.get_default_memory_pool().used_bytes()
        for _ in range(10):replay.launch(stream=stream)
        stream.synchronize();pool_after=cp.get_default_memory_pool().used_bytes()
        if pool_after!=pool_before:raise ArithmeticError('Replay grew retained CuPy allocations')
    for plan in system.native_factors:plan.bind_stream()
    if any(p.analysis_calls!=2 for p in system.native_factors):raise ArithmeticError('Native analysis repeated')
    passed=all(np.isfinite(e) and e<1e-9 for e in errors.values())
    report=dict(passed=passed,scope='Small native ABI/LU, 65-step three-PZT equivalence, and warmed action replay only',
        gpu=cp.cuda.runtime.getDeviceProperties(0)['name'].decode(),library=library.identity,
        lu_checks=records,errors=errors,largest_relative_difference=max(errors.values()),
        executor_code_objects_unchanged=True,executor_modes=stats,dofs=next_dof,
        native_analysis_calls=sum(p.analysis_calls for p in system.native_factors),
        timings=timings,median_host_speedup=timings['ordinary']['median_host_wall_ms']/timings['graph']['median_host_wall_ms'],
        timing_scope='Three batches of ten warmed identical action pairs on the local GPU; excludes graph construction; not a full-plate speed prediction',
        replay_pool_bytes_before=pool_before,replay_pool_bytes_after=pool_after,
        fallback_preserved_and_tested=True,precision='FP64',full_simulations_started=0)
    (args.output/'gpu-check.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)
    system.close()
    if not passed:raise ArithmeticError('Native graph numerical equivalence failed')


if __name__=='__main__':main()
