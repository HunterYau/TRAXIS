"""Prepare or run a serial uniform/LTS/polyphase broadband comparison.

No cloud lifecycle calls. --prepare never builds a GPU system or starts a sim.
"""
import argparse
import copy
import json
from pathlib import Path
import time
import numpy as np
from examples import pzt_broadband_study as baseline
from examples import resolved_pzt_queue as queue
from examples.pzt_convergence import cases
from examples.pzt_broadband import paired_metrics
from traxis.gpu_enriched_verlet import gpu_explicit_simulate
from traxis.gpu_local_time import gpu_lts_simulate
from traxis.local_time import fine_partition,polyphase_reconstruct
from traxis.storage import write_json,atomic_write,sha256_file
from examples.plate_outputs import queue_lock
from examples.experimental_five import PeakGuard
from traxis.resources import ResourceLimits

SOURCES=['traxis/local_time.py','traxis/gpu_local_time.py','examples/pzt_local_time_study.py']


def identity():
    return dict(baseline=baseline.identity(),lts={p:sha256_file(baseline.PROJECT/p) for p in SOURCES})


def prepare(root,substeps,frequencies,edge_width,boxes,device_budget_mib=20000):
    if root.exists(): raise FileExistsError('Use a new output run directory')
    if substeps not in (1,2,4): raise ValueError('Study ratios: 1, 2 or 4')
    if not np.isfinite(edge_width) or edge_width<0: raise ValueError('Nonnegative edge width required')
    if not np.isfinite(device_budget_mib) or not 0<device_budget_mib<=65536:
        raise ValueError('Explicit device budget must be within 65536 MiB')
    for box in boxes:
        b=np.asarray(box)
        if b.shape!=(6,) or not np.isfinite(b).all() or np.any(b[::2]>b[1::2]):
            raise ValueError('Fine boxes must have six finite ordered bounds')
    config=queue.configuration()
    if config.get('setup_profile') or config.get('damping_profile') or config['body']['bond']!='perfect':
        raise ValueError('Study requires the original lossless ideal-open broadband setup')
    case=cases()[0]
    if case['steps']*substeps>500000: raise ValueError('Fine reference exceeds supported step count')
    tasks=[f'kernel-{r+1}' for r in range(substeps)]
    tasks += [f'{method}-{f}' for f in frequencies for method in ('uniform','lts')]
    proof=identity()
    root.mkdir(parents=True,exist_ok=True)
    write_json(root/'plan.json',dict(schema=1,identity=proof,configuration=config,case=case,
        substeps=substeps,frequencies_hz=frequencies,edge_width_m=edge_width,boxes=boxes,
        device_budget_mib=device_budget_mib,
        coarse_dt_s=baseline.DURATION_S/case['steps'],tasks=tasks,simulation_started=False,
        limitation='Research implementation; full plate accuracy, phase and speedup unvalidated. Coarse CFL remains conservative.'))
    print(f'Prepared {len(tasks)} serial tasks; no simulation started. {root}',flush=True)


def run(root):
    plan=json.loads((root/'plan.json').read_text())
    if plan['identity']!=identity(): raise ValueError('Frozen sources/configuration changed')
    with queue_lock(root):
        case,p,H=plan['case'],plan['substeps'],plan['coarse_dt_s']
        steps=case['steps']
        config=copy.deepcopy(plan['configuration'])
        config['mesh']['cells']=[case['cells'],case['cells'],case.get('nz',1)]
        config['body'].update(zip(('sectors','rings','layers'),case['body']))
        # Screen the large vector workspace BEFORE building full operators.
        import cupy as cp
        import psutil
        ndof=3*np.prod([v*config['mesh']['order']+1 for v in config['mesh']['cells']],dtype=np.int64)
        forecast=int(ndof)*48*8
        free=cp.cuda.runtime.memGetInfo()[0]
        if forecast>.8*free or forecast>plan.get('device_budget_mib',20000)*1024**2:
            raise MemoryError('Full LTS plate estimate exceeds current GPU memory or declared budget; use a cloud GPU')
        if int(ndof)*8*8>psutil.virtual_memory().available:
            raise MemoryError('Full plate host vectors exceed currently available RAM')
        def event(phase,**details): print(json.dumps(dict(phase=phase,**details)),flush=True)
        # Existing builder checks local host/device budgets. Each solver also
        # checks current free VRAM and peak workspace before allocating states.
        guard=PeakGuard(ResourceLimits(**config['resource_limits']),telemetry_errors_fatal=False)
        guard.check(force=True)
        system,load,probes,required=queue.build(config,guard,event)
        system.screen=dict(system.screen,max_device_mib=plan.get('device_budget_mib',20000))
        if H>.8*system.stable_dt_s: raise ValueError('Coarse step fails conservative CFL; prepare finer settings')
        fine=fine_partition(system,load,plan['boxes'],plan['edge_width_m'])
        def progress(state):
            write_json(root/'status.json',dict(status='running',task=task,
                percent=100*state['steps_completed']/state['requested_steps'],**state),overwrite=True)
            if state['steps_completed']%256==0 or state['steps_completed']==state['requested_steps']:
                event('propagating',task=task,percent=100*state['steps_completed']/state['requested_steps'],**state)
        for task in plan['tasks']:
            folder=root/task
            folder.mkdir(exist_ok=True)
            receipt=folder/'result.json'
            if receipt.exists():
                saved=json.loads(receipt.read_text())
                if saved.get('completed') and saved.get('sha256')==sha256_file(folder/'trace.npz'): continue
            # Preserve partial task metadata before retrying the task from t=0.
            stamp=str(time.time_ns())
            for name in ('trace.npz','result.json'):
                path=folder/name
                if path.exists(): path.rename(folder/(name+'.interrupted-'+stamp))
            write_json(receipt,dict(completed=False,status='running'))
            tick=time.perf_counter()
            try:
                kind,number=task.split('-');number=int(number)
                if kind=='kernel':
                    source=lambda t,index=number: float(abs(t-index*H/p)<H/p*.25)
                else:
                    source=baseline.source(number,steps*p)[0]
                if kind=='uniform':
                    sim=gpu_explicit_simulate(system,steps*p,H/p,probes,load,force=source,
                        output_stride=1,energy_stride=256,progress=progress,max_workspace_mib=4096,guard=guard)
                    t,y,stats=sim.time_s[::p],sim.displacement_m[::p],sim.statistics
                else:
                    t,y,stats=gpu_lts_simulate(system,steps,H,probes,load,force=source,
                        fine=fine,substeps=p,progress=progress,guard=guard,
                        max_workspace_mib=plan.get('device_budget_mib',20000))
                atomic_write(folder/'trace.npz',lambda f:np.savez_compressed(f,time_s=t,voltage_v=y))
                write_json(receipt,dict(completed=True,seconds=time.perf_counter()-tick,
                    statistics=stats,sha256=sha256_file(folder/'trace.npz')),overwrite=True)
            except BaseException as exc:
                write_json(receipt,dict(completed=False,status='interrupted_or_failed',error=str(exc)),overwrite=True)
                write_json(root/'status.json',dict(status='interrupted_or_failed',task=task),overwrite=True)
                raise
        kernels=np.stack([np.load(root/f'kernel-{r+1}'/'trace.npz')['voltage_v'] for r in range(p)])
        rows=[]
        for f in plan['frequencies_hz']:
            direct=np.load(root/f'lts-{f}'/'trace.npz')['voltage_v']
            uniform=np.load(root/f'uniform-{f}'/'trace.npz')['voltage_v']
            force=baseline.source(f,steps*p)[0](np.arange(steps*p+1)*H/p)
            reconstructed=polyphase_reconstruct(kernels,force)
            metrics=dict(frequency_hz=f,local_vs_uniform=paired_metrics(uniform,direct),
                         broadband_vs_local=paired_metrics(direct,reconstructed))
            rows.append(metrics);event('comparison',**metrics)
            atomic_write(root/f'reconstructed-{f}.npz',lambda out:np.savez_compressed(out,time_s=np.arange(steps+1)*H,voltage_v=reconstructed),overwrite=True)
        write_json(root/'comparisons.json',rows,overwrite=True)
        timing={task:json.loads((root/task/'result.json').read_text())['seconds'] for task in plan['tasks']}
        write_json(root/'timings.json',timing,overwrite=True)
        write_json(root/'status.json',dict(status='completed'),overwrite=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare',action='store_true');mode.add_argument('--run',action='store_true')
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--substeps',type=int,default=2)
    parser.add_argument('--frequencies',type=int,nargs='+',default=baseline.FREQUENCIES)
    parser.add_argument('--edge-width-m',type=float,default=0.)
    parser.add_argument('--device-budget-mib',type=float,default=20000,
                        help='Full study defaults to a cloud GPU budget; actual free VRAM is still checked')
    parser.add_argument('--fine-box',type=float,nargs=6,action='append',default=[])
    args=parser.parse_args()
    if args.prepare:
        if not args.frequencies or any(f not in baseline.FREQUENCIES for f in args.frequencies):
            raise ValueError('Use available experimental frequencies')
        prepare(args.root,args.substeps,args.frequencies,args.edge_width_m,args.fine_box,args.device_budget_mib)
    else: run(args.root)


if __name__=='__main__': main()
