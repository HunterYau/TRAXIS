"""Two matched broadband methods on one intact plate, 50 through 300 kHz.

No cloud lifecycle calls. --prepare never builds a GPU system or starts a sim.
"""
import argparse
import copy
import json
import signal
from pathlib import Path
import time
import numpy as np
from examples import pzt_broadband_study as baseline
from examples import resolved_pzt_queue as queue
from examples.pzt_convergence import cases
from examples.pzt_broadband import paired_metrics,interior_impulse,reconstruct,within_target
from traxis.gpu_enriched_verlet import gpu_explicit_simulate
from traxis.gpu_local_time import gpu_lts_simulate,lts_stability_screen
from traxis.lts_stability import optimized_stability_screen
from traxis.local_time import fine_partition
from traxis.storage import write_json,atomic_write,sha256_file
from examples.plate_outputs import queue_lock
from examples.experimental_five import PeakGuard
from traxis.resources import ResourceLimits

SOURCES=['traxis/local_time.py','traxis/gpu_local_time.py','examples/pzt_local_time_study.py']
FREQUENCIES=[50000,100000,150000,200000,250000,300000]


def stopwatch(seconds):
    seconds=max(0.,float(seconds))
    hours,remainder=divmod(int(seconds),3600)
    minutes,secs=divmod(remainder,60)
    return f'{hours:02d}:{minutes:02d}:{secs:02d}'


def excitation(frequency,steps):
    if frequency not in FREQUENCIES: raise ValueError('Unsupported benchmark frequency')
    # Identical prescribed physical drive for both numerical methods.
    # No measured receiver or generator recording is loaded for this benchmark.
    duration=5/frequency
    def pulse(t):
        t=np.asarray(t,dtype=float)
        if not np.isfinite(t).all(): raise ValueError('Finite excitation time required')
        y=np.where((t>0)&(t<duration),np.sin(2*np.pi*frequency*t)*np.sin(np.pi*t/duration)**2,0.)
        return float(y) if y.ndim==0 else y
    return pulse,dict(frequency_hz=frequency,num_cycles=5,pulse_end_s=duration,
        provenance='prescribed unit-coefficient Hann sine for numerical comparison',
        experimental_ground_truth=False)


def identity():
    config=queue.configuration()
    paths=[*sorted((baseline.PROJECT/'traxis').glob('*.py')),
           baseline.PROJECT/'examples/resolved_pzt_queue.py',queue.CONFIG,
           baseline.PROJECT/config['material_profile'],
           *[baseline.PROJECT/p for p in SOURCES],
           baseline.PROJECT/'examples/pzt_broadband.py',
           baseline.PROJECT/'examples/pzt_convergence.py']
    return dict(scope='intact plate numerical method comparison; no experimental data',
        sources={str(p.relative_to(baseline.PROJECT)).replace('\\','/'):sha256_file(p) for p in paths})


def prepare(root,substeps,frequencies,edge_width,boxes,device_budget_mib=20000):
    if root.exists(): raise FileExistsError('Use a new output run directory')
    if substeps!=2: raise ValueError('Acceleration study currently validates ratio 2 only')
    if (not frequencies or len(set(frequencies))!=len(frequencies)
            or any(type(f) is not int or f not in FREQUENCIES for f in frequencies)):
        raise ValueError('Unique available experimental frequencies required')
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
    # Always refine an edge strip of at least one in-plane mesh element.
    edge_width=max(edge_width,config['plate_m'][0]/case['cells'],config['plate_m'][1]/case['cells'])
    # Hold the validated fine step FIXED. Increase H, halve global updates.
    tasks=['control-kernel','lts-kernel']
    proof=identity()
    root.mkdir(parents=True,exist_ok=True)
    write_json(root/'plan.json',dict(schema=3,identity=proof,configuration=config,case=case,
        plate_condition='intact',excitation_sources={str(f):excitation(f,case['steps'])[1] for f in frequencies},
        substeps=substeps,frequencies_hz=frequencies,edge_width_m=edge_width,boxes=boxes,
        device_budget_mib=device_budget_mib,
        fine_dt_s=baseline.DURATION_S/case['steps'],
        coarse_dt_s=substeps*baseline.DURATION_S/case['steps'],
        coarse_steps=case['steps']//substeps,source_sampling='coarse',
        tasks=tasks,simulation_started=False,
        limitation='Split-LFC forcing sampled on coarse grid. Stability screen required; full-plate phase accuracy and speedup unvalidated.'))
    print(f'Prepared {len(tasks)} serial tasks; no simulation started. {root}',flush=True)


def validate_plan(plan):
    p=plan['substeps'];case=plan['case'];frequencies=plan['frequencies_hz']
    expected=['control-kernel','lts-kernel']
    if (plan.get('schema')!=3 or p!=2 or case!=cases()[0]
            or plan['configuration']!=queue.configuration()
            or plan['fine_dt_s']!=baseline.DURATION_S/case['steps']
            or plan['coarse_dt_s']!=p*plan['fine_dt_s']
            or plan['coarse_steps']*p!=case['steps'] or plan['source_sampling']!='coarse'
            or not frequencies or len(set(frequencies))!=len(frequencies)
            or any(type(f) is not int or f not in FREQUENCIES for f in frequencies)
            or plan['plate_condition']!='intact'
            or plan['edge_width_m']<max(plan['configuration']['plate_m'][:2])/case['cells']
            or plan['excitation_sources']!={str(f):excitation(f,case['steps'])[1] for f in frequencies}
            or plan['tasks']!=expected):
        raise ValueError('Prepared scientific settings or task order changed')
    if not np.isfinite(plan['edge_width_m']) or not 0<plan['device_budget_mib']<=65536:
        raise ValueError('Invalid partition or memory budget')
    for box in plan['boxes']:
        b=np.asarray(box)
        if b.shape!=(6,) or not np.isfinite(b).all() or np.any(b[::2]>b[1::2]):
            raise ValueError('Invalid fine region')


def reconstruct_method(root,task,plan):
    """Publish all six frequencies before advancing to the next method."""
    control=task=='control-kernel'
    steps=plan['case']['steps'] if control else plan['coarse_steps']
    dt=plan['fine_dt_s'] if control else plan['coarse_dt_s']
    folder=root/('normal-broadband' if control else 'lts-broadband')
    kernel=np.load(root/task/'trace.npz')['voltage_v']
    tick=time.perf_counter()
    for f in plan['frequencies_hz']:
        target=folder/f'{f//1000}khz';target.mkdir(parents=True,exist_ok=True)
        force,provenance=excitation(f,steps)
        t=np.arange(steps+1)*dt
        y=reconstruct(kernel,force(t))
        atomic_write(target/'waveforms.npz',lambda out:np.savez_compressed(out,time_s=t,voltage_v=y),overwrite=True)
        write_json(target/'source.json',provenance,overwrite=True)
        print(json.dumps(dict(phase='frequency_saved',method=folder.name,frequency_hz=f)),flush=True)
    write_json(folder/'reconstruction.json',dict(seconds=time.perf_counter()-tick,
        frequencies_hz=plan['frequencies_hz'],includes_output_writes=True),overwrite=True)


def run(root,stability_backend='auto'):
    plan=json.loads((root/'plan.json').read_text())
    if plan['identity']!=identity(): raise ValueError('Frozen sources/configuration changed')
    validate_plan(plan)
    with queue_lock(root):
        case,p,H=plan['case'],plan['substeps'],plan['coarse_dt_s']
        steps=plan['coarse_steps']
        h=plan['fine_dt_s']
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
        build_started=time.perf_counter()
        system,load,probes,required=queue.build(config,guard,event)
        write_json(root/'build.json',dict(seconds=time.perf_counter()-build_started),overwrite=True)
        system.screen=dict(system.screen,max_device_mib=plan.get('device_budget_mib',20000))
        fine=fine_partition(system,load,plan['boxes'],plan['edge_width_m'])
        write_json(root/'partition.json',dict(fine_dofs=int(fine.sum()),total_dofs=len(fine),
            edge_width_m=plan['edge_width_m'],fine_dt_s=h,coarse_dt_s=H,
            fine_boxes=plan['boxes'],plate_condition='intact'),overwrite=True)
        started=time.perf_counter()
        def stability_progress(details):
            write_json(root/'status.json',dict(status='checking_stability',**details),overwrite=True)
            event('stability_progress',**{k:v for k,v in details.items() if k!='phase'})
        try: stability=optimized_stability_screen(system,fine,H,p,progress=stability_progress,
            cache_dir=baseline.PROJECT/'outputs/stability-screen-cache',backend=stability_backend)
        except Exception as exc:
            write_json(root/'status.json',dict(status='preflight_failed',error=str(exc)),overwrite=True)
            raise
        write_json(root/'stability-screen.json',dict(**stability,seconds=time.perf_counter()-started),overwrite=True)
        event('time_steps',fine_dt_ns=h*1e9,coarse_dt_ns=H*1e9,
              baseline_global_updates=case['steps'],lts_global_updates=steps,
              source_sampling='coarse',stability=stability)
        def progress(state):
            elapsed=time.perf_counter()-tick
            state=dict(state,elapsed_seconds=elapsed,stopwatch=stopwatch(elapsed))
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
                if (saved.get('completed') and (folder/'trace.npz').is_file()
                        and saved.get('sha256')==sha256_file(folder/'trace.npz')):
                    reconstruct_method(root,task,plan)
                    continue
            # Preserve partial task metadata before retrying the task from t=0.
            stamp=str(time.time_ns())
            for name in ('trace.npz','result.json'):
                path=folder/name
                if path.exists(): path.rename(folder/(name+'.interrupted-'+stamp))
            write_json(receipt,dict(completed=False,status='running'))
            tick=time.perf_counter()
            try:
                if task=='control-kernel': kind,number='control',0
                elif task=='lts-kernel': kind,number='kernel',0
                else: kind,number=task.split('-');number=int(number)
                if kind=='kernel':source=interior_impulse(H)
                elif kind=='control':source=interior_impulse(h)
                else:
                    source=baseline.source(number,steps)[0]
                if kind=='control':
                    sim=gpu_explicit_simulate(system,case['steps'],h,probes,load,force=source,
                        output_stride=1,energy_stride=256,progress=progress,max_workspace_mib=4096,guard=guard)
                    t,y,stats=sim.time_s,sim.displacement_m,sim.statistics
                    stats=dict(stats,energy_time_s=sim.energy_time_s.tolist(),modified_energy_j=sim.modified_energy_j.tolist())
                else:
                    t,y,stats=gpu_lts_simulate(system,steps,H,probes,load,force=source,
                        fine=fine,substeps=p,progress=progress,guard=guard,
                        source_sampling='coarse',
                        max_workspace_mib=plan.get('device_budget_mib',20000))
                # Validate invariance only after the numerical source is over.
                source_end=h if kind=='control' else H if kind=='kernel' else 5/number
                tail=np.asarray(stats['modified_energy_j'])[np.asarray(stats['energy_time_s'])>source_end+2*H]
                drift=float(np.ptp(tail)/abs(tail[0])) if len(tail)>1 and tail[0]!=0 else None
                stats['unforced_relative_energy_drift']=drift
                if drift is not None and drift>1e-8: raise ArithmeticError('Unforced discrete energy drift exceeds 1e-8')
                atomic_write(folder/'trace.npz',lambda f:np.savez_compressed(f,time_s=t,voltage_v=y))
                write_json(receipt,dict(completed=True,seconds=time.perf_counter()-tick,
                    statistics=stats,sha256=sha256_file(folder/'trace.npz')),overwrite=True)
                reconstruct_method(root,task,plan)
                event('simulation_finished',task=task,seconds=time.perf_counter()-tick,
                      stopwatch=stopwatch(time.perf_counter()-tick))
            except BaseException as exc:
                write_json(receipt,dict(completed=False,status='interrupted_or_failed',error=str(exc)),overwrite=True)
                write_json(root/'status.json',dict(status='interrupted_or_failed',task=task),overwrite=True)
                raise
        rows=[]
        for f in plan['frequencies_hz']:
            reconstructed=np.load(root/'lts-broadband'/f'{f//1000}khz'/'waveforms.npz')['voltage_v']
            original_full=np.load(root/'normal-broadband'/f'{f//1000}khz'/'waveforms.npz')['voltage_v']
            original=original_full[::p]
            metrics=dict(frequency_hz=f,comparison_dt_s=H,
                         reference='normal broadband; numerical reference only',
                         local_vs_original_broadband=paired_metrics(original,reconstructed))
            metrics['accuracy_pass_1_percent']=within_target(metrics['local_vs_original_broadband'])
            rows.append(metrics);event('comparison',**metrics)
            write_json(root/'lts-broadband'/f'{f//1000}khz'/'comparison.json',metrics,overwrite=True)
        write_json(root/'comparisons.json',rows,overwrite=True)
        timing={task:json.loads((root/task/'result.json').read_text())['seconds'] for task in plan['tasks']}
        write_json(root/'timings.json',timing,overwrite=True)
        original_seconds=timing['control-kernel']+json.loads((root/'normal-broadband'/'reconstruction.json').read_text())['seconds']
        local_seconds=timing['lts-kernel']+json.loads((root/'lts-broadband'/'reconstruction.json').read_text())['seconds']
        screen_seconds=json.loads((root/'stability-screen.json').read_text())['seconds']
        write_json(root/'performance.json',dict(
            original_broadband_seconds=original_seconds,lts_broadband_seconds=local_seconds,
            original_over_lts_speedup=original_seconds/local_seconds,
            original_over_lts_speedup_including_screen=original_seconds/(local_seconds+screen_seconds),
            accuracy_pass=all(row['accuracy_pass_1_percent'] for row in rows),
            shared_operator_build_seconds=json.loads((root/'build.json').read_text())['seconds'],
            stability_screen_seconds=screen_seconds,
            scope='One kernel plus all frequency reconstructions/output writes per method; build and stability screening reported separately.'),overwrite=True)
        atomic_write(root/'all-results.txt',lambda f:f.write(json.dumps(dict(comparisons=rows,timings=timing),indent=2).encode()),overwrite=True)
        lines=['Local time stepping compared with normal broadband (numerical reference).',
               'Raw voltage error: lower is better. Pearson r: +1 is best. No fitted alignment.',
               f"Normal broadband plate solve stopwatch: {stopwatch(timing['control-kernel'])} ({timing['control-kernel']:.3f} s)",
               f"LTS broadband plate solve stopwatch: {stopwatch(timing['lts-kernel'])} ({timing['lts-kernel']:.3f} s)",
               f'Normal broadband including six frequency reconstructions: {stopwatch(original_seconds)} ({original_seconds:.3f} s)',
               f'LTS broadband including six frequency reconstructions: {stopwatch(local_seconds)} ({local_seconds:.3f} s)',
               f'Shared plate/PZT construction: {stopwatch(json.loads((root/"build.json").read_text())["seconds"])}',
               f'Stability check: {stopwatch(screen_seconds)} ({screen_seconds:.3f} s)',
               'Each method solves the plate once; all six frequencies share that solve time.']
        for row in rows:
            for channel,values in row['local_vs_original_broadband'].items():
                error=values['raw_l2_change_percent'];corr=values['correlation']
                lines.append(f"{row['frequency_hz']//1000} kHz | {channel} | raw voltage error: "
                    +(f'{error:.6g}%' if error is not None else 'undefined')
                    +' | waveform Pearson r: '+(f'{corr:.8g}' if corr is not None else 'undefined'))
        summary='\n'.join(lines)+'\n'
        atomic_write(root/'summary.txt',lambda f:f.write(summary.encode()),overwrite=True)
        print(summary,flush=True)
        write_json(root/'status.json',dict(status='completed'),overwrite=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare',action='store_true');mode.add_argument('--run',action='store_true')
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--substeps',type=int,default=2)
    parser.add_argument('--frequencies',type=int,nargs='+',default=FREQUENCIES)
    parser.add_argument('--edge-width-m',type=float,default=0.)
    parser.add_argument('--device-budget-mib',type=float,default=20000,
                        help='Full study defaults to a cloud GPU budget; actual free VRAM is still checked')
    parser.add_argument('--fine-box',type=float,nargs=6,action='append',default=[])
    parser.add_argument('--stability-backend',choices=['auto','cpu','gpu','original'],default='auto',
                        help='Screening only; original uses the unmodified historical checker')
    args=parser.parse_args()
    if args.prepare:
        if not args.frequencies or any(f not in FREQUENCIES for f in args.frequencies):
            raise ValueError('Use supported benchmark frequencies')
        prepare(args.root,args.substeps,args.frequencies,args.edge_width_m,args.fine_box,args.device_budget_mib)
    else:
        def interrupted(signum,frame): raise KeyboardInterrupt('Simulation interrupted by termination signal')
        previous=signal.signal(signal.SIGTERM,interrupted)
        try: run(args.root,stability_backend=args.stability_backend)
        finally: signal.signal(signal.SIGTERM,previous)


if __name__=='__main__': main()
