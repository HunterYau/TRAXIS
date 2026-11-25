"""Prepared, resumable combined-only PZT mesh study. --run is always explicit."""
import argparse
import copy
import csv
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

import numpy as np
import psutil
from examples import pzt_mesh_protocol as protocol
from examples import pzt_mesh_metrics as metrics
from examples.pzt_broadband import interior_impulse,reconstruct
from examples.pzt_local_time_study import excitation,stopwatch
from examples.experimental_five import PeakGuard
from examples.plate_outputs import queue_lock
from traxis.config import Material
from traxis.gpu_tensor import build_gpu_tensor
from traxis.gpu_pzt_body import couple_gpu_bodies
from traxis.gpu_enriched_verlet import sparse_probe_set
from traxis.gpu_local_time import gpu_lts_simulate
from traxis.local_time import fine_partition
from traxis.lts_stability import optimized_stability_screen
from traxis.pzt_body import attach_disk
from traxis.pzt_receiver import DiskGeometry
from traxis.piezoelectric import PiezoMaterial,read_profile
from traxis.resources import ResourceLimits
from traxis.storage import write_json,atomic_write,sha256_file,canonical_hash


def read(path):return json.loads(Path(path).read_text())


def save(path,data):write_json(Path(path),data,overwrite=True)


def wave(path):
    with np.load(path,allow_pickle=False) as a:return a['time_s'].copy(),a['voltage_v'].copy()


def completed(folder):
    path=folder/'result.json'
    if not path.exists():return False
    r=read(path)
    if not r.get('completed'):return False
    required={'comparisons.json','build.json','time-plan.json'}
    for stage in ('primary','half-step'):
        required.update((stage+'/trace.npz',stage+'/result.json'))
        required.update(f'{stage}/{f//1000}khz/waveforms.npz' for f in (50000,100000,150000,200000,250000,300000))
    for f in (50000,100000,150000,200000,250000,300000):
        required.update(f'plots/{f//1000}khz/{relation}/plots.json' for relation in ('vs-baseline','temporal'))
    if not required.issubset(r['files']) or r['configuration']['id']!=folder.name:
        raise ValueError('Completion receipt is missing required study artifacts')
    for name,digest in r['files'].items():
        if sha256_file(folder/name)!=digest:raise ValueError('Completed artifact checksum changed: '+str(folder/name))
    return True


def available_host_bytes():
    available=psutil.virtual_memory().available
    for root,q,u in (('/sys/fs/cgroup','memory.max','memory.current'),
                     ('/sys/fs/cgroup/memory','memory.limit_in_bytes','memory.usage_in_bytes')):
        try:
            limit=int((Path(root)/q).read_text());used=int((Path(root)/u).read_text())
            available=min(available,max(0,limit-used))
        except (OSError,ValueError):pass
    return available


def build(plan,row,guard,event):
    import cupy as cp
    e=row['estimate'];c=plan['configuration']
    if e['estimated_host_peak_bytes']>.8*available_host_bytes():
        raise MemoryError('Mesh host-memory forecast exceeds 80% of currently available allocation; use a larger cloud host')
    if e['estimated_gpu_peak_bytes']>.8*cp.cuda.runtime.memGetInfo()[0]:
        raise MemoryError('Mesh GPU-memory forecast exceeds 80% of currently free VRAM')
    if e['estimated_gpu_peak_bytes']>plan['protocol']['device_budget_mib']*1024**2:
        raise MemoryError('Mesh forecast exceeds declared GPU budget')
    layout=protocol.layout_for(c)
    # Budget metadata only; coordinates, quadrature, order and plate stay fixed.
    layout.screen['max_device_mib']=plan['protocol']['device_budget_mib']
    base=build_gpu_tensor(layout,Material(**c['host_material']),guard=guard)
    material=PiezoMaterial.from_profile(read_profile(protocol.PROJECT/c['material_profile']))
    bodies=[];next_dof=layout.screen['vector_dofs']
    for center,surface,electrical in ((c['actuator_center_xy_m'],'top','voltage'),
        (c['receiver_center_xy_m'],'top','open'),(c['receiver_center_xy_m'],'bottom','open')):
        event('assembling_pzt',body=len(bodies)+1)
        if e['attachment_workspace_bytes']>.8*available_host_bytes():
            raise MemoryError('Insufficient available RAM for next dense PZT attachment')
        b=attach_disk(layout,material,DiskGeometry(tuple(center),c['disk_radius_m'],c['disk_thickness_m'],surface,1),
            first_dof=next_dof,electrical=electrical,bond=c['body']['bond'],input_capacitance_f=0.,
            sectors=row['angular'],rings=row['radial'],layers=row['thickness'],guard=guard,
            max_workspace_mib=plan['protocol']['component_workspace_mib'],max_component_nodes=8192,max_component_elements=65536)
        bodies.append(b);next_dof=b.next_dof
    event('coupling_pzts')
    if e['merge_workspace_bytes']>.8*available_host_bytes():raise MemoryError('Insufficient RAM for consistent coupled mass blocks')
    system=couple_gpu_bodies(base,bodies,guard=guard,max_workspace_mib=plan['protocol']['component_workspace_mib'])
    if len(system.mass_diagonal)!=e['total_mechanical_dofs']:raise ArithmeticError('DOF forecast mismatch')
    if [len(p.indices) for p in system.body_patches]!=e['patch_dofs']:raise ArithmeticError('PZT support forecast mismatch')
    force=np.zeros(next_dof);force[bodies[0].indices]=-bodies[0].coupling
    probes=sparse_probe_set([(b.indices,b.coupling/b.total_capacitance_f) for b in bodies[1:]],next_dof)
    return system,force,probes


def select_time(plan,system,fine,event,backend,cache):
    H=plan['baseline_coarse_dt_s'];attempts=[]
    while H/2>.8*system.stable_dt_s:H/=2
    while True:
        steps=round(plan['duration_s']/H)
        if steps*2>500000:raise ValueError('Primary plus half-step verification exceeds the unchanged solver step limit')
        try:
            proof=optimized_stability_screen(system,fine,H,2,progress=lambda x:event('stability',**x),backend=backend,cache_dir=cache)
            return dict(coarse_dt_s=H,fine_dt_s=H/2,primary_steps=steps,half_step_steps=2*steps,
                full_system_stable_dt_s=system.stable_dt_s,stability=proof,rejected_step_attempts=attempts)
        except ArithmeticError as exc:
            if 'fails effective-operator stability screen' not in str(exc):raise
            attempts.append(dict(coarse_dt_s=H,error=str(exc)));event('coarse_step_rejected',coarse_dt_s=H)
            H/=2


def reconstruct_stage(folder,stage,frequencies):
    t,kernel=wave(folder/stage/'trace.npz')
    for frequency in frequencies:
        dest=folder/stage/f'{frequency//1000}khz';dest.mkdir(parents=True,exist_ok=True)
        y=reconstruct(kernel,excitation(frequency,len(t)-1)[0](t))
        atomic_write(dest/'waveforms.npz',lambda f:np.savez_compressed(f,time_s=t,voltage_v=y),overwrite=True)


def analysis(folder,plan,row,root):
    config=plan['configuration'];baseline_rows=[];temporal_rows=[];previous_rows=[]
    previous=next((c for c in plan['cases'] if c['family']==row['family'] and c['level']==row['level']-1),None)
    for frequency in plan['protocol']['frequencies_hz']:
        t,y=wave(folder/'primary'/f'{frequency//1000}khz/waveforms.npz')
        th,yh=wave(folder/'half-step'/f'{frequency//1000}khz/waveforms.npz')
        tb,yb=wave(root/'baseline/lts-broadband'/f'{frequency//1000}khz/waveforms.npz')
        baseline_rows.append(metrics.compare(tb,yb,t,y,frequency,config))
        temporal_rows.append(metrics.compare(t,y,th,yh,frequency,config))
        if previous and completed(root/'cases'/previous['id']):
            tp,yp=wave(root/'cases'/previous['id']/'primary'/f'{frequency//1000}khz/waveforms.npz')
            previous_rows.append(metrics.compare(tp,yp,t,y,frequency,config))
        elif row['level']==1:previous_rows.append(baseline_rows[-1])
        for name,ta,ya,tz,yz in (('vs-baseline',tb,yb,t,y),('temporal',t,y,th,yh)):
            plots=folder/'plots'/f'{frequency//1000}khz'/name
            if (plots/'plots.json').exists():
                for path,digest in read(plots/'plots.json')['files'].items():
                    if sha256_file(plots/path)!=digest:raise ValueError('Saved plot checksum mismatch')
            else:
                if plots.exists():plots.rename(plots.with_name(name+'.interrupted-'+str(time.time_ns())))
                paths=metrics.plots(plots,ta,ya,tz,yz,f"{row['id']} {frequency//1000} kHz {name}")
                save(plots/'plots.json',dict(files={p:sha256_file(plots/p) for p in paths}))
    result=dict(vs_baseline=baseline_rows,temporal=temporal_rows,vs_previous=previous_rows,
        temporal_target_percent=plan['protocol']['temporal_target_percent'],
        temporal_qualified=metrics.temporal_pass(temporal_rows,plan['protocol']['temporal_target_percent']),
        baseline_temporal_qualified=False,
        interpretation='Thresholds apply to measured differences, not proof of exact solution error. Baseline timing discretization is fixed and not rerun; temporal-unresolved comparisons cannot certify spatial convergence.')
    save(folder/'comparisons.json',result)
    return result


def worker(root,case_id,backend='auto'):
    root=Path(root);plan=protocol.validate(root);row=next(r for r in plan['cases'] if r['id']==case_id)
    folder=root/'cases'/case_id;folder.mkdir(parents=True,exist_ok=True)
    if completed(folder):return
    started=time.perf_counter();last=-math.inf;system=None
    guard=PeakGuard(ResourceLimits(**plan['configuration']['resource_limits']),telemetry_errors_fatal=False)
    def event(kind,**details):
        details.pop('phase',None)
        stage_elapsed=details.pop('elapsed_seconds',None)
        if stage_elapsed is not None:details['stage_elapsed_seconds']=stage_elapsed
        data=dict(status='running',case=case_id,phase=kind,elapsed_seconds=time.perf_counter()-started,**details)
        save(root/'status.json',data);print(json.dumps(data),flush=True)
    save(folder/'owner.json',dict(pid=os.getpid(),created_unix=psutil.Process().create_time()))
    try:
        event('building');guard.check(force=True)
        tick=time.perf_counter();system,force,probes=build(plan,row,guard,event)
        save(folder/'build.json',dict(seconds=time.perf_counter()-tick,dofs=len(system.mass_diagonal),
            screen=system.screen,body_reports=[b.report for b in system.body_attachments]))
        fine=fine_partition(system,force,plan['boxes'],plan['edge_width_m'])
        settings=select_time(plan,system,fine,event,backend,root/'stability-cache')
        if (folder/'time-plan.json').exists() and read(folder/'time-plan.json')['coarse_dt_s']!=settings['coarse_dt_s']:
            raise ValueError('Restart changed the selected time grid; retain results and investigate')
        save(folder/'time-plan.json',settings)
        for stage,factor in (('primary',1),('half-step',2)):
            dest=folder/stage;dest.mkdir(parents=True,exist_ok=True)
            H=settings['coarse_dt_s']/factor;steps=settings['primary_steps']*factor
            proof=optimized_stability_screen(system,fine,H,2,backend=backend,cache_dir=root/'stability-cache',
                progress=lambda x:event('stability',**x))
            identity=canonical_hash(dict(plan=canonical_hash(plan),case=case_id,stage=stage,H=H,
                model=proof.get('model_fingerprint'),stable_dt_s=system.stable_dt_s))
            if (dest/'result.json').exists() and read(dest/'result.json').get('completed'):
                receipt=read(dest/'result.json')
                if receipt['identity']!=identity or sha256_file(dest/'trace.npz')!=receipt['sha256']:
                    raise ValueError('Completed stage integrity mismatch; refusing rerun')
                event('reusing_completed_kernel',stage=stage)
            else:
                save(dest/'attempt.json',dict(started_unix=time.time(),steps=steps,coarse_dt_s=H))
                def progress(state):
                    nonlocal last
                    now=time.perf_counter()
                    if now-last>=2 or state['steps_completed']==steps:
                        last=now;event('propagating',stage=stage,percent=100*state['steps_completed']/steps,
                            steps_completed=state['steps_completed'],requested_steps=steps,
                            simulation_stopwatch=stopwatch(state['elapsed_seconds']))
                event('starting_kernel',stage=stage,coarse_dt_s=H,fine_dt_s=H/2,steps=steps)
                tick=time.perf_counter()
                t,y,stats=gpu_lts_simulate(system,steps,H,probes,force,force=interior_impulse(H),
                    fine=fine,substeps=2,source_sampling='coarse',progress=progress,guard=guard,
                    max_workspace_mib=plan['protocol']['device_budget_mib'])
                tail=np.asarray(stats['modified_energy_j'])[np.asarray(stats['energy_time_s'])>3*H]
                drift=float(np.ptp(tail)/abs(tail[0])) if len(tail)>1 and tail[0]!=0 else None
                if drift is None or drift>1e-8:raise ArithmeticError('Unforced energy conservation failed')
                atomic_write(dest/'trace.npz',lambda f:np.savez_compressed(f,time_s=t,voltage_v=y),overwrite=True)
                save(dest/'result.json',dict(completed=True,identity=identity,sha256=sha256_file(dest/'trace.npz'),
                    seconds=time.perf_counter()-tick,coarse_dt_s=H,fine_dt_s=H/2,steps=steps,
                    statistics=stats,unforced_energy_drift=drift,stability=proof))
            reconstruct_stage(folder,stage,plan['protocol']['frequencies_hz'])
        event('analyzing_and_plotting');comparison=analysis(folder,plan,row,root)
        guard.check(force=True)
        current_seconds=time.perf_counter()-started
        previous_seconds=sum(read(p).get('current_attempt_wall_seconds',read(p).get('total_wall_seconds',0.))
            for p in (folder/'attempts').glob('*/result.json'))
        files={p.relative_to(folder).as_posix():sha256_file(p) for p in folder.rglob('*')
               if p.is_file() and p.name not in ('result.json','worker.log','owner.json') and 'attempts' not in p.parts}
        # Include stage result receipts as well as their voltage arrays.
        for stage in ('primary','half-step'):files[stage+'/result.json']=sha256_file(folder/stage/'result.json')
        result=dict(completed=True,status='completed' if comparison['temporal_qualified'] else 'completed_temporal_unresolved',
            configuration=row,total_wall_seconds=current_seconds+previous_seconds,current_attempt_wall_seconds=current_seconds,
            primary_seconds=read(folder/'primary/result.json')['seconds'],half_step_seconds=read(folder/'half-step/result.json')['seconds'],
            dofs=len(system.mass_diagonal),fine_dofs=int(fine.sum()),
            resources=dict(sampled_peaks=getattr(guard,'peaks',{}),telemetry_error=guard.telemetry_error),
            resource_sampling_note='Process RSS sampled cooperatively; GPU total memory via nvidia-smi approximately every 2 s; these are sampled peaks, not exact allocation high-water marks.',
            temporal_qualified=comparison['temporal_qualified'],files=files)
        save(folder/'result.json',result);event('configuration_saved',result=result)
    except BaseException as exc:
        result=dict(completed=False,status='interrupted' if isinstance(exc,(KeyboardInterrupt,SystemExit)) else 'failed',
            configuration=row,total_wall_seconds=time.perf_counter()-started,current_attempt_wall_seconds=time.perf_counter()-started,
            error=type(exc).__name__+': '+str(exc),
            traceback=traceback.format_exc(),dofs=len(system.mass_diagonal) if system is not None else None,
            estimated_dofs=row['estimate']['total_mechanical_dofs'],
            frequency_channels=[dict(frequency_hz=f,channel=c,status='not_available',error=str(exc))
                for f in plan['protocol']['frequencies_hz'] for c in metrics.CHANNELS],
            resources=dict(sampled_peaks=getattr(guard,'peaks',{})))
        save(folder/'result.json',result);event('configuration_failed',error=result['error'])
        raise


def assess(root):
    root=Path(root);plan=protocol.validate(root);reports=[];flat=[];finest=[]
    for row in plan['cases']:
        folder=root/'cases'/row['id'];receipt=folder/'result.json'
        if not receipt.exists():
            reports.append(dict(configuration=row,status='pending',completed=False));continue
        report=read(receipt);reports.append(report)
        if completed(folder):
            comparison=read(folder/'comparisons.json')
            for relation in ('vs_baseline','vs_previous','temporal'):
                for f in comparison[relation]:
                    for channel,m in f['channels'].items():
                        flat.append(dict(configuration=row['id'],relation=relation,frequency_hz=f['frequency_hz'],channel=channel,
                            status=report['status'],error=None,
                            temporal_qualified=comparison['temporal_qualified'],
                            **{k:v for k,v in m.items() if k not in ('arrivals','thresholds')},
                            **{mode+'_'+field:m['arrivals'].get(mode,{}).get(field)
                               for mode in ('S0','A0') for field in ('difference_us','error_percent','reliable')},
                            pass_1_percent=all(m['thresholds']['1.0'].values()),pass_0_1_percent=all(m['thresholds']['0.1'].values())))
            reference=next(r for r in plan['cases'] if r['family']==row['family'] and r['level']==3)
            ref=root/'cases'/reference['id']
            if row['level']!=3 and completed(ref):
                for frequency in plan['protocol']['frequencies_hz']:
                    t,y=wave(ref/'primary'/f'{frequency//1000}khz/waveforms.npz')
                    tc,yc=wave(folder/'primary'/f'{frequency//1000}khz/waveforms.npz')
                    finest.append(dict(case=row['id'],reference=reference['id'],metrics=metrics.compare(t,y,tc,yc,frequency,plan['configuration'])))
        else:
            for f in plan['protocol']['frequencies_hz']:
                for channel in metrics.CHANNELS:
                    flat.append(dict(configuration=row['id'],relation='vs_baseline',frequency_hz=f,channel=channel,
                        status=report['status'],error=report.get('error','incomplete'),temporal_qualified=None,
                        **{key:None for key in ('raw_l2_percent','envelope_l2_percent','envelope_normalized_l2_percent',
                            'wiggle_l2_percent','raw_pearson','wiggle_pearson','envelope_pearson','rmse_v',
                            'max_absolute_voltage_difference_v','pass_1_percent','pass_0_1_percent')}))
                    flat[-1].update({mode+'_'+field:None for mode in ('S0','A0') for field in ('difference_us','error_percent','reliable')})
    output=dict(configurations=reports,finest_in_each_series=finest,baseline_rerun=False,
        optimal_mesh_selected=False,all_configurations_recorded=True,
        interpretation='No single mesh is selected. Numerical pairwise changes are not physical accuracy. Temporal-unresolved cases and failed cases are retained.')
    save(root/'all-results.json',output)
    if flat:
        with (root/'metrics.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(flat[0]));writer.writeheader();writer.writerows(flat)
    lines=['PZT mesh convergence results; no optimal mesh selected. Baseline reused.']
    for report in reports:
        c=report['configuration'];lines.append(f"{c['id']} R/A/T={c['radial']}/{c['angular']}/{c['thickness']} | {report['status']} | wall seconds={report.get('total_wall_seconds')} | DOFs={report.get('dofs')} | temporal qualified={report.get('temporal_qualified')}")
        if report.get('error'):lines.append(report['error'])
    for row in flat:
        lines.append(f"{row['configuration']} {row['relation']} {row['frequency_hz']//1000}kHz {row['channel']} | raw={row['raw_l2_percent']}% envelope={row['envelope_l2_percent']}% wiggle={row['wiggle_l2_percent']}% | Pearson raw={row['raw_pearson']} envelope={row['envelope_pearson']} | <1%={row['pass_1_percent']} <0.1%={row['pass_0_1_percent']}")
    (root/'all-results.txt').write_text('\n'.join(lines)+'\n')
    return output


def run(root,backend='auto',retry_failed=False):
    root=Path(root).resolve();plan=protocol.validate(root)
    with queue_lock(root):
        save(root/'scheduler.json',dict(pid=os.getpid(),created_unix=psutil.Process().create_time()))
        for row in plan['cases']:
            folder=root/'cases'/row['id'];folder.mkdir(parents=True,exist_ok=True)
            if completed(folder):continue
            prior=read(folder/'result.json') if (folder/'result.json').exists() else None
            if prior and prior['status']=='failed' and not retry_failed:continue
            if (folder/'owner.json').exists():
                owner=read(folder/'owner.json')
                try:
                    if abs(psutil.Process(owner['pid']).create_time()-owner['created_unix'])<1:
                        raise RuntimeError('A previous worker is still active; refusing duplicate calculation')
                except psutil.NoSuchProcess:pass
            if prior:
                attempt=folder/'attempts'/str(time.time_ns());attempt.mkdir(parents=True)
                (folder/'result.json').rename(attempt/'result.json')
                if (folder/'worker.log').exists():(folder/'worker.log').rename(attempt/'worker.log')
            save(root/'status.json',dict(status='running',case=row['id'],phase='launching_worker'))
            tick=time.perf_counter()
            with (folder/'worker.log').open('ab',buffering=0) as log:
                process=subprocess.Popen([sys.executable,'-u','-m','examples.pzt_mesh_study','--worker',row['id'],
                    '--root',str(root),'--stability-backend',backend],cwd=protocol.PROJECT,
                    stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
                try:code=process.wait()
                except KeyboardInterrupt:
                    process.terminate()
                    process.wait()
                    if not (folder/'result.json').exists():
                        save(folder/'result.json',dict(completed=False,status='interrupted',configuration=row,
                            total_wall_seconds=time.perf_counter()-tick,current_attempt_wall_seconds=time.perf_counter()-tick,
                            estimated_dofs=row['estimate']['total_mechanical_dofs'],
                            error='Worker interrupted; completed stage receipts retained.'))
                    assess(root)
                    save(root/'status.json',dict(status='interrupted',case=row['id'],note='Completed kernels retained; resume explicitly'))
                    raise
            if code and not (folder/'result.json').exists():
                save(folder/'result.json',dict(completed=False,status='failed',configuration=row,
                    total_wall_seconds=time.perf_counter()-tick,current_attempt_wall_seconds=time.perf_counter()-tick,
                    estimated_dofs=row['estimate']['total_mechanical_dofs'],
                    error=f'Worker exited with code {code}, possibly external termination/OOM; no completion claimed.'))
            assess(root)
        report=assess(root);pending=[r['configuration']['id'] for r in report['configurations'] if r['status']=='pending']
        failed=[r['configuration']['id'] for r in report['configurations'] if not r.get('completed')]
        save(root/'status.json',dict(status='completed_with_failures' if failed else 'completed',failed=failed,pending=pending,
            completed=[r['configuration']['id'] for r in report['configurations'] if r.get('completed')]))


def main():
    parser=argparse.ArgumentParser(__doc__);mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare',action='store_true');mode.add_argument('--preflight',action='store_true')
    mode.add_argument('--run',action='store_true');mode.add_argument('--worker')
    mode.add_argument('--report',action='store_true');parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--baseline',type=Path);parser.add_argument('--assembly-sources',type=Path)
    parser.add_argument('--retry-failed',action='store_true')
    parser.add_argument('--stability-backend',choices=['auto','gpu','cpu','original'],default='auto')
    args=parser.parse_args()
    def interrupt(signum,frame):raise KeyboardInterrupt('Worker interrupted; completed kernels are retained')
    signal.signal(signal.SIGTERM,interrupt)
    if args.prepare:
        if args.baseline is None or args.assembly_sources is None:parser.error('Preparation needs --baseline and --assembly-sources')
        protocol.prepare(args.root,args.baseline,args.assembly_sources)
    elif args.preflight:
        p=protocol.validate(args.root);print(json.dumps(dict(ready=True,configurations=len(p['cases']),baseline_rerun=False,full_simulations_started=0)))
    elif args.worker:worker(args.root,args.worker,args.stability_backend)
    elif args.report:assess(args.root)
    else:run(args.root,args.stability_backend,args.retry_failed)


if __name__=='__main__':main()
