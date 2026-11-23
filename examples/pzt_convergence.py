"""Resumable full-plate numerical convergence audit; no experimental fitting."""
import argparse
import copy
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import numpy as np
import psutil

from examples import resolved_pzt_queue as queue
from examples.embedded_wave import ThermalPause
from examples.experimental_five import PROJECT, PeakGuard, record
from examples.plate_outputs import queue_lock, archive_attempt
from traxis.gpu_enriched_verlet import gpu_explicit_simulate
from traxis.resources import ResourceLimits
from traxis.storage import write_json, atomic_write, sha256_file
from examples.pzt_sampling import antialiased_samples


def cases():
    # Time step is identical across the isolated space/body studies. All sample
    # times coincide: output stride doubles when step count doubles.
    return [dict(name='baseline',cells=250,body=[32,6,3],steps=65536),
        *[dict(name=f'time-{factor}',cells=250,body=[32,6,3],steps=65536*factor)
          for factor in (2,4)],
        *[dict(name=f'plate-{n}',cells=n,body=[32,6,3],steps=131072)
          for n in (200,230,260)],
        *[dict(name=f'plate-thickness-{nz}',cells=190,nz=nz,body=[32,6,3],steps=131072)
          for nz in (1,2)],
        *[dict(name='body-'+name,cells=250,body=resolution,steps=131072)
          for name,resolution in [('radial',[32,7,3]),('angular',[40,6,3]),('thickness',[32,6,4])]],
        dict(name='coupled-fine',cells=260,body=[32,7,3],steps=262144)]


def fingerprints():
    paths=[*sorted((PROJECT/'traxis').glob('*.py')),Path(__file__),
        PROJECT/'examples/resolved_pzt_queue.py',PROJECT/'examples/experimental_five.py',
        PROJECT/'examples/embedded_wave.py',PROJECT/'examples/plate_outputs.py',
        PROJECT/'examples/pzt_sampling.py',queue.CONFIG,
        PROJECT/'configs/pzt/nce51-noliac-ver1404.json']
    return {str(p.relative_to(PROJECT)).replace('\\','/'):sha256_file(p) for p in paths}


def metrics(a,b):
    """Raw paired voltage changes; no shift, gain or polarity fitting."""
    rows={}
    for name,x,y in [('top',a[:,0],b[:,0]),('bottom',a[:,1],b[:,1]),
                     ('S0-related',a[:,0]+a[:,1],b[:,0]+b[:,1]),
                     ('A0-related',a[:,0]-a[:,1],b[:,0]-b[:,1])]:
        norm=float(np.linalg.norm(y));change=float(np.linalg.norm(x-y))
        rows[name]=dict(raw_l2_change_percent=100*change/norm if norm>1e-20 else None,
                       difference_l2_v=change,reference_l2_v=norm,
                       max_abs_change_v=float(np.max(abs(x-y))))
    return rows


def energy_tail(time_s,energy_j,drive_time_s,drive,dt_s):
    """Check the unforced interval after the *interpolated* drive is zero.

    np.interp ramps a last nonzero sample down to the next zero sample. That
    final ramp still does work; it must not enter an unforced energy check.
    """
    active=np.flatnonzero(np.asarray(drive)!=0.)
    end=(float(drive_time_s[min(int(active[-1])+1,len(drive_time_s)-1)])
         if len(active) else float(drive_time_s[0]))
    tail=np.asarray(energy_j)[np.asarray(time_s)>end+2*dt_s]
    drift=float(np.ptp(tail)/abs(tail[0])) if len(tail)>1 and tail[0]!=0 else None
    return dict(energy_relative_drift=drift,drive_support_end_s=end,
                post_drive_energy_samples=len(tail))


def worker_alive(folder):
    owner_path=folder/'owner.json'
    if not owner_path.exists():return False
    owner=json.loads(owner_path.read_text())
    try:
        process=psutil.Process(owner['pid'])
        if 'started_unix' in owner:
            return process.is_running() and abs(process.create_time()-owner['started_unix'])<1.
        # Legacy receipts recorded only PID. Check command identity before
        # treating that PID as this audit's worker.
        command=process.cmdline()
        return ('examples.pzt_convergence' in command and '--worker' in command
                and str(folder.parent.parent) in command
                and folder.name in command)
    except psutil.NoSuchProcess:return False
    except psutil.AccessDenied:
        raise RuntimeError('Cannot confirm whether the previous worker is still alive: '+str(folder))


def prepare_case(folder):
    """Skip verified successes; preserve and restart interrupted attempts."""
    if worker_alive(folder):raise RuntimeError('Previous worker is still running: '+str(folder))
    result=folder/'result.json'
    if result.exists():
        saved=json.loads(result.read_text())
        if saved.get('completed'):
            if sha256_file(folder/'waveforms.npz')!=saved['waveforms_sha256']:
                raise ValueError('Completed waveform checksum changed: '+str(folder))
            return False
        error=saved.get('error','')
        if error and not error.startswith(('KeyboardInterrupt','SystemExit')):
            raise RuntimeError('Failed numerical case needs investigation before retry: '+str(folder))
    archive_attempt(folder)
    return True


def assess(root):
    plan=json.loads((root/'plan.json').read_text());report=[]
    def read(frequency,name):
        folder=root/f'{frequency//1000}khz'/name
        result_path=folder/'result.json'
        if not result_path.exists():return None
        result=json.loads(result_path.read_text())
        if not result.get('completed'):return None
        if sha256_file(folder/'waveforms.npz')!=result['waveforms_sha256']:
            raise ValueError('Saved waveform changed: '+str(folder))
        with np.load(folder/'waveforms.npz') as d:return d['time_s'].copy(),d['voltage_v'].copy()
    pairs=[('temporal coarse','baseline','time-2'),('temporal fine','time-2','time-4'),
           ('spatial coarse','plate-200','plate-230'),('spatial fine','plate-230','plate-260'),
           ('production spatial','time-2','plate-260'),
           ('plate thickness','plate-thickness-1','plate-thickness-2'),
           *[('PZT '+name,'time-2','body-'+name) for name in ('radial','angular','thickness')],
           ('coupled production','baseline','coupled-fine')]
    for frequency in plan['frequencies_hz']:
        for label,first,second in pairs:
            a=read(frequency,first);b=read(frequency,second)
            if a is None or b is None:continue
            if len(a[0])!=len(b[0]) or not np.allclose(a[0],b[0],rtol=0,atol=1e-15):
                raise ValueError('Unmatched output time grids')
            changes=metrics(a[1],b[1])
            grid=np.arange(1750)/5000000
            sampled=metrics(antialiased_samples(a[0],a[1],grid),antialiased_samples(b[0],b[1],grid))
            passed=all(v['raw_l2_change_percent'] is not None and v['raw_l2_change_percent']<1.
                       for v in changes.values())
            report.append(dict(frequency_hz=frequency,study=label,first=first,second=second,
                               channels=changes,within_one_percent=passed,
                               anti_aliased_5mhz_channels=sampled))
    for frequency in plan['frequencies_hz']:
        traces=[read(frequency,name) for name in ('baseline','time-2','time-4')]
        if all(t is not None for t in traces):
            d1=float(np.linalg.norm(traces[0][1]-traces[1][1]));d2=float(np.linalg.norm(traces[1][1]-traces[2][1]))
            ratio=d1/d2 if d2>1e-20 else None
            report.append(dict(frequency_hz=frequency,study='temporal observed order',
                difference_ratio=ratio,order=math.log2(ratio) if ratio and ratio>0 else None,
                second_order_supported=bool(ratio is not None and 3.5<ratio<4.5)))
            grid=np.arange(1750)/5000000
            aligned=[antialiased_samples(t,v,grid) for t,v in traces]
            d1=float(np.linalg.norm(aligned[0]-aligned[1]));d2=float(np.linalg.norm(aligned[1]-aligned[2]))
            ratio=d1/d2 if d2>1e-20 else None
            report.append(dict(frequency_hz=frequency,study='temporal order, anti-aliased 5 MHz band',
                difference_ratio=ratio,order=math.log2(ratio) if ratio and ratio>0 else None,
                second_order_supported=bool(ratio is not None and 3.5<ratio<4.5)))
    complete=all(read(f,c['name']) is not None for f in plan['frequencies_hz'] for c in plan['cases'])
    answer=dict(completed=complete,records=report,threshold_percent=1.,
        scope='Full 1m plate, 350 microseconds, unaltered physics and drive; raw voltage comparisons.',
        limitations=['One-percent target is an engineering criterion, not a physical accuracy guarantee.',
          'Related-mode traces are sums/differences, not proven isolated modes.',
          'Finite successive refinements cannot prove convergence to the exact solution.',
          'In-plane series uses Q5. Thickness pair holds 190x190 in-plane cells fixed; no polynomial-order study.',
          'Body refinements are bounded by 1024 nodes; no absolute error bound.'])
    answer['limitations'].append('5 MHz sampled metrics remove frequencies above their Nyquist band; they do not establish full-band convergence or identify the actual NI filter.')
    write_json(root/'findings.json',answer,overwrite=True)
    lines=['PZT coupled-system convergence audit',f'Complete: {complete}',
           'Raw voltage changes; lower is better. Target: <1% in each channel.']
    for row in report:
        if 'channels' in row:
            values='; '.join(f'{k}: {v["raw_l2_change_percent"]:.4f}%' if v['raw_l2_change_percent'] is not None
                            else k+': undefined (zero reference)' for k,v in row['channels'].items())
            lines.append(f'{row["frequency_hz"]//1000} kHz | {row["study"]} | {values} | '+
                         ('within target' if row['within_one_percent'] else 'NEEDS REFINEMENT'))
            sampled=row['anti_aliased_5mhz_channels']
            lines.append('  Anti-aliased 5 MHz band: '+ '; '.join(
                f'{k}: {v["raw_l2_change_percent"]:.4f}%' if v['raw_l2_change_percent'] is not None
                else k+': undefined' for k,v in sampled.items()))
        else:lines.append(f'{row["frequency_hz"]//1000} kHz | {row["study"]}: {row["order"]}')
    lines.extend(answer['limitations'])
    if complete:
        lines.append('Review changes and observed order before adopting new default resolution. '
                     'No material parameters were fitted or changed.')
    atomic_write(root/'findings.txt',lambda f:f.write(('\n'.join(lines)+'\n').encode()),overwrite=True)
    return answer


def worker(root,frequency,name):
    plan=json.loads((root/'plan.json').read_text())
    if plan['source_sha256']!=fingerprints():raise ValueError('Frozen audit sources changed')
    case=next(c for c in plan['cases'] if c['name']==name)
    folder=root/f'{frequency//1000}khz'/name;folder.mkdir(parents=True,exist_ok=True)
    if (folder/'result.json').exists():raise FileExistsError(folder/'result.json')
    os.environ.setdefault('CUPY_CACHE_DIR',str(root/'cache'))
    started=time.perf_counter();last=-math.inf
    def event(phase,**details):
        print(json.dumps(dict(frequency_hz=frequency,case=name,phase=phase,
                             wall_seconds=time.perf_counter()-started,**details)),flush=True)
    result=dict(completed=False,case=case,frequency_hz=frequency)
    try:
        config=copy.deepcopy(plan['configuration'])
        config['mesh']['cells']=[case['cells'],case['cells'],case.get('nz',1)]
        config['body'].update(zip(('sectors','rings','layers'),case['body']))
        config['mesh']['minimum_steps']=case['steps']
        guard=PeakGuard(ResourceLimits(**{k:None for k in config['resource_limits']}),telemetry_errors_fatal=False)
        guard.check(force=True)
        cooling=ThermalPause(guard,82.,78.,allow_missing=True)
        system,load,probes,required=queue.build(config,guard,event)
        if case['steps']<required:raise ValueError('Chosen common time step is not stable for this mesh')
        row=record(frequency);drive=np.asarray(row['nifgen_avg_waveform_ch1'])
        drive_t=np.arange(len(drive))/row['sampleRate']
        force=queue.actuator_signal(None,drive_t,drive)
        def progress(state):
            nonlocal last
            cooling.check(state['steps_completed'])
            if time.perf_counter()-last>15 or state['steps_completed']==case['steps']:
                event('propagating',percent=100*state['steps_completed']/case['steps'],**state)
                last=time.perf_counter()
        simulation=gpu_explicit_simulate(system,case['steps'],.00035/case['steps'],probes,load,
            force=lambda t:float(force(t)),output_stride=8*(case['steps']//65536),
            energy_stride=256,guard=guard,progress=progress,max_workspace_mib=4096)
        atomic_write(folder/'waveforms.npz',lambda f:np.savez_compressed(f,
            time_s=simulation.time_s,voltage_v=simulation.displacement_m,
            energy_time_s=simulation.energy_time_s,modified_energy_j=simulation.modified_energy_j))
        check=energy_tail(simulation.energy_time_s,simulation.modified_energy_j,
                          drive_t,drive,.00035/case['steps'])
        drift=check['energy_relative_drift']
        result.update(check)
        if drift is None or drift>1e-8:raise ArithmeticError('Lossless energy invariant failed')
        result.update(completed=True,waveforms_sha256=sha256_file(folder/'waveforms.npz'),
            energy_relative_drift=drift,stable_dt_s=system.stable_dt_s,
            statistics=simulation.statistics,resource_peaks=guard.peaks)
        event('completed',energy_relative_drift=drift)
    except BaseException as exc:
        result['error']=f'{type(exc).__name__}: {exc}'
        raise
    finally:
        result['wall_seconds']=time.perf_counter()-started
        write_json(folder/'result.json',result)


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--run',action='store_true')
    parser.add_argument('--assess',action='store_true')
    parser.add_argument('--worker',nargs=2,metavar=('HZ','CASE'))
    args=parser.parse_args();root=args.root.resolve()
    if not root.is_relative_to(PROJECT/'outputs'):raise ValueError('Audit outputs must stay in project outputs')
    if args.prepare:
        root.mkdir(parents=True,exist_ok=True)
        if any(root.glob('*khz/*/result.json')):
            raise FileExistsError('An audit with results already exists; use --run to resume')
        write_json(root/'plan.json',dict(configuration=queue.configuration(),cases=cases(),
            frequencies_hz=[250000,100000],source_sha256=fingerprints(),
            physical_parameters_fitted=False),overwrite=True)
        assess(root);print('Prepared:',root)
    elif args.worker:worker(root,int(args.worker[0]),args.worker[1])
    elif args.assess:print(json.dumps(assess(root),indent=2))
    elif args.run:
        with queue_lock(root):
            plan=json.loads((root/'plan.json').read_text())
            if plan['source_sha256']!=fingerprints():raise ValueError('Prepared sources changed')
            for folder in root.glob('*khz/*'):
                if folder.is_dir() and worker_alive(folder):
                    raise RuntimeError('An audit worker is still running: '+str(folder))
            write_json(root/'status.json',dict(status='running',pid=os.getpid()),overwrite=True)
            try:
                for frequency in plan['frequencies_hz']:
                    for case in plan['cases']:
                        folder=root/f'{frequency//1000}khz'/case['name'];folder.mkdir(parents=True,exist_ok=True)
                        if not prepare_case(folder):continue
                        print(f'Starting {frequency//1000} kHz {case["name"]}',flush=True)
                        with (folder/'worker.log').open('a',encoding='utf-8') as log:
                            process=subprocess.Popen([sys.executable,'-m','examples.pzt_convergence','--root',str(root),
                                '--worker',str(frequency),case['name']],cwd=PROJECT,stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace')
                            try:started_unix=psutil.Process(process.pid).create_time()
                            except psutil.NoSuchProcess:started_unix=None
                            owner=dict(pid=process.pid)
                            if started_unix is not None:owner['started_unix']=started_unix
                            write_json(folder/'owner.json',owner,overwrite=True)
                            for line in process.stdout:
                                log.write(line);log.flush();print(line,end='',flush=True)
                            code=process.wait()
                        assess(root)
                        if code:raise RuntimeError(f'Case failed (exit {code}): {folder}')
                        print((root/'findings.txt').read_text(),flush=True)
                write_json(root/'status.json',dict(status='completed'),overwrite=True)
            except BaseException as exc:
                write_json(root/'status.json',dict(status='interrupted',error=str(exc)),overwrite=True)
                raise
    else:parser.error('Choose --prepare, --run, --worker or --assess')


if __name__=='__main__':main()
