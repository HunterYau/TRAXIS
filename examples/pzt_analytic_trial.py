"""Two approved 250 kHz source-reconstruction runs, saved and assessed serially."""
import argparse
import copy
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import psutil
from scipy.signal import periodogram

from examples import resolved_pzt_queue as queue
from examples.embedded_wave import ThermalPause
from examples.experimental_five import PROJECT, PeakGuard, record
from examples.plate_outputs import queue_lock
from examples.pzt_convergence import prepare_case, worker_alive, metrics
from examples.pzt_analytic_source import analytic_source
from examples.pzt_sampling import antialiased_samples
from traxis.gpu_enriched_verlet import gpu_explicit_simulate
from traxis.resources import ResourceLimits
from traxis.storage import atomic_write, write_json, sha256_file


def identity():
    proof=queue.preflight()
    sources=proof['source_sha256'].copy()
    for name in ('examples/pzt_analytic_trial.py','examples/pzt_analytic_source.py',
                 'examples/pzt_convergence.py'):
        sources[name]=sha256_file(PROJECT/name)
    return dict(production_preflight=proof,source_sha256=sources)


def prepare(root,reference):
    if root.exists():raise FileExistsError('Use --run to resume an existing trial')
    config=queue.configuration()
    old_plan=json.loads((reference/'plan.json').read_text())
    if old_plan['configuration']!=config:raise ValueError('Reference physics differs')
    row=record(250000);drive=np.asarray(row['nifgen_avg_waveform_ch1'])
    _,source=analytic_source(250000,5,np.arange(len(drive))/row['sampleRate'],drive)
    cases=[]
    for name,control,steps in [('analytic-original-dt','baseline',65536),('analytic-half-dt','time-2',131072)]:
        folder=reference/'250khz'/control;result=json.loads((folder/'result.json').read_text())
        if not result['completed'] or sha256_file(folder/'waveforms.npz')!=result['waveforms_sha256']:
            raise ValueError('Missing or changed completed control: '+str(folder))
        cases.append(dict(name=name,steps=steps,control_directory=str(folder),
                          control_waveforms_sha256=result['waveforms_sha256']))
    frozen=identity()
    root.mkdir(parents=True)
    write_json(root/'plan.json',dict(configuration=config,frequency_hz=250000,cases=cases,
        identity=frozen,source=source,comparison='Unfiltered full-band first; 5 MHz band separately labelled',
        approval='User approved exactly these two source-reconstruction simulations.'))
    import zipfile
    with zipfile.ZipFile(root/'frozen-source.zip','w',compression=zipfile.ZIP_DEFLATED) as archive:
        for name in frozen['source_sha256']:archive.write(PROJECT/name,name)
        archive.write(root/'plan.json','plan.json')
    assess(root)
    print('Prepared two 250 kHz runs:',root,flush=True)


def saved_trace(folder,checksum=None):
    if checksum is None:
        r=folder/'result.json'
        if not r.exists():return None
        result=json.loads(r.read_text())
        if not result.get('completed'):return None
        checksum=result['waveforms_sha256']
    path=folder/'waveforms.npz'
    if sha256_file(path)!=checksum:raise ValueError('Changed saved trace: '+str(path))
    with np.load(path) as data:return data['time_s'].copy(),data['voltage_v'].copy()


def spectrum(time_s,voltage):
    f,p=periodogram(voltage[:,0]-voltage[:,1],fs=1/(time_s[1]-time_s[0]))
    df=f[1]-f[0]
    return dict(a0_related_power_4_5_to_5_5mhz_v2=float(p[(f>=4.5e6)&(f<=5.5e6)].sum()*df),
                a0_related_power_above_500khz_v2=float(p[f>500000].sum()*df))


def assess(root):
    plan=json.loads((root/'plan.json').read_text());rows=[];traces=[]
    for case in plan['cases']:
        trial=saved_trace(root/case['name']);traces.append(trial)
        if trial is None:continue
        control=saved_trace(Path(case['control_directory']),case['control_waveforms_sha256'])
        if not np.allclose(trial[0],control[0],rtol=0,atol=1e-15):raise ValueError('Control time grid differs')
        original=spectrum(*control);changed=spectrum(*trial)
        first=original['a0_related_power_4_5_to_5_5mhz_v2'];second=changed['a0_related_power_4_5_to_5_5mhz_v2']
        rows.append(dict(case=case['name'],study='analytic source vs linear-source control at identical dt',
            full_band_voltage_change=metrics(trial[1],control[1]),control_spectrum=original,
            analytic_spectrum=changed,high_frequency_power_change_db=float(10*np.log10(second/first))
                if min(first,second)>0 else None))
    complete=all(t is not None for t in traces)
    if complete:
        if not np.allclose(traces[0][0],traces[1][0],rtol=0,atol=1e-15):raise ValueError('Refinement time grids differ')
        raw=metrics(traces[0][1],traces[1][1]);grid=np.arange(1750)/5000000
        sampled=metrics(*[antialiased_samples(t,v,grid) for t,v in traces])
        rows.append(dict(study='analytic-source temporal refinement',full_band_voltage_change=raw,
            anti_aliased_5mhz_voltage_change=sampled,
            full_band_within_one_percent=all(v['raw_l2_change_percent'] is not None
                and v['raw_l2_change_percent']<1. for v in raw.values())))
    limits=['Raw full-band metrics are primary; filtering is not evidence that raw ringing disappeared.',
        'Two time levels measure change but do not establish convergence order.',
        'This tests representation of the known ideal source, not calibrated experimental electronics.',
        'Subtracted voltage is A0-related, not a verified pure A0 mode.']
    write_json(root/'findings.json',dict(completed=complete,records=rows,limitations=limits),overwrite=True)
    lines=['Analytic drive controlled experiment',f'Completed both approved runs: {complete}',
           '250 kHz; full 1m plate; 350 microseconds; no gain or phase fitting.']
    for row in rows:
        lines.append(row.get('case','Two-run comparison')+' | '+row['study'])
        if 'high_frequency_power_change_db' in row:
            value=row['high_frequency_power_change_db']
            lines.append('  4.5-5.5 MHz A0-related power change vs control: '+
                         (f'{value:.2f} dB (negative means reduction)' if value is not None else 'undefined'))
        lines.append('  Full-band voltage change: '+'; '.join(
            f'{k} {v["raw_l2_change_percent"]:.4f}%' if v['raw_l2_change_percent'] is not None
            else k+' undefined' for k,v in row['full_band_voltage_change'].items()))
        if 'full_band_within_one_percent' in row:
            lines.append('  Every full-band channel below 1%: '+str(row['full_band_within_one_percent']))
            lines.append('  Separately labelled anti-aliased 5 MHz band: '+'; '.join(
                f'{k} {v["raw_l2_change_percent"]:.4f}%' if v['raw_l2_change_percent'] is not None
                else k+' undefined' for k,v in row['anti_aliased_5mhz_voltage_change'].items()))
    lines.extend(limits)
    atomic_write(root/'findings.txt',lambda stream:stream.write(('\n'.join(lines)+'\n').encode()),overwrite=True)
    return complete


def worker(root,name):
    plan=json.loads((root/'plan.json').read_text())
    if plan['identity']!=identity():raise ValueError('Frozen sources/configuration/data changed')
    case=next(c for c in plan['cases'] if c['name']==name)
    folder=root/name
    if (folder/'result.json').exists():raise FileExistsError(folder/'result.json')
    folder.mkdir(parents=True,exist_ok=True)
    os.environ.setdefault('CUPY_CACHE_DIR',str(root/'cache'))
    started=time.perf_counter();last=-math.inf
    result=dict(completed=False,frequency_hz=250000,case=case)
    def event(phase,**details):
        elapsed=time.perf_counter()-started
        if phase=='propagating':
            done=details['steps_completed'];total=details['requested_steps'];fraction=done/total
            eta=elapsed*(1/fraction-1) if fraction else 0.
            print(f'{name} | {100*fraction:6.2f}% | {done:,}/{total:,} steps | '
                  f'elapsed {elapsed/60:.1f} min | ETA {eta/60:.1f} min',flush=True)
        else:print(f'{name} | {phase} | elapsed {elapsed:.1f} s',flush=True)
        write_json(folder/'status.json',dict(status='running',phase=phase,elapsed_seconds=elapsed,
            **{k:v for k,v in details.items() if k!='body_reports'}),overwrite=True)
    try:
        config=copy.deepcopy(plan['configuration']);config['mesh']['minimum_steps']=case['steps']
        guard=PeakGuard(ResourceLimits(**{k:None for k in config['resource_limits']}),telemetry_errors_fatal=False)
        guard.check(force=True);cooling=ThermalPause(guard,82.,78.,allow_missing=True)
        system,load,probes,required=queue.build(config,guard,event)
        if case['steps']<required:raise ValueError('Requested dt is not stable for the unchanged system')
        row=record(250000);drive=np.asarray(row['nifgen_avg_waveform_ch1'])
        drive_t=np.arange(len(drive))/row['sampleRate'];force,source=analytic_source(250000,5,drive_t,drive)
        result['source']=source
        def progress(state):
            nonlocal last
            cooling.check(state['steps_completed'])
            if time.perf_counter()-last>=15 or state['steps_completed']==case['steps']:
                event('propagating',**state);last=time.perf_counter()
        simulation=gpu_explicit_simulate(system,case['steps'],.00035/case['steps'],probes,load,
            force=force,output_stride=8*(case['steps']//65536),energy_stride=256,guard=guard,
            progress=progress,max_workspace_mib=4096)
        atomic_write(folder/'waveforms.npz',lambda f:np.savez_compressed(f,
            time_s=simulation.time_s,voltage_v=simulation.displacement_m,
            energy_time_s=simulation.energy_time_s,modified_energy_j=simulation.modified_energy_j,
            drive_time_s=drive_t,recorded_generator_voltage_v=drive,analytic_normalized_drive_v=force(drive_t)))
        result.update(simulation_completed=True,waveforms_sha256=sha256_file(folder/'waveforms.npz'))
        atomic_write(folder/'final-state.npz',lambda f:np.savez_compressed(f,
            displacement_m=simulation.final_displacement_m,velocity_m_s=simulation.final_velocity_m_s))
        result['final_state_sha256']=sha256_file(folder/'final-state.npz')
        tail=simulation.modified_energy_j[simulation.energy_time_s>source['pulse_end_s']+2*.00035/case['steps']]
        drift=float(np.ptp(tail)/abs(tail[0])) if len(tail)>1 and tail[0]!=0 else None
        result['energy_relative_drift']=drift
        if drift is None or drift>1e-8:raise ArithmeticError('Post-pulse lossless energy check failed')
        result.update(completed=True,stable_dt_s=system.stable_dt_s,statistics=simulation.statistics,
            resource_peaks=guard.peaks,body_reports=[b.report for b in system.body_attachments])
        event('completed')
    except BaseException as exc:
        result['error']=f'{type(exc).__name__}: {exc}'
        raise
    finally:
        result['wall_seconds']=time.perf_counter()-started
        write_json(folder/'result.json',result)
        write_json(folder/'status.json',dict(status='completed' if result['completed'] else 'interrupted',
            error=result.get('error'),elapsed_seconds=result['wall_seconds']),overwrite=True)


def run(root):
    with queue_lock(root):
        plan=json.loads((root/'plan.json').read_text())
        if plan['identity']!=identity():raise ValueError('Prepared source/configuration/data changed')
        for folder in root.iterdir():
            if folder.is_dir() and worker_alive(folder):raise RuntimeError('Trial worker still running')
        write_json(root/'status.json',dict(status='running',pid=os.getpid()),overwrite=True)
        try:
            for index,case in enumerate(plan['cases'],1):
                folder=root/case['name'];folder.mkdir(parents=True,exist_ok=True)
                if not prepare_case(folder):continue
                print(f'TEST {index}/2 | 250 kHz | {case["name"]} | {case["steps"]:,} steps',flush=True)
                with (folder/'worker.log').open('a',encoding='utf-8') as log:
                    process=subprocess.Popen([sys.executable,'-u','-m','examples.pzt_analytic_trial',
                        '--root',str(root),'--worker',case['name']],cwd=PROJECT,stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace')
                    try:created=psutil.Process(process.pid).create_time()
                    except psutil.NoSuchProcess:created=0.
                    write_json(folder/'owner.json',dict(pid=process.pid,started_unix=created),overwrite=True)
                    for line in process.stdout:log.write(line);log.flush();print(line,end='',flush=True)
                    code=process.wait()
                assess(root);print((root/'findings.txt').read_text(),flush=True)
                if code:raise RuntimeError(f'Case failed, exit {code}: {folder}')
            write_json(root/'status.json',dict(status='completed'),overwrite=True)
            print('BOTH APPROVED RUNS FINISHED. No further simulations scheduled.',flush=True)
        except BaseException as exc:
            write_json(root/'status.json',dict(status='interrupted',error=str(exc)),overwrite=True)
            raise


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--reference',type=Path,default=PROJECT/'outputs/pzt-development/full-plate-convergence-20261121')
    parser.add_argument('--prepare',action='store_true');parser.add_argument('--run',action='store_true')
    parser.add_argument('--assess',action='store_true');parser.add_argument('--worker')
    args=parser.parse_args();root=args.root.resolve()
    if not root.is_relative_to(PROJECT/'outputs'):raise ValueError('Outputs must stay in project')
    if args.prepare:prepare(root,args.reference.resolve())
    elif args.worker:worker(root,args.worker)
    elif args.run:run(root)
    elif args.assess:assess(root);print((root/'findings.txt').read_text())
    else:parser.error('Choose --prepare, --run, --worker or --assess')


if __name__=='__main__':main()
