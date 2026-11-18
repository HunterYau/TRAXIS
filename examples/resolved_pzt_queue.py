"""Fresh five-frequency body-coupled queue; preflight by default.

Explicit --prepare freezes inputs; --run launches the prepared serial queue.
This is an exploratory perfect-bond model, not a calibrated experiment.
"""
import argparse
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np
from scipy.signal import hilbert

from examples.experimental_five import PROJECT, PeakGuard, record
from examples.embedded_wave import ThermalPause
from examples.compare_pzt_recordings import packet
from examples.pzt_terminal import result_lines
from traxis.config import Material, Plate
from traxis.gpu_tensor import make_layout, build_gpu_tensor, streamed_action
from traxis.gpu_pzt_body import couple_gpu_bodies
from traxis.gpu_enriched_verlet import gpu_explicit_simulate, sparse_probe_set
from traxis.pzt_body import attach_disk
from traxis.pzt_receiver import DiskGeometry
from traxis.piezoelectric import PiezoMaterial, read_profile
from traxis.resources import ResourceLimits
from traxis.elmer_reference import run_monitored
from traxis.storage import atomic_write, canonical_hash, sha256_file, write_json

ROOT=PROJECT/'outputs/wandowski-resolved-pzt-v5'
CONFIG=PROJECT/'configs/pzt/resolved-wandowski-v1.json'


def configuration():
    config=json.loads(CONFIG.read_text())
    if config['frequencies_hz'] != [50000,100000,150000,200000,250000]:
        raise ValueError('only verified plate-1 frequency records are supported')
    if config['body']['bond']!='perfect' or config['ready_for_training_data']:
        raise ValueError('unsupported bond or acceptance state')
    return config


def preflight():
    config=configuration()
    if sha256_file(PROJECT/config['data_file'])!=config['data_sha256']:
        raise ValueError('experimental reference changed')
    material=PiezoMaterial.from_profile(read_profile(PROJECT/config['material_profile']))
    for frequency in config['frequencies_hz']:
        row=record(frequency)
        data=np.asarray(row['niscope_avg_waveform']);drive=np.asarray(row['nifgen_avg_waveform_ch1'])
        if (data.ndim!=2 or data.shape[1]!=4 or not np.isfinite(data).all()
                or drive.ndim!=1 or not np.isfinite(drive).all() or np.max(abs(drive))<=0
                or np.any(np.ptp(data[:,[2,3]],axis=0)==0) or row['Number_of_averages']!=50):
            raise ValueError('missing or invalid measured PZT/drive record')
    files=[*sorted((PROJECT/'traxis').glob('*.py')),Path(__file__),CONFIG,
        PROJECT/'examples/experimental_five.py',PROJECT/'examples/embedded_wave.py',
        PROJECT/'examples/compare_pzt_recordings.py',PROJECT/config['material_profile']]
    files.append(PROJECT/'examples/pzt_terminal.py')
    if config.get('launch_ready'):
        proof_path=PROJECT/config['readiness_evidence']
        if sha256_file(proof_path)!=config['readiness_sha256']:raise ValueError('readiness evidence changed')
        proof=json.loads(proof_path.read_text())
        if not proof.get('passed') or proof['selected_resolution']!=[config['body'][k] for k in ('sectors','rings','layers')]:
            raise ValueError('selected body mesh lacks passing readiness evidence')
        for name,checksum in proof['source_sha256'].items():
            if sha256_file(PROJECT/name)!=checksum:raise ValueError('body readiness source changed: '+name)
        for name,checksum in proof['evidence_sha256'].items():
            if sha256_file(PROJECT/name)!=checksum:raise ValueError('readiness check evidence changed: '+name)
        files.append(proof_path)
    answer=dict(configuration=config,source_sha256={str(p.relative_to(PROJECT)).replace('\\','/'):
        sha256_file(p) for p in files},simulation_started=False,material_id=material.id)
    answer['fingerprint']=canonical_hash(answer)
    return answer


def build(config,guard,event):
    mesh=config['mesh']
    layout=make_layout(Plate(*config['plate_m']),tuple(mesh['cells']),mesh['order'],
        max_dofs=32000000,max_device_mib=6144,max_host_mib=4096)
    material=PiezoMaterial.from_profile(read_profile(PROJECT/config['material_profile']))
    base=build_gpu_tensor(layout,Material(**config['host_material']),guard=guard)
    bodies=[];next_dof=layout.screen['vector_dofs']
    settings=config['body']
    for center,surface,electrical in [
        (config['actuator_center_xy_m'],'top','voltage'),
        (config['receiver_center_xy_m'],'top','open'),
        (config['receiver_center_xy_m'],'bottom','open')]:
        event('assembling_pzt_body',body_number=len(bodies)+1,surface=surface)
        body=attach_disk(layout,material,DiskGeometry(tuple(center),config['disk_radius_m'],
            config['disk_thickness_m'],surface,1),first_dof=next_dof,electrical=electrical,
            sectors=settings['sectors'],rings=settings['rings'],layers=settings['layers'],
            bond=settings['bond'],guard=guard,max_workspace_mib=2048)
        bodies.append(body);next_dof=body.next_dof
    event('coupling_pzt_bodies')
    system=couple_gpu_bodies(base,bodies,guard=guard,max_workspace_mib=2048)
    # Stability may require more steps than the old bare-plate model.
    steps=max(mesh['minimum_steps'],math.ceil(mesh['duration_s']/(.8*system.stable_dt_s)))
    steps=math.ceil(steps/256)*256
    if steps>500000: raise ValueError('required stable step count exceeds supported limit')
    force=np.zeros(next_dof);force[bodies[0].indices]=-bodies[0].coupling
    probes=sparse_probe_set([(b.indices,b.coupling/b.total_capacitance_f) for b in bodies[1:]],next_dof)
    event('bodies_ready',requested_steps=steps,stable_dt_s=system.stable_dt_s,
        body_reports=[b.report for b in bodies],device_screen_mib=system.screen['device_storage_screen_mib'])
    return system,force,probes,steps


def compare(time_s,voltage,experimental,frequency,sample_rate):
    """Same predeclared paired timing method, with unresolved results retained."""
    measured_time=np.arange(len(experimental))/sample_rate
    result={'method':{'timing':'paired flight-time difference, lower better; same predeclared windows',
        'shape':'100 * L2 difference of unit-L2 Hilbert envelopes on common 0–349.8 us grid; lower better',
        'limits':'Shape normalizes amplitude; no time shift or polarity fitting. Not overall accuracy or absolute-voltage validation.'}}
    for mode,sign,label in [('S0',1,'fast'),('A0',-1,'slow')]:
        observed=packet(measured_time,experimental[:,0]+sign*experimental[:,1],frequency,mode)
        predicted=packet(time_s,voltage[:,0]+sign*voltage[:,1],frequency,mode)
        status='exploratory_timing'
        if observed['window_edge'] or predicted['window_edge'] or observed['flight_us']<=0:
            status='unresolved_arrival'
        if mode=='A0' and frequency>140000:status='above_reported_A0_detection_range'
        difference=(100*abs(predicted['flight_us']-observed['flight_us'])/observed['flight_us']
            if observed['flight_us']>0 else None)
        result[label+'_arrival_error_percent']=difference if status=='exploratory_timing' else None
        result[label+'_status']=status
        result[label+'_details']=dict(measured=observed,simulated=predicted,raw_candidate_error_percent=difference)
        grid=np.arange(1750)*2e-7;envelopes=[]
        for t,channels in ((measured_time,experimental),(time_s,voltage)):
            trace=np.interp(grid,t,channels[:,0]+sign*channels[:,1])
            trace-=trace[grid<20e-6].mean()
            envelopes.append(abs(hilbert(trace,8192)[:len(grid)]))
        norms=[np.linalg.norm(e) for e in envelopes]
        result[label+'_envelope_shape_error_percent']=(float(100*np.linalg.norm(
            envelopes[0]/norms[0]-envelopes[1]/norms[1])) if min(norms)>0 else None)
        if min(norms)==0:
            result[label+'_arrival_error_percent']=None
            result[label+'_status']='no_signal'
    return result


def worker(frequency,directory,*,smoke_steps=None):
    frozen=json.loads((ROOT/'prepared.json').read_text())
    if frozen!=preflight():raise ValueError('prepared source/config/data changed')
    config=frozen['configuration']
    if frequency not in config['frequencies_hz']:raise ValueError('frequency not scheduled')
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter();guard=PeakGuard(ResourceLimits(max_seconds=86400,max_rss_mib=4096))
    result=dict(completed=False,frequency_hz=frequency,ready_for_training_data=False,
        protocol_fingerprint=frozen['fingerprint'],smoke_test=smoke_steps is not None)
    def event(phase,**details):
        print(json.dumps(dict(phase=phase,elapsed_seconds=time.perf_counter()-started,**details)),flush=True)
    try:
        system,force,probes,steps=build(config,guard,event)
        dt=config['mesh']['duration_s']/steps
        requested_steps=steps if smoke_steps is None else smoke_steps
        row=record(frequency);drive=np.asarray(row['nifgen_avg_waveform_ch1'])
        drive_t=np.arange(len(drive))/row['sampleRate'];reference=drive/np.max(abs(drive))
        cooling=ThermalPause(guard,76.,72.);last=-math.inf
        def progress(state):
            nonlocal last
            cooling.check(state['steps_completed'])
            if time.perf_counter()-last>=15 or state['steps_completed']==requested_steps:
                event('propagating',**state);last=time.perf_counter()
        simulation=gpu_explicit_simulate(system,requested_steps,dt,probes,force,
            force=lambda t:float(np.interp(t,drive_t,reference,left=0.,right=0.)),
            output_stride=8,energy_stride=256,guard=guard,progress=progress,max_workspace_mib=4096)
        if smoke_steps is not None:
            result.update(completed=True,actual_steps=requested_steps,simulation_statistics=simulation.statistics,
                body_reports=[b.report for b in system.body_attachments],stable_dt_s=system.stable_dt_s,
                full_steps=steps,
                note='Short execution check only; no experimental result or full-interval claim')
            return
        voltage=simulation.displacement_m # Probe weights explicitly produce volts, not displacement.
        measured=np.asarray(row['niscope_avg_waveform'])[:,[2,3]]
        atomic_write(directory/'waveforms.npz',lambda f:np.savez_compressed(f,
            simulation_time_s=simulation.time_s,ideal_open_reference_voltage_v=voltage,
            experimental_time_s=np.arange(len(measured))/row['sampleRate'],experimental_pzt_voltage_v=measured,
            drive_time_s=drive_t,recorded_generator_voltage_v=drive,unit_peak_reference_voltage_v=reference,
            energy_time_s=simulation.energy_time_s,modified_energy_j=simulation.modified_energy_j))
        # Preserve the completed integration even if checking/analysis later fails.
        atomic_write(directory/'final-state.npz',lambda f:np.savez_compressed(f,
            displacement_m=simulation.final_displacement_m,velocity_m_s=simulation.final_velocity_m_s))
        result.update(simulation_completed=True,waveforms_sha256=sha256_file(directory/'waveforms.npz'),
            final_state_sha256=sha256_file(directory/'final-state.npz'))
        event('checking_final_force')
        gpu_force=system.stiffness@simulation.final_displacement_m
        nh=system.layout.screen['vector_dofs'];cpu=np.zeros(len(gpu_force))
        cpu[:nh]=streamed_action(system.layout,Material(**config['host_material']),
            simulation.final_displacement_m[:nh],guard,chunk_elements=64)
        for body in system.body_attachments:
            cpu[body.indices]+=body.stiffness@simulation.final_displacement_m[body.indices]
        error=float(np.linalg.norm(cpu-gpu_force)/max(np.linalg.norm(cpu),1e-300))
        # Use the actual recorded drive end, not a nominal pulse duration.
        active=np.flatnonzero(abs(reference)>1e-12)
        drive_end=drive_t[active[-1]] if len(active) else 0.
        tail=simulation.modified_energy_j[simulation.energy_time_s>drive_end+2*dt]
        drift=float(np.ptp(tail)/max(abs(tail[0]),1e-300)) if len(tail)>1 else None
        if error>1e-10 or (drift is not None and drift>1e-8):
            raise ArithmeticError('independent force/energy check failed')
        event('comparing_measured_recordings')
        comparison=compare(simulation.time_s,voltage,measured,frequency,row['sampleRate'])
        write_json(directory/'comparison.json',comparison)
        result.update(completed=True,cpu_force_relative_error=error,modified_energy_relative_drift=drift,
            comparison=comparison,body_reports=[b.report for b in system.body_attachments],
            simulation_statistics=simulation.statistics,waveforms_sha256=sha256_file(directory/'waveforms.npz'),
            final_state_sha256=sha256_file(directory/'final-state.npz'))
        event('trial_complete',frequency_hz=frequency,comparison=comparison)
    except Exception as exc:
        result['error']=f'{type(exc).__name__}: {exc}'
        raise
    finally:
        result.update(wall_seconds=time.perf_counter()-started,resource_peaks=getattr(guard,'peaks',{}))
        write_json(directory/'result.json',result)
        summary='\n'.join(result_lines(frequency,result,directory))+'\n'
        atomic_write(directory/'summary.txt',lambda f:f.write(summary.encode('utf-8')))
        print(summary,flush=True)


def run_queue():
    if not configuration().get('launch_ready',False):
        raise RuntimeError('Long queue held: resolve the documented body-mesh refinement and reference-selection requirements first')
    if json.loads((ROOT/'prepared.json').read_text())!=preflight():raise ValueError('prepared inputs changed')
    if (ROOT/'queue-owner.json').exists():
        raise RuntimeError('This queue was already launched. Existing results are preserved; do not start it twice.')
    write_json(ROOT/'queue-owner.json',dict(pid=os.getpid(),started_unix=time.time()))
    summaries=[]
    def announce(message):
        with (ROOT/'queue.log').open('a',encoding='utf-8') as stream:
            stream.write(message+'\n')
        print(message,flush=True)
    frequencies=configuration()['frequencies_hz']
    for index,frequency in enumerate(frequencies,1):
        if (ROOT/'STOP_AFTER_CURRENT').exists():
            announce('Stopped between experiments. Completed results retained.');return
        announce(f'Starting experiment {index}/{len(frequencies)}: {frequency//1000} kHz')
        trial=ROOT/f'{frequency//1000}khz';trial.mkdir(exist_ok=False)
        command=[sys.executable,'-m','examples.resolved_pzt_queue','--worker',str(frequency),str(trial/'native')]
        report=run_monitored(command,PROJECT,{**os.environ,'CUPY_CACHE_DIR':str(ROOT/'cupy-cache')},
            PeakGuard(ResourceLimits(max_seconds=86500,max_rss_mib=4096)),trial/'worker.log')
        write_json(trial/'launcher.json',report)
        if report['exit_code']!=0:
            announce(f'{frequency//1000} kHz failed; queue stopped. Check {trial}/worker.log')
            raise RuntimeError(f'{frequency} Hz failed; queue stopped')
        result=json.loads((trial/'native/result.json').read_text())
        if not result.get('completed') or not isinstance(result.get('comparison'),dict):
            raise RuntimeError('worker did not save a completed comparison; queue stopped')
        if (json.loads((trial/'native/comparison.json').read_text())!=result['comparison']
                or not (trial/'native/summary.txt').is_file()):
            raise RuntimeError('comparison or text summary not saved; queue stopped')
        for name,key in [('waveforms.npz','waveforms_sha256'),('final-state.npz','final_state_sha256')]:
            if sha256_file(trial/'native'/name)!=result[key]:raise ValueError('saved output checksum mismatch')
        announce('\n'.join(result_lines(frequency,result,trial/'native')))
        summaries.append(dict(frequency_hz=frequency,wall_seconds=result['wall_seconds'],comparison=result['comparison']))
    write_json(ROOT/'all-results.json',summaries)
    announce(f'All {len(frequencies)} experiments saved and compared. Results: {ROOT}/all-results.json')


def main():
    parser=argparse.ArgumentParser(__doc__);actions=parser.add_mutually_exclusive_group()
    actions.add_argument('--prepare',action='store_true');actions.add_argument('--run',action='store_true')
    actions.add_argument('--worker',nargs=2);actions.add_argument('--smoke',type=Path,
        help='run 128 steps on full geometry, save to a new directory, then stop')
    args=parser.parse_args()
    if args.worker:worker(int(args.worker[0]),args.worker[1]);return
    if args.smoke:worker(50000,args.smoke,smoke_steps=128);return
    if args.run:run_queue();return
    checked=preflight()
    if args.prepare:
        ROOT.mkdir(parents=True,exist_ok=True)
        path=ROOT/'prepared.json'
        if path.exists():
            if json.loads(path.read_text())!=checked:raise ValueError('Existing prepared queue differs; preserve it and use a new output version')
        else:write_json(path,checked)
    print(json.dumps(checked,indent=2))


if __name__=='__main__':main()
