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
from examples.pzt_terminal import result_lines, export_queue_results
from examples.plate_outputs import (CURRENT_RUN, result_directory, read_saved, text_file,
    queue_lock, owner_alive, process_owner, trial_status, archive_attempt)
from examples.output_locations import organized_output
from traxis.config import Material, Plate
from traxis.damping import damping_parameters
from traxis.arrival import adaptive_packet
from traxis.experimental_setup import applied_actuator_voltage
from traxis.signal_comparison import waveform_metrics
from traxis.gpu_tensor import make_layout, build_gpu_tensor, streamed_action
from traxis.gpu_pzt_body import couple_gpu_bodies
from traxis.gpu_enriched_verlet import gpu_explicit_simulate, sparse_probe_set
from traxis.pzt_body import attach_disk
from traxis.pzt_receiver import DiskGeometry
from traxis.piezoelectric import PiezoMaterial, read_profile
from traxis.resources import ResourceLimits
from traxis.elmer_reference import run_monitored
from traxis.storage import atomic_write, canonical_hash, sha256_file, write_json

ROOT=CURRENT_RUN
CONFIG=PROJECT/'configs/pzt/resolved-wandowski-ideal-restored.json'


def queue_resource_limits():
    """Monitor resources without automatic resource or time cutoffs."""
    config=configuration()
    return ResourceLimits(**config['resource_limits'])


def configuration():
    config=json.loads(CONFIG.read_text())
    start=config.get('start_experiment',1)
    if (type(start) is not int or not 1 <= start <= 5
            or config['frequencies_hz'] != [50000,100000,150000,200000,250000][start-1:]):
        raise ValueError('only verified plate-1 frequency records are supported')
    if config['body']['bond'] not in ('perfect','elastic_interface') or config['ready_for_training_data']:
        raise ValueError('unsupported bond or acceptance state')
    if config['body']['bond']=='elastic_interface' and not config.get('setup_profile'):
        raise ValueError('finite bond requires an explicit setup profile')
    names={'max_seconds','max_rss_mib','max_system_memory_percent','min_available_mib',
           'max_gpu_memory_percent','max_gpu_temperature_c','max_cpu_temperature_c'}
    if config.get('resource_limits')!={name:None for name in names}:
        raise ValueError('this queue requires the user-requested monitoring-only resource policy')
    ResourceLimits(**config['resource_limits'])
    if not 0 < config['cooling']['lower_c'] < config['cooling']['upper_c']:
        raise ValueError('cooling resume temperature must be below its pause temperature')
    return config


def preflight():
    config=configuration()
    if sha256_file(organized_output(PROJECT/config['data_file']))!=config['data_sha256']:
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
    files.append(PROJECT/'examples/plate_outputs.py')
    files.append(PROJECT/'examples/output_locations.py')
    if config.get('restored_from'):
        receipt=PROJECT/config['restored_from']['prepared_receipt']
        if sha256_file(receipt)!=config['restored_from']['sha256']:
            raise ValueError('previous ideal-settings receipt changed')
        files.append(receipt)
    if config.get('setup_profile'):
        setup_path=PROJECT/config['setup_profile']
        setup=json.loads(setup_path.read_text())
        if (setup['receiver']['resistance_ohm']<=0 or setup['amplifier']['gain_v_v']<=0
                or setup['receiver']['scope_capacitance_f']<0 or setup['receiver']['cable_length_m']<0
                or setup['receiver']['cable_capacitance_f_m']<0):raise ValueError('invalid assumed hardware settings')
        files.append(setup_path)
    if config.get('damping_profile'):
        profile_path=PROJECT/config['damping_profile']
        profile=json.loads(profile_path.read_text())
        for frequency in config['frequencies_hz']:damping_parameters(profile,frequency)
        files.append(profile_path)
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
    setup=read_setup(config)
    input_cap=0.
    if setup:
        readout=setup['receiver']
        input_cap=readout['scope_capacitance_f']+readout['cable_length_m']*readout['cable_capacitance_f_m']
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
            bond=settings['bond'],adhesive=setup['adhesive'] if setup else None,
            input_capacitance_f=input_cap if electrical=='open' else 0.,
            guard=guard,max_workspace_mib=4096)
        bodies.append(body);next_dof=body.next_dof
    event('coupling_pzt_bodies')
    system=couple_gpu_bodies(base,bodies,guard=guard,max_workspace_mib=4096)
    # Stability may require more steps than the old bare-plate model.
    steps=max(mesh['minimum_steps'],math.ceil(mesh['duration_s']/(.8*system.stable_dt_s)))
    steps=math.ceil(steps/256)*256
    if steps>500000: raise ValueError('required stable step count exceeds supported limit')
    force=np.zeros(next_dof);force[bodies[0].indices]=-bodies[0].coupling
    probes=sparse_probe_set([(b.indices,b.coupling/b.total_capacitance_f) for b in bodies[1:]],next_dof)
    event('bodies_ready',requested_steps=steps,stable_dt_s=system.stable_dt_s,
        body_reports=[b.report for b in bodies],device_screen_mib=system.screen['device_storage_screen_mib'])
    return system,force,probes,steps


def read_setup(config):
    return json.loads((PROJECT/config['setup_profile']).read_text()) if config.get('setup_profile') else None


def configured_damping(config,frequency):
    if config.get('damping_profile'):
        return damping_parameters(json.loads((PROJECT/config['damping_profile']).read_text()),frequency)
    return dict(model='none',mass_damping_per_s=0.,scope='Restored ideal lossless model')


def actuator_signal(setup,drive_t,drive):
    if setup:
        return lambda t:applied_actuator_voltage(t,drive_t,drive,setup['amplifier'])
    reference=drive/np.max(abs(drive))
    return lambda t:np.interp(t,drive_t,reference,left=0.,right=0.)


def receiver_ports(system,setup):
    if not setup:
        return [] # Ideal open ports are already condensed into mechanical stiffness.
    return [dict(indices=b.indices,coupling=b.coupling,capacitance_f=b.total_capacitance_f,
                 resistance_ohm=setup['receiver']['resistance_ohm'],probe_index=j)
            for j,b in enumerate(system.body_attachments[1:])]


def compare(time_s,voltage,experimental,frequency,sample_rate,*,config=None):
    """Same predeclared paired timing method, with unresolved results retained."""
    measured_time=np.arange(len(experimental))/sample_rate
    result={'method':{'timing':'Independent theory-guided adaptive envelope packets; fixed-window result retained for audit; lower better',
        'shape':'100 * L2 difference of unit-L2 Hilbert envelopes on common 0–349.8 us grid; lower better',
        'limits':'Shape normalizes amplitude; no time shift or polarity fitting. Not overall accuracy or absolute-voltage validation.'}}
    result['method']['wiggles']={
        'interval':'Entire shared 0-349.8 us, 5 MHz grid; first-20-us mean removed',
        'normalized_difference':'100*L2 difference of unit-L2 signed voltage traces; lower better, range 0-200; removes overall strength only',
        'raw_difference':'100*L2(simulated-measured)/L2(measured), no strength normalization; drive/load mismatch remains',
        'correlation':'Pearson signed-voltage correlation, -1 to +1; +1 best; ignores overall strength',
        'alignment':'No fitted shift, time stretch, polarity reversal or peak-search window',
        'mode_warning':'Sum/difference traces may contain residual modes and reflections; not proof of isolated S0/A0'}
    result['method']['envelope_pearson']='Pearson r of Hilbert envelopes over the entire shared 0-349.8 us record; no fitted shift; -1 to +1, +1 best; undefined for constant/zero envelopes. This is not an accuracy percentage.'
    grid=np.arange(1750)*2e-7
    real=np.column_stack([np.interp(grid,measured_time,experimental[:,j]) for j in range(2)])
    sim=np.column_stack([np.interp(grid,time_s,voltage[:,j]) for j in range(2)])
    real-=real[grid<20e-6].mean(axis=0);sim-=sim[grid<20e-6].mean(axis=0)
    result['waveform_channels']={name:waveform_metrics(real[:,j],sim[:,j])
                                 for j,name in enumerate(('top','bottom'))}
    geometry={}
    if config is not None:
        source=np.asarray(config['actuator_center_xy_m']);receiver=np.asarray(config['receiver_center_xy_m'])
        width,height,thickness=config['plate_m']
        images=[np.array([-source[0],source[1]]),np.array([2*width-source[0],source[1]]),
                np.array([source[0],-source[1]]),np.array([source[0],2*height-source[1]])]
        geometry=dict(distance_m=float(np.linalg.norm(receiver-source)),thickness_m=thickness,
            material=Material(**config['host_material']),cycles=config['num_cycles'],
            reflection_distance_m=min(float(np.linalg.norm(receiver-image)) for image in images))
    for mode,sign,label in [('S0',1,'fast'),('A0',-1,'slow')]:
        real_trace=experimental[:,0]+sign*experimental[:,1];sim_trace=voltage[:,0]+sign*voltage[:,1]
        fixed_real=packet(measured_time,real_trace,frequency,mode)
        fixed_sim=packet(time_s,sim_trace,frequency,mode)
        fixed_error=(100*abs(fixed_sim['flight_us']-fixed_real['flight_us'])/fixed_real['flight_us']
                     if fixed_real['flight_us']>0 else None)
        result[label+'_fixed_window']=dict(measured=fixed_real,simulated=fixed_sim,
            raw_candidate_error_percent=fixed_error,reliable_window=not(fixed_real['window_edge'] or fixed_sim['window_edge']))
        observed=adaptive_packet(measured_time,real_trace,frequency,mode,**geometry)
        predicted=adaptive_packet(time_s,sim_trace,frequency,mode,**geometry)
        result[label+'_adaptive_window']=dict(measured=observed,simulated=predicted)
        adaptive_ok=all(p['status']=='localized_packet' for p in (observed,predicted))
        fixed_ok=(result[label+'_fixed_window']['reliable_window'] and fixed_real['flight_us']>0)
        same_peaks=all(abs(a['packet_peak_us']-b['packet_peak_us'])<.21
                       for a,b in ((observed,fixed_real),(predicted,fixed_sim)))
        method='adaptive'
        # Never choose whichever paired percentage is smaller.
        if not adaptive_ok and fixed_ok and same_peaks and all(
                p['status'] in ('unresolved_arrival','localized_packet') for p in (observed,predicted)):
            observed,predicted=fixed_real,fixed_sim
            method='fixed_same_peak_fallback'
        status='exploratory_timing'
        if method=='adaptive' and (not adaptive_ok or (fixed_ok and not same_peaks)):
            status='unresolved_arrival'
        if mode=='A0' and frequency>140000:status='above_reported_A0_detection_range'
        difference=(100*abs(predicted['flight_us']-observed['flight_us'])/observed['flight_us']
            if observed['flight_us']>0 else None)
        result[label+'_arrival_error_percent']=difference if status=='exploratory_timing' else None
        result[label+'_status']=status
        result[label+'_arrival_method']=method
        result[label+'_details']=dict(measured=observed,simulated=predicted,raw_candidate_error_percent=difference)
        result[label+'_waveform']=waveform_metrics(real[:,0]+sign*real[:,1],sim[:,0]+sign*sim[:,1])
        grid=np.arange(1750)*2e-7;envelopes=[]
        for t,channels in ((measured_time,experimental),(time_s,voltage)):
            trace=np.interp(grid,t,channels[:,0]+sign*channels[:,1])
            trace-=trace[grid<20e-6].mean()
            envelopes.append(abs(hilbert(trace,8192)[:len(grid)]))
        norms=[np.linalg.norm(e) for e in envelopes]
        result[label+'_envelope_pearson_r']=waveform_metrics(envelopes[0],envelopes[1])['correlation']
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
    setup=read_setup(config)
    # Direct startup checks need the same writable cache as scheduled workers.
    os.environ.setdefault('CUPY_CACHE_DIR',str(ROOT/'cache/cupy'))
    if frequency not in config['frequencies_hz']:raise ValueError('frequency not scheduled')
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    if (directory/'result.json').exists():
        raise FileExistsError('Existing frequency results must be preserved before retrying')
    write_json(directory/'worker-owner.json',process_owner())
    started=time.perf_counter();guard=PeakGuard(queue_resource_limits(),telemetry_errors_fatal=False)
    result=dict(completed=False,frequency_hz=frequency,ready_for_training_data=False,
        protocol_fingerprint=frozen['fingerprint'],smoke_test=smoke_steps is not None,
        assumed_setup=setup,setup_specimen_calibrated=False,model_assumptions=config['model_assumptions'])
    def event(phase,**details):
        print(json.dumps(dict(phase=phase,elapsed_seconds=time.perf_counter()-started,**details)),flush=True)
    try:
        system,force,probes,steps=build(config,guard,event)
        dt=config['mesh']['duration_s']/steps
        damping=configured_damping(config,frequency)
        result['damping']=damping
        event('damping_configured')
        requested_steps=steps if smoke_steps is None else smoke_steps
        row=record(frequency);drive=np.asarray(row['nifgen_avg_waveform_ch1'])
        drive_t=np.arange(len(drive))/row['sampleRate'];reference=drive/np.max(abs(drive))
        applied_voltage=actuator_signal(setup,drive_t,drive)
        ports=receiver_ports(system,setup)
        event('experiment_setup_configured')
        cooling=ThermalPause(guard,config['cooling']['upper_c'],config['cooling']['lower_c'],
                             allow_missing=True);last=-math.inf
        def progress(state):
            nonlocal last
            cooling.check(state['steps_completed'])
            if time.perf_counter()-last>=15 or state['steps_completed']==requested_steps:
                event('propagating',**state);last=time.perf_counter()
        simulation=gpu_explicit_simulate(system,requested_steps,dt,probes,force,
            force=lambda t:float(applied_voltage(t)),
            output_stride=8,energy_stride=256,guard=guard,progress=progress,max_workspace_mib=4096,
            mass_damping_per_s=damping['mass_damping_per_s'],electrical_ports=ports)
        if smoke_steps is not None:
            result.update(completed=True,actual_steps=requested_steps,simulation_statistics=simulation.statistics,
                body_reports=[b.report for b in system.body_attachments],stable_dt_s=system.stable_dt_s,
                full_steps=steps,
                note='Short execution check only; no experimental result or full-interval claim')
            return
        voltage=simulation.displacement_m # Probe weights explicitly produce volts, not displacement.
        measured=np.asarray(row['niscope_avg_waveform'])[:,[2,3]]
        atomic_write(directory/'waveforms.npz',lambda f:np.savez_compressed(f,
            simulation_time_s=simulation.time_s,simulated_receiver_voltage_v=voltage,
            experimental_time_s=np.arange(len(measured))/row['sampleRate'],experimental_pzt_voltage_v=measured,
            drive_time_s=drive_t,recorded_generator_voltage_v=drive,unit_peak_reference_voltage_v=reference,
            assumed_applied_actuator_voltage_v=applied_voltage(drive_t),
            energy_time_s=simulation.energy_time_s,modified_energy_j=simulation.modified_energy_j))
        # Preserve the completed integration even if checking/analysis later fails.
        atomic_write(directory/'final-state.npz',lambda f:np.savez_compressed(f,
            displacement_m=simulation.final_displacement_m,velocity_m_s=simulation.final_velocity_m_s,
            receiver_free_charge_c=np.asarray([p['final_free_charge_c'] for p in simulation.statistics['electrical_ports']])))
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
        active=np.flatnonzero(abs(drive if setup else reference)>1e-12)
        drive_end=drive_t[active[-1]]+(setup['amplifier']['delay_s'] if setup else 0.) if len(active) else 0.
        tail=simulation.modified_energy_j[simulation.energy_time_s>drive_end+2*dt]
        drift=float(np.ptp(tail)/max(abs(tail[0]),1e-300)) if len(tail)>1 else None
        growth=float(max(0.,np.max(np.diff(tail)))/max(abs(tail[0]),1e-300)) if len(tail)>1 else None
        lossless=not damping['mass_damping_per_s'] and not ports
        if (error>1e-10 or (growth is not None and growth>1e-8)
                or (lossless and drift is not None and drift>1e-8)):
            raise ArithmeticError('independent force/energy check failed')
        event('comparing_measured_recordings')
        comparison=compare(simulation.time_s,voltage,measured,frequency,row['sampleRate'],config=config)
        write_json(directory/'comparison.json',comparison)
        result.update(completed=True,cpu_force_relative_error=error,
            modified_energy_relative_drift=drift if lossless else None,
            post_drive_energy_relative_range=drift,post_drive_energy_max_relative_growth=growth,
            comparison=comparison,body_reports=[b.report for b in system.body_attachments],
            simulation_statistics=simulation.statistics,waveforms_sha256=sha256_file(directory/'waveforms.npz'),
            final_state_sha256=sha256_file(directory/'final-state.npz'))
        event('trial_complete',frequency_hz=frequency,comparison=comparison)
    except BaseException as exc:
        result['error']=f'{type(exc).__name__}: {exc}'
        raise
    finally:
        result.update(wall_seconds=time.perf_counter()-started,resource_peaks=getattr(guard,'peaks',{}))
        write_json(directory/'result.json',result)
        summary='\n'.join(result_lines(frequency,result,directory))+'\n'
        summary_path=(directory/'summary.txt' if smoke_steps is not None
                      else ROOT/'summaries'/f'{frequency//1000}khz.txt')
        text_file(summary_path,summary)
        trial_status(directory,'completed' if result.get('completed') else 'interrupted',
                     reason=result.get('error',''))
        print(summary,flush=True)


def validate_completed(frequency):
    trial=ROOT/f'{frequency//1000}khz';directory=result_directory(trial)
    result=read_saved(directory/'result.json')
    if not result or not result.get('completed'):
        return None
    if not isinstance(result.get('comparison'),dict):
        raise ValueError(f'{frequency} Hz has no completed comparison')
    if read_saved(directory/'comparison.json')!=result['comparison']:
        raise ValueError('comparison differs from saved result')
    for name,key in [('waveforms.npz','waveforms_sha256'),('final-state.npz','final_state_sha256')]:
        if sha256_file(directory/name)!=result[key]:
            raise ValueError('saved output checksum mismatch: '+str(directory/name))
    return result


def run_queue():
    config=configuration()
    if not config.get('launch_ready',False):
        raise RuntimeError('Long queue held: resolve the documented body-mesh refinement and reference-selection requirements first')
    if not (ROOT/'prepared.json').is_file():
        raise RuntimeError('Prepare this experiment folder first with --prepare')
    with queue_lock(ROOT):
        if (ROOT/'queue-owner.json').exists() and owner_alive(read_saved(ROOT/'queue-owner.json')) is not False:
            raise RuntimeError('This queue is already running or ownership cannot be checked.')
        for frequency in config['frequencies_hz']:
            trial=ROOT/f'{frequency//1000}khz'
            owner=read_saved(result_directory(trial)/'worker-owner.json')
            if owner and owner_alive(owner) is not False:
                raise RuntimeError(f'{frequency} Hz worker is still running; cannot resume twice.')
        _run_queue(config)


def _run_queue(config):
    frequencies=config['frequencies_hz']
    completed={f:validate_completed(f) for f in frequencies}
    # Old completed batches can be viewed after reporting code changes. Pending
    # simulations still require the exact prepared source/config/data receipt.
    if not all(completed.values()) and read_saved(ROOT/'prepared.json')!=preflight():
        raise ValueError('prepared source/config/data changed; use a new experiment folder instead of mixing different models')
    write_json(ROOT/'queue-owner.json',process_owner(),overwrite=True)
    write_json(ROOT/'queue-state.json',dict(status='running'),overwrite=True)
    summaries=[]
    def announce(message):
        log=ROOT/'logs/queue.log';log.parent.mkdir(exist_ok=True)
        with log.open('a',encoding='utf-8') as stream:
            stream.write(message+'\n')
        print(message,flush=True)
    export_queue_results(ROOT,frequencies)
    start=config.get('start_experiment',1);total=start+len(frequencies)-1
    active=None
    try:
        for index,frequency in enumerate(frequencies,start):
            result=completed[frequency]
            trial=ROOT/f'{frequency//1000}khz'
            if result is None:
                if (ROOT/'STOP_AFTER_CURRENT').exists():
                    write_json(ROOT/'queue-state.json',dict(status='stopped_between_experiments'),overwrite=True)
                    announce('Stopped between experiments. Completed results retained.');return
                archive_attempt(trial)
                active=trial
                trial_status(trial,'running',frequency_hz=frequency)
                export_queue_results(ROOT,frequencies)
                announce(f'Starting experiment {index}/{total}: {frequency//1000} kHz')
                command=[sys.executable,'-m','examples.resolved_pzt_queue','--root',str(ROOT),
                         '--worker',str(frequency),str(trial)]
                report=run_monitored(command,PROJECT,{
                    **os.environ,'CUPY_CACHE_DIR':str(ROOT/'cache/cupy')},
                    PeakGuard(queue_resource_limits(),telemetry_errors_fatal=False),trial/'worker.log',max_log_mib=None)
                write_json(trial/'launcher.json',report)
                if report['exit_code']!=0:
                    trial_status(trial,'failed',reason=f'Worker exit code {report["exit_code"]}; see worker.log')
                    raise RuntimeError(f'{frequency} Hz failed; queue stopped')
                result=validate_completed(frequency)
                if result is None:
                    raise RuntimeError('worker did not save a completed comparison; queue stopped')
                trial_status(trial,'completed',frequency_hz=frequency)
                active=None
            else:
                announce(f'Keeping completed experiment {index}/{total}: {frequency//1000} kHz')
            export_queue_results(ROOT,frequencies)
            announce('\n'.join(result_lines(frequency,result,result_directory(trial))))
            summaries.append(dict(frequency_hz=frequency,wall_seconds=result['wall_seconds'],comparison=result['comparison']))
        write_json(ROOT/'all-results.json',summaries,overwrite=True)
        write_json(ROOT/'queue-state.json',dict(status='completed'),overwrite=True)
        announce(f'All {len(frequencies)} experiments saved and compared. Results: {ROOT}/all-results.txt')
    except BaseException as exc:
        if active is not None:
            status=read_saved(active/'status.json') or {}
            if status.get('status')!='failed':
                trial_status(active,'interrupted',reason=f'{type(exc).__name__}: {exc}')
        write_json(ROOT/'queue-state.json',dict(status='interrupted',reason=f'{type(exc).__name__}: {exc}'),overwrite=True)
        raise
    finally:
        try:
            export_queue_results(ROOT,frequencies)
        finally:
            owner=read_saved(ROOT/'queue-owner.json') or process_owner()
            write_json(ROOT/'queue-owner.json',{**owner,'finished_unix':time.time()},overwrite=True)


def main():
    global ROOT
    parser=argparse.ArgumentParser(__doc__);actions=parser.add_mutually_exclusive_group()
    parser.add_argument('--root',type=Path,default=ROOT,help='experiment folder; reuse it to resume, choose a new one for a different setup')
    actions.add_argument('--prepare',action='store_true');actions.add_argument('--run',action='store_true')
    actions.add_argument('--worker',nargs=2);actions.add_argument('--smoke',type=Path,
        help='run 128 steps on full geometry, save to a new directory, then stop')
    args=parser.parse_args()
    ROOT=args.root.resolve()
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
