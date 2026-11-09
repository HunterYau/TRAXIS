"""Measure enriched GPU operators and complete mechanical observation windows."""

import argparse
import json
import math
import os
from pathlib import Path
import sys
import time
import zipfile

import numpy as np

from examples.matrixfree_step import INPUTS,_json,capacity_context,driver_hashes,load_mesh
from traxis.config import positive
from traxis.elmer_plate_refinement import reference_source_hashes
from traxis.elmer_reference import run_monitored
from traxis.enriched_factor import streamed_action
from traxis.enriched_tet import build_topology
from traxis.fem import tone_burst
from traxis.gpu_enriched import build_gpu_enriched
from traxis.gpu_enriched_verlet import gpu_explicit_simulate
from traxis.gpu_verification import gpu_package_identity
from traxis.highorder_patches import highorder_surface_patch
from traxis.highorder_verification import distribution_identity
from traxis.resources import ResourceGuard,ResourceLimitError,ResourceLimits
from traxis.storage import atomic_write,canonical_hash,sha256_file,write_json


def identity():
    return {**driver_hashes(),'enriched_gpu_wave.py':sha256_file(Path(__file__))}


def checked_settings(values,plate):
    fields = {'duration_s','frequency_hz','cycles','peak_patch_force_n','source_center_xy_m',
        'receiver_centers_xy_m','radius_m','cfl_fraction','gpu','operator_tolerance','energy_tolerance'}
    if set(values)!=fields:
        raise ValueError('enriched waveform settings have missing or unknown fields')
    for name in ('duration_s','frequency_hz','peak_patch_force_n','radius_m','cfl_fraction'):
        positive(name,values[name])
    gpu = values['gpu']
    if (set(gpu)!={'max_device_mib','chunk_elements','device_id'}
            or not math.isfinite(gpu['max_device_mib']) or not 0<gpu['max_device_mib']<=6144
            or type(gpu['chunk_elements']) is not int or not 1<=gpu['chunk_elements']<=1024
            or type(gpu['device_id']) is not int or not 0<=gpu['device_id']<=31
            or type(values['cycles']) is not int or not 1<=values['cycles']<=10
            or values['duration_s']>1e-3 or values['duration_s']<2*values['cycles']/values['frequency_hz']
            or values['cfl_fraction']>.95 or values['operator_tolerance']!=1e-10 or values['energy_tolerance']!=1e-8):
        raise ValueError('enriched waveform duration, CFL, GPU limits or criteria are invalid')
    centers = np.asarray([values['source_center_xy_m'],*values['receiver_centers_xy_m']])
    if (np.iscomplexobj(centers) or centers.ndim!=2 or centers.shape[1]!=2
            or not 2<=len(centers)<=9 or not np.isfinite(centers).all()
            or np.any(centers<values['radius_m'])
            or np.any(centers>np.array([plate.width_m,plate.height_m])-values['radius_m'])
            or len(np.unique(centers,axis=0))!=len(centers)):
        raise ValueError('source and receivers need distinct interior circular footprints')
    return values


def physical_field(points,plate):
    x,y,z = (points/np.array([plate.width_m,plate.height_m,plate.thickness_m])).T
    return np.array([np.sin(2*np.pi*x)+.17*np.cos(3*np.pi*z),
        .5*np.cos(np.pi*y)*np.sin(np.pi*z),.25*np.sin(np.pi*(x+y))*np.cos(np.pi*z)]).ravel()


def volume_moments(system,plate,material):
    lengths = np.array([plate.width_m,plate.height_m,plate.thickness_m])
    mass = system.mass_diagonal[:len(system.points_m)]
    rows = []
    for degree in range(3):
        for i in range(degree+1):
            for j in range(degree-i+1):
                powers = np.array([i,j,degree-i-j])
                measured = float(mass@np.prod(system.points_m**powers,axis=1))
                expected = float(material.density_kg_m3*np.prod(lengths)*np.prod(lengths**powers/(powers+1)))
                rows.append(dict(powers=powers.tolist(),measured=measured,expected=expected,relative_error=abs(measured/expected-1)))
    if max(r['relative_error'] for r in rows)>1e-10:
        raise ArithmeticError('native-order enriched positive mass fails intact-box moments')
    return rows


def circular_patch(system,plate,center,radius,settings,guard):
    vector,report = highorder_surface_patch(system,plate,center,radius,guard=guard,trace_degree=3,
        max_scalar_dofs=settings.max_dofs//3,max_tetrahedra=settings.max_tetrahedra,max_workspace_mib=256)
    weights = vector.reshape(3,-1)[2]
    selected = np.flatnonzero(weights)
    xy = (system.points_m[selected,:2]-center)/radius
    moments = []
    for a,b,expected in ((0,0,1.),(1,0,0.),(0,1,0.),(2,0,.25),(1,1,0.),(0,2,.25)):
        measured = float(weights[selected]@(xy[:,0]**a*xy[:,1]**b))
        moments.append(dict(powers=[a,b],expected=expected,measured=measured,absolute_error=abs(measured-expected)))
    report['nodal_polynomial_moments'] = moments
    if report['full_disk_area_relative_error']>1e-10 or max(r['absolute_error'] for r in moments)>1e-9:
        raise ArithmeticError('enriched circular footprint fails area or represented quadratic moments')
    return vector,report


def worker(plan_path,output,guard):
    plan_path,output = Path(plan_path).resolve(),Path(output).resolve()
    plan = _json(plan_path)
    if (plan['kind']!='enriched_gpu_mechanical_window' or plan['mode'] not in ('operator','wave')
            or plan['source_sha256']!=reference_source_hashes() or plan['driver_sha256']!=identity()
            or plan['runtime']!=distribution_identity(guard) or plan['gpu_packages']!=gpu_package_identity(guard)):
        raise ValueError('enriched GPU measurement source/runtime/plan changed')
    root = Path(plan['capacity_directory'])
    if set(plan['input_sha256'])!=set(INPUTS) or any(sha256_file(root/n)!=h for n,h in plan['input_sha256'].items()):
        raise ValueError('retained intact capacity inputs changed')
    context = capacity_context(root)
    _,_,plate,material,settings = context
    cfg = checked_settings(plan['settings'],plate)
    if not output.is_relative_to(plan_path.parent) or (output.exists() and any(output.iterdir())):
        raise ValueError('enriched GPU output must be fresh and within its plan directory')
    output.mkdir(parents=True,exist_ok=True)
    started = time.perf_counter()
    report = dict(completed=False,passed=False,ready_for_training_data=False,training_updates=0,
        mode=plan['mode'],requested_duration_s=cfg['duration_s'],steps_completed=0,full_window_completed=False,
        operator_check_passed=False,plan_fingerprint=plan['fingerprint'],stages=[])
    def progress(stage,**fields):
        report['stages'].append(dict(stage=stage,elapsed_seconds=time.perf_counter()-started,**fields))
        write_json(output/'progress.json',report,overwrite=True)
    try:
        mesh = load_mesh(root,context)
        progress('mesh_loaded',vertices=len(mesh.points_m),tetrahedra=len(mesh.tetrahedra))
        last_upload = -math.inf
        def uploaded(event):
            nonlocal last_upload
            elapsed = time.perf_counter()-started
            if elapsed-last_upload>=1 or event['elements_ready']==event['total_elements']:
                progress('elements_ready',**event)
                last_upload = elapsed
        system = build_gpu_enriched(mesh,material,settings,guard=guard,progress=uploaded,**cfg['gpu'])
        report.update(screen=system.screen,device=system.device,stable_dt_s=system.stable_dt_s,
            mass_moments=volume_moments(system,plate,material),vector_dofs=system.space.ndof)
        dt_limit = min(cfg['cfl_fraction']*system.stable_dt_s,.05/cfg['frequency_hz'])
        steps = math.ceil(cfg['duration_s']/dt_limit)
        dt = cfg['duration_s']/steps
        report.update(planned_steps=steps,dt_s=dt)
        if steps>500000:
            raise MemoryError('declared observation window exceeds the explicit step bound')
        progress('operator_ready',stable_dt_s=system.stable_dt_s,planned_steps=steps,dt_s=dt)
        field = physical_field(system.points_m,plate)
        before = time.perf_counter()
        gpu_action = system.stiffness@field
        host_action_seconds = time.perf_counter()-before
        storage,cp = system.stiffness.storage,system.stiffness.storage.cp
        with storage.lock,cp.cuda.Device(storage.device_id):
            device_field = cp.asarray(field)
            device_action = cp.empty_like(device_field)
            cp.cuda.get_current_stream().synchronize()
            first,last = cp.cuda.Event(),cp.cuda.Event()
            before = time.perf_counter()
            first.record()
            for _ in range(20):
                storage._launch(device_field,device_action,1.,0.)
            last.record()
            last.synchronize()
            device_wall_seconds = (time.perf_counter()-before)/20
            device_event_seconds = cp.cuda.get_elapsed_time(first,last)/20000
            del device_field,device_action
        topology = build_topology(mesh,max_vertices=settings.max_vertices,max_tetrahedra=settings.max_tetrahedra,
            max_dofs=settings.max_dofs,max_workspace_mib=settings.max_topology_mib)
        before = time.perf_counter()
        reference_topology = streamed_action(topology,material,physical_field(topology.points_m,plate),guard,
            chunk_elements=cfg['gpu']['chunk_elements'])
        n = len(system.points_m)
        reference = reference_topology.reshape(3,n)[:,system.native_to_topology].ravel()
        discrepancy = float(np.linalg.norm(gpu_action-reference)/np.linalg.norm(reference))
        report['operator'] = dict(relative_l2=discrepancy,host_action_seconds=host_action_seconds,
            device_action_wall_seconds=device_wall_seconds,device_action_event_seconds=device_event_seconds,
            timed_device_actions=20,cpu_reference_seconds=time.perf_counter()-before)
        atomic_write(output/'operator-comparison.npz',lambda stream:np.savez_compressed(stream,
            input_field=field,gpu_action=gpu_action,reference_action=reference,native_to_topology=system.native_to_topology,
            points_m=system.points_m,nodal_mass_kg=system.mass_diagonal[:n]))
        report['operator_comparison_sha256'] = sha256_file(output/'operator-comparison.npz')
        if not math.isfinite(discrepancy) or discrepancy>cfg['operator_tolerance']:
            raise ArithmeticError('full enriched GPU action disagrees with independent streamed CPU stiffness')
        report['operator_check_passed'] = True
        progress('operator_verified',**report['operator'])
        del reference_topology,reference,gpu_action,field
        if plan['mode']=='wave':
            load,source_info = circular_patch(system,plate,cfg['source_center_xy_m'],cfg['radius_m'],settings,guard)
            probes,probe_info = [],[]
            for center in cfg['receiver_centers_xy_m']:
                vector,info = circular_patch(system,plate,center,cfg['radius_m'],settings,guard)
                probes.append(vector)
                probe_info.append(info)
            probes = np.asarray(probes)
            report.update(source=source_info,receivers=probe_info)
            atomic_write(output/'mechanical-patches.npz',lambda stream:np.savez_compressed(stream,load=load,probes=probes))
            report['mechanical_patches_sha256'] = sha256_file(output/'mechanical-patches.npz')
            report['estimated_stiffness_seconds'] = (steps+1)*device_wall_seconds
            remaining = guard.limits.max_seconds-(time.monotonic()-guard.started)
            if report['estimated_stiffness_seconds']>remaining:
                raise ResourceLimitError('measured stiffness cost alone exceeds the remaining full-window guard')
            progress('mechanical_patches_ready',estimated_stiffness_seconds=report['estimated_stiffness_seconds'])
            last_step = -math.inf
            def stepped(event):
                nonlocal last_step
                report['steps_completed'] = event['steps_completed']
                elapsed = time.perf_counter()-started
                if elapsed-last_step>=1 or event['steps_completed']==steps:
                    progress('propagating',**event)
                    last_step = elapsed
            before = time.perf_counter()
            result = gpu_explicit_simulate(system,steps,dt,probes,load,
                force=lambda t:cfg['peak_patch_force_n']*tone_burst(t,cfg['frequency_hz'],cfg['cycles']),
                guard=guard,progress=stepped)
            propagation_seconds = time.perf_counter()-before
            atomic_write(output/'mechanical-window.npz',lambda stream:np.savez_compressed(stream,
                time_s=result.time_s,displacement_m=result.displacement_m,energy_time_s=result.energy_time_s,
                physical_energy_j=result.physical_energy_j,modified_energy_j=result.modified_energy_j,
                final_displacement_m=result.final_displacement_m,final_velocity_m_s=result.final_velocity_m_s))
            report['mechanical_window_sha256'] = sha256_file(output/'mechanical-window.npz')
            after = result.modified_energy_j[result.energy_time_s>cfg['cycles']/cfg['frequency_hz']+2*dt]
            if len(after)<2 or after.mean()<=0:
                raise ArithmeticError('full-window postburst energy is missing or nonpositive')
            energy_range = float(np.ptp(after)/abs(after.mean()))
            report.update(propagation_seconds=propagation_seconds,propagation_statistics=result.statistics,
                postburst_modified_energy_relative_range=energy_range,
                peak_receiver_displacement_m=np.max(abs(result.displacement_m),axis=0).tolist())
            if energy_range>cfg['energy_tolerance'] or not np.any(result.displacement_m):
                raise ArithmeticError('full-window energy or nonzero receiver check failed')
            if result.statistics['steps_completed']!=steps or result.time_s[-1]!=steps*dt:
                raise ArithmeticError('explicit driver did not complete its declared observation window')
            report['full_window_completed'] = True
            progress('window_complete',propagation_seconds=propagation_seconds)
        if (plan['source_sha256']!=reference_source_hashes() or plan['driver_sha256']!=identity()
                or plan['runtime']!=distribution_identity(guard) or plan['gpu_packages']!=gpu_package_identity(guard)):
            raise ValueError('enriched GPU source/runtime changed during measurement')
        report.update(completed=True,passed=True,
            scope='Declared enriched mechanical operator/window and energy checks only; no mesh/time, defect or PZT/physical acceptance.')
    except Exception as error:
        report['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        report['wall_seconds'] = time.perf_counter()-started
        report['resources'] = guard.report()
        report['fingerprint'] = canonical_hash(report)
        write_json(output/'wave-report.json',report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capacity',type=Path)
    parser.add_argument('--settings',type=Path,default=Path(__file__).resolve().parents[1]/'configs/enriched-gpu-wave.json')
    parser.add_argument('--mode',choices=('operator','wave'),default='operator')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--plan',type=Path)
    parser.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    parser.add_argument('--max-seconds',type=float,default=600)
    parser.add_argument('--max-rss-mib',type=float,default=1024)
    args = parser.parse_args()
    guard = ResourceGuard(ResourceLimits(max_seconds=args.max_seconds,max_rss_mib=args.max_rss_mib,
        max_system_memory_percent=85,min_available_mib=2048))
    if args.worker:
        return 0 if worker(args.plan,args.output,guard)['passed'] else 2
    if not args.capacity:
        parser.error('provide the retained intact capacity directory')
    root,output = args.capacity.resolve(),args.output.resolve()
    context = capacity_context(root)
    cfg = checked_settings(json.loads(args.settings.read_text()),context[2])
    if output.exists() and any(output.iterdir()):
        parser.error('enriched GPU measurement needs a fresh output directory')
    output.mkdir(parents=True,exist_ok=True)
    result = dict(completed=False,passed=False,ready_for_training_data=False,training_updates=0,mode=args.mode)
    try:
        plan = dict(kind='enriched_gpu_mechanical_window',mode=args.mode,settings=cfg,capacity_directory=str(root),
            source_sha256=reference_source_hashes(),driver_sha256=identity(),runtime=distribution_identity(guard),
            gpu_packages=gpu_package_identity(guard),input_sha256={n:sha256_file(root/n) for n in INPUTS})
        plan['fingerprint'] = canonical_hash(plan)
        write_json(output/'enriched-gpu-plan.json',plan)
        def snapshot(stream):
            with zipfile.ZipFile(stream,'w',compression=zipfile.ZIP_DEFLATED) as archive:
                for name in plan['source_sha256']:
                    archive.writestr('traxis/'+name,(Path(__file__).resolve().parents[1]/'traxis'/name).read_bytes())
                for name in plan['driver_sha256']:
                    archive.writestr('examples/'+name,(Path(__file__).parent/name).read_bytes())
        atomic_write(output/'source-snapshot.zip',snapshot)
        result.update(plan_fingerprint=plan['fingerprint'],source_snapshot_sha256=sha256_file(output/'source-snapshot.zip'))
        temporary,cache = output/'temporary',output/'kernel-cache'
        temporary.mkdir()
        cache.mkdir()
        remaining = max(1.,guard.limits.max_seconds-(time.monotonic()-guard.started))
        command = [sys.executable,'-m','examples.enriched_gpu_wave','--worker','--plan',str(output/'enriched-gpu-plan.json'),
            '--output',str(output/'native'),'--max-seconds',str(remaining),'--max-rss-mib',str(guard.limits.max_rss_mib)]
        env = {**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','CUPY_CACHE_DIR':str(cache),
            'CUPY_CACHE_SAVE_CUDA_SOURCE':'1','TEMP':str(temporary),'TMP':str(temporary),'TMPDIR':str(temporary)}
        result['worker_run'] = run_monitored(command,Path(__file__).resolve().parents[1],env,guard,output/'worker.log')
        if result['worker_run']['exit_code']!=0:
            raise RuntimeError('enriched GPU worker stopped; inspect progress/report and worker log')
        native = _json(output/'native/wave-report.json')
        if (native['completed'] is not True or native['passed'] is not True or native['plan_fingerprint']!=plan['fingerprint']
                or native['operator_check_passed'] is not True or native['mode']!=args.mode
                or (args.mode=='wave' and native['full_window_completed'] is not True)):
            raise ValueError('enriched GPU measurement receipt is incomplete or changed')
        artifacts = {'operator-comparison.npz':'operator_comparison_sha256'}
        if args.mode=='wave':
            artifacts.update({'mechanical-patches.npz':'mechanical_patches_sha256','mechanical-window.npz':'mechanical_window_sha256'})
        if any(sha256_file(output/'native'/name)!=native[key] for name,key in artifacts.items()):
            raise ValueError('enriched GPU retained arrays changed')
        if (plan['source_sha256']!=reference_source_hashes() or plan['driver_sha256']!=identity()
                or plan['runtime']!=distribution_identity(guard) or plan['gpu_packages']!=gpu_package_identity(guard)):
            raise ValueError('enriched GPU parent source/runtime changed')
        result.update(completed=True,passed=True,vector_dofs=native['vector_dofs'],
            full_window_completed=native['full_window_completed'],steps_completed=native['steps_completed'],
            native_report_sha256=sha256_file(output/'native/wave-report.json'))
    except Exception as error:
        result['error'] = f'{type(error).__name__}: {error}'
    finally:
        result['resources'] = guard.report()
        result['fingerprint'] = canonical_hash(result)
        write_json(output/'monitor-result.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('resources','fingerprint')}))
    return 0 if result['passed'] else 1


if __name__=='__main__':
    raise SystemExit(main())
