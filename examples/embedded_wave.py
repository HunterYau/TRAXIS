"""Frozen, guarded mechanical wave windows on the embedded recess grid."""
import argparse
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import sys
import time
import zipfile

import numpy as np
import psutil

from examples.tensor_wave import (PROJECT, archive_schema, checked_receipt, identities,
    load_arrays, read_json, sealed, snapshot_sources)
from traxis.config import CircularRecess, Material, Plate
from traxis.elmer_reference import run_monitored
from traxis.embedded_recess import make_embedded_recess_mesh, storage_screen
from traxis.fem import tone_burst
from traxis.gpu_enriched_verlet import gpu_explicit_simulate
from traxis.gpu_mapped_tensor import build_gpu_mapped
from traxis.gpu_verification import gpu_package_identity
from traxis.highorder_verification import distribution_identity
from traxis.mapped_patches import circular_patch
from traxis.mapped_tensor import build_system
from traxis.resources import ResourceGuard, ResourceLimits
from traxis.storage import atomic_write, sha256_file, write_json


FULL_LEVELS = ((3,196608), (4,131072), (4,196608), (4,294912), (5,196608))


def case_config(scope='smoke', order=4, steps=None, defective=False):
    if type(order) is not int or type(defective) is not bool:
        raise ValueError('wave degree and material state need explicit types')
    if scope == 'smoke':
        steps = 8064 if steps is None else steps
        valid = order == 4 and type(steps) is int and steps in (8064,16128)
        geometry = dict(plate=asdict(Plate(.08,.06,.003175)), recess=asdict(CircularRecess(.041,.029,.008,.0015)),
            angular_cells=2, radial_cells=[1,1], outer_cell_size_m=.04, duration_s=.0000315,
            source_center_xy_m=[.006,.006], receiver_centers_xy_m=[[.04,.03],[.074,.054],[.06,.015]])
    elif scope == 'full':
        steps = 131072 if steps is None else steps
        valid = type(steps) is int and (order,steps) in FULL_LEVELS
        geometry = dict(plate=asdict(Plate(.5,.5,.003175)), recess=asdict(CircularRecess(.255,.245,.015,.0015)),
            angular_cells=12, radial_cells=[2,6], outer_cell_size_m=.005, duration_s=.0005,
            source_center_xy_m=[.02,.02], receiver_centers_xy_m=[[.25,.25],[.48,.48],[.35,.20]])
    else:
        raise ValueError('unknown embedded window scope')
    if not valid:
        raise ValueError('embedded window is not one of the declared refinement levels')
    return dict(kind='embedded_mechanical_window_case', scope=scope, order=order, steps=steps,
        defective=defective, **geometry, material=asdict(Material()), geometry_order=3, layers=[1,1],
        radius_m=.003, frequency_hz=100000., cycles=2, force_scale_n=1., output_stride=1, energy_stride=256,
        max_mesh_bytes=512*1024**2, max_operator_bytes=1536*1024**2, max_device_mib=3072, max_host_mib=2048,
        max_workspace_mib=1536, operator_tolerance=1e-10, energy_tolerance=1e-8,
        cooling_upper_c=76., cooling_lower_c=72.)


def checked_case(case):
    if case != case_config(case['scope'],case['order'],case['steps'],case['defective']):
        raise ValueError('embedded window physics, geometry, guards or criteria changed')
    return case


def mesh_controls(case):
    return dict(geometry_order=case['geometry_order'], angular_cells=case['angular_cells'],
        radial_cells=tuple(case['radial_cells']), layers=tuple(case['layers']), defective=case['defective'],
        outer_cell_size_m=case['outer_cell_size_m'], max_bytes=case['max_mesh_bytes'])


def screen(case):
    checked_case(case)
    return storage_screen(Plate(**case['plate']),CircularRecess(**case['recess']),case['order'],**mesh_controls(case))


def validate_plan(plan, guard):
    checked_receipt(plan); checked_case(plan['case'])
    if (plan['kind'] != 'embedded_mechanical_window' or plan['mesh_screen'] != screen(plan['case'])
            or plan['declared_full_levels'] != [list(p) for p in FULL_LEVELS]
            or any(plan[k] != v for k,v in identities().items())
            or any(sha256_file(PROJECT/name) != digest for name,digest in plan['analysis_sources'].items())
            or plan['runtime'] != distribution_identity(guard) or plan['gpu_packages'] != gpu_package_identity(guard)):
        raise ValueError('embedded window source, runtime or frozen settings changed')


class ThermalPause:
    """Pause host submission between synchronized batches; physical time stays fixed."""
    def __init__(self, guard, upper, lower, *, sleep=time.sleep, clock=time.monotonic):
        self.guard,self.upper,self.lower,self.sleep,self.clock = guard,upper,lower,sleep,clock
        self.events=[]; self.pause_seconds=0.

    def temperature(self):
        if len(self.guard.last_gpus) != 1:
            raise RuntimeError('embedded cooling needs the observed single GPU temperature sensor')
        return self.guard.last_gpus[0]['temperature_c']

    def check(self, steps):
        temperature=self.temperature()
        if temperature < self.upper:
            return
        before=self.clock(); event=dict(steps_completed=steps,start_temperature_c=temperature)
        print(json.dumps(dict(phase='cooling',**event)),flush=True)
        while self.temperature() > self.lower:
            self.sleep(.25); self.guard.check(force=True)
        event.update(pause_seconds=self.clock()-before,end_temperature_c=self.temperature())
        self.pause_seconds += event['pause_seconds']; self.events.append(event)
        print(json.dumps(dict(phase='cooled',**event)),flush=True)


def worker(plan_path, output, guard):
    plan=checked_receipt(read_json(plan_path)); case=checked_case(plan['case']); output.mkdir()
    report=dict(completed=False,passed=False,ready_for_training_data=False,training_updates=0,
        full_window_completed=False,steps_completed=0,planned_steps=case['steps'],plan_fingerprint=plan['fingerprint'])
    started=time.perf_counter(); last=-math.inf
    def phase(name, **extra):
        print(json.dumps(dict(phase=name,elapsed_seconds=time.perf_counter()-started,**extra)),flush=True)
    try:
        validate_plan(plan,guard); guard.check(force=True)
        if psutil.disk_usage(str(output)).free < 2048*1024**2:
            raise MemoryError('embedded wave proof needs at least 2 GiB free disk')
        mesh=make_embedded_recess_mesh(Plate(**case['plate']),CircularRecess(**case['recess']),
            case['order'],guard=guard,**mesh_controls(case)); phase('mesh_built')
        cpu=build_system(mesh,Material(**case['material']),guard,max_bytes=case['max_operator_bytes'],max_batch_vectors=1)
        system=build_gpu_mapped(cpu,guard=guard,max_device_mib=case['max_device_mib'],max_host_mib=case['max_host_mib'])
        dt=case['duration_s']/case['steps']
        if dt > .8*system.stable_dt_s:
            raise ValueError('declared wave step exceeds 80% of the conservative stability bound')
        report.update(system=system.screen,device=system.device,dt_s=dt,vector_dofs=3*len(mesh.points_m))
        phase('operator_ready',vector_dofs=report['vector_dofs'],stable_dt_s=system.stable_dt_s)
        patches=[circular_patch(mesh,center,case['radius_m'],guard=guard) for center in
            [case['source_center_xy_m'],*case['receiver_centers_xy_m']]]
        patch_arrays={}
        for i,p in enumerate(patches):
            patch_arrays[f'patch{i}_indices']=p.indices.astype(np.int64)
            patch_arrays[f'patch{i}_weights']=p.weights
            patch_arrays[f'patch{i}_direction']=p.direction
        atomic_write(output/'mechanical-patches.npz',lambda f:np.savez_compressed(f,**patch_arrays))
        report['patches']=[p.report for p in patches]
        report['mechanical_patches_sha256']=sha256_file(output/'mechanical-patches.npz')
        ns=len(mesh.points_m); load=patches[0].vector(ns); probes=np.zeros((3,3*ns))
        for row,p in zip(probes,patches[1:]):
            row.reshape(3,ns)[:,p.indices]=p.direction[:,None]*p.weights
        cooling=ThermalPause(guard,case['cooling_upper_c'],case['cooling_lower_c']); cooling.temperature()
        phase('patches_ready')
        def progressed(event):
            nonlocal last
            report['steps_completed']=event['steps_completed']; cooling.check(event['steps_completed'])
            elapsed=time.perf_counter()-started
            if elapsed-last >= 15 or event['steps_completed']==case['steps']:
                phase('propagating',**event); last=elapsed
        before=time.perf_counter()
        result=gpu_explicit_simulate(system,case['steps'],dt,probes,load,
            force=lambda t:case['force_scale_n']*tone_burst(t,case['frequency_hz'],case['cycles']),
            output_stride=case['output_stride'],energy_stride=case['energy_stride'],guard=guard,progress=progressed,
            max_workspace_mib=case['max_workspace_mib'])
        report['propagation_seconds']=time.perf_counter()-before
        final_force=system.stiffness@result.final_displacement_m
        atomic_write(output/'mechanical-window.npz',lambda f:np.savez_compressed(f,
            time_s=result.time_s,displacement_m=result.displacement_m,energy_time_s=result.energy_time_s,
            physical_energy_j=result.physical_energy_j,modified_energy_j=result.modified_energy_j,
            final_displacement_m=result.final_displacement_m,final_velocity_m_s=result.final_velocity_m_s,
            mass_diagonal_kg=system.mass_diagonal,final_gpu_force_n=final_force))
        report['mechanical_window_sha256']=sha256_file(output/'mechanical-window.npz')
        after=result.modified_energy_j[result.energy_time_s>case['cycles']/case['frequency_hz']+2*dt]
        if len(after)<2 or after.mean()<=0:
            raise ArithmeticError('embedded postburst energy is missing or nonpositive')
        variation=float(np.ptp(after)/abs(after.mean()))
        complete=(result.statistics['steps_completed']==case['steps'] and
                  abs(result.time_s[-1]-case['duration_s'])<1e-18)
        if not complete or variation>case['energy_tolerance'] or np.any(np.linalg.norm(result.displacement_m,axis=0)==0):
            raise ArithmeticError('embedded window, modified energy or receiver identifiability failed')
        validate_plan(plan,guard)
        report.update(completed=True,passed=True,full_window_completed=True,
            postburst_modified_energy_relative_range=variation,propagation_statistics=result.statistics,
            energy_samples=len(result.energy_time_s),cooling_events=cooling.events,cooling_pause_seconds=cooling.pause_seconds,
            peak_receiver_displacement_m=np.max(abs(result.displacement_m),axis=0).tolist(),
            scope='Complete embedded mechanical window only; no spatial/time/PZT or physical-data acceptance.')
    except Exception as error:
        report['error']=f'{type(error).__name__}: {error}'; raise
    finally:
        report['wall_seconds']=time.perf_counter()-started; report['resources']=guard.report()
        write_json(output/'wave-report.json',sealed(report))
    return report


def read_wave(root, guard=None, *, states=False):
    root=Path(root).resolve()
    plan,parent,native=(checked_receipt(read_json(root/p)) for p in
        ('embedded-wave-plan.json','monitor-result.json','native/wave-report.json'))
    case=checked_case(plan['case']); counts=screen(case); steps=case['steps']; n=counts['vector_dofs']
    if (any(not r['completed'] or not r['passed'] or r['ready_for_training_data'] or r['training_updates']!=0 for r in (parent,native))
            or parent['worker_run']['exit_code']!=0 or not native['full_window_completed']
            or parent['plan_fingerprint']!=plan['fingerprint'] or native['plan_fingerprint']!=plan['fingerprint']
            or parent['report_sha256']!=sha256_file(root/'native/wave-report.json')
            or native['steps_completed']!=steps or native['planned_steps']!=steps or native['vector_dofs']!=n
            or native['dt_s']!=case['duration_s']/steps or len(native['patches'])!=4):
        raise ValueError('embedded window receipts are incomplete or incompatible')
    energy_steps=np.r_[np.arange(0,steps+1,case['energy_stride']),steps] if steps%case['energy_stride'] else np.arange(0,steps+1,case['energy_stride'])
    if native['energy_samples']!=len(energy_steps): raise ValueError('embedded energy sample count changed')
    spec=dict(time_s=(steps+1,),displacement_m=(steps+1,3),energy_time_s=(len(energy_steps),),
        physical_energy_j=(len(energy_steps),),modified_energy_j=(len(energy_steps),),
        final_displacement_m=(n,),final_velocity_m_s=(n,),mass_diagonal_kg=(n,),final_gpu_force_n=(n,))
    patch_spec={f'patch{i}_{key}':shape for i,p in enumerate(native['patches'])
        for key,shape in [('indices',(p['scalar_support'],)),('weights',(p['scalar_support'],)),('direction',(3,))]}
    archive_schema(root/'native/mechanical-window.npz',native['mechanical_window_sha256'],spec)
    archive_schema(root/'native/mechanical-patches.npz',native['mechanical_patches_sha256'],patch_spec)
    selected=None if states else ('time_s','displacement_m','energy_time_s','physical_energy_j','modified_energy_j')
    arrays=load_arrays(root/'native/mechanical-window.npz',spec,selected)
    patches=load_arrays(root/'native/mechanical-patches.npz',patch_spec)
    if (not np.array_equal(arrays['time_s'],np.arange(steps+1)*(case['duration_s']/steps))
            or not np.array_equal(arrays['energy_time_s'],energy_steps*(case['duration_s']/steps))):
        raise ValueError('embedded trace or energy time grid changed')
    stats=native['propagation_statistics']
    if (stats['steps_completed']!=steps or stats['requested_steps']!=steps
            or abs(stats['duration_s']-case['duration_s'])>1e-18):
        raise ValueError('embedded integration statistics changed')
    after=arrays['modified_energy_j'][arrays['energy_time_s']>case['cycles']/case['frequency_hz']+2*case['duration_s']/steps]
    if len(after)<2 or after.mean()<=0 or np.ptp(after)/abs(after.mean())>case['energy_tolerance']:
        raise ValueError('embedded saved postburst energy failed')
    if np.any(np.linalg.norm(arrays['displacement_m'],axis=0)==0):
        raise ValueError('embedded saved receiver traces are zero')
    for i in range(4):
        ids=patches[f'patch{i}_indices']; weights=patches[f'patch{i}_weights']; direction=patches[f'patch{i}_direction']
        if (ids.dtype.kind!='i' or len(ids)==0 or len(np.unique(ids))!=len(ids) or ids.min()<0 or ids.max()>=n//3
                or abs(float(weights.sum())-1)>1e-9 or not np.array_equal(direction,[0.,0.,1.])):
            raise ValueError('embedded saved patch topology, force or direction changed')
    if states and np.any(arrays['mass_diagonal_kg']<=0):
        raise ValueError('embedded saved mass is nonpositive')
    return dict(root=root,plan=plan,parent=parent,native=native,case=case,window=arrays,patches=patches,
        input_identities={p:sha256_file(root/p) for p in ('embedded-wave-plan.json','monitor-result.json','native/wave-report.json',
            'native/mechanical-window.npz','native/mechanical-patches.npz')})


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--scope',choices=('smoke','full'),default='smoke'); parser.add_argument('--order',type=int,default=4)
    parser.add_argument('--steps',type=int); parser.add_argument('--recess',action='store_true')
    parser.add_argument('--worker',action='store_true'); parser.add_argument('--plan',type=Path)
    args=parser.parse_args(); output=args.output.resolve()
    case=checked_case(read_json(args.plan)['case']) if args.worker else case_config(args.scope,args.order,args.steps,args.recess)
    guard=ResourceGuard(ResourceLimits(max_seconds=2400 if case['scope']=='full' else 180,
        max_rss_mib=2048 if case['scope']=='full' else 1024,max_system_memory_percent=85,min_available_mib=2048))
    if args.worker: return 0 if worker(args.plan,output,guard)['passed'] else 2
    if not output.is_relative_to(PROJECT/'outputs'): raise ValueError('embedded windows must stay in project outputs')
    output.mkdir(parents=True,exist_ok=False)
    analysis={name:sha256_file(PROJECT/name) for name in ('examples/embedded_wave.py','examples/verify_embedded_window.py')}
    plan=dict(kind='embedded_mechanical_window',case=case,declared_full_levels=[list(p) for p in FULL_LEVELS],
        mesh_screen=screen(case),**identities(),analysis_sources=analysis,runtime=distribution_identity(guard),
        gpu_packages=gpu_package_identity(guard))
    snapshot_sources(output/'source-snapshot.zip',plan)
    with zipfile.ZipFile(output/'source-snapshot.zip','a',compression=zipfile.ZIP_DEFLATED) as archive:
        for name in analysis: archive.writestr(name,(PROJECT/name).read_bytes())
    plan['source_snapshot_sha256']=sha256_file(output/'source-snapshot.zip'); write_json(output/'embedded-wave-plan.json',sealed(plan))
    temporary=output/'temporary'; temporary.mkdir()
    env={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','CUPY_CACHE_DIR':str(output/'kernel-cache'),
        'CUPY_CACHE_SAVE_CUDA_SOURCE':'1',**{name:str(temporary) for name in ('TEMP','TMP','TMPDIR')}}
    result=dict(completed=False,passed=False,ready_for_training_data=False,training_updates=0,plan_fingerprint=plan['fingerprint'])
    try:
        command=[sys.executable,'-m','examples.embedded_wave','--worker','--plan',str(output/'embedded-wave-plan.json'),'--output',str(output/'native')]
        result['worker_run']=run_monitored(command,PROJECT,env,guard,output/'worker.log')
        native=checked_receipt(read_json(output/'native/wave-report.json'))
        if result['worker_run']['exit_code']!=0 or not native['passed'] or native['plan_fingerprint']!=plan['fingerprint']:
            raise ValueError('embedded wave worker failed; inspect retained log/report')
        for filename,key in [('mechanical-window.npz','mechanical_window_sha256'),('mechanical-patches.npz','mechanical_patches_sha256')]:
            if sha256_file(output/'native'/filename)!=native[key]: raise ValueError('embedded raw proof changed')
        validate_plan(plan,guard)
        if sha256_file(output/'source-snapshot.zip')!=plan['source_snapshot_sha256']: raise ValueError('embedded snapshot changed')
        result.update(completed=True,passed=True,report_sha256=sha256_file(output/'native/wave-report.json'))
    except Exception as error: result['error']=f'{type(error).__name__}: {error}'
    finally:
        result['resources']=guard.report(); write_json(output/'monitor-result.json',sealed(result))
    print(json.dumps({k:v for k,v in result.items() if k!='resources'})); return 0 if result['passed'] else 2


if __name__=='__main__': raise SystemExit(main())
