"""Check discrete Green replay against fresh or retained direct CUDA waveforms."""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import time
from types import SimpleNamespace
import zipfile

import numpy as np

from examples.discrete_green import synthesize
from examples.embedded_wave import ThermalPause, case_config, mesh_controls, read_wave
from examples.embedded_wave_evidence import validate_saved_plan
from examples.tensor_wave import PROJECT, checked_receipt, identities, read_json, sealed, snapshot_sources
from traxis.config import CircularRecess, Material, Plate
from traxis.embedded_recess import make_embedded_recess_mesh
from traxis.fem import tone_burst
from traxis.gpu_enriched_verlet import gpu_explicit_simulate, tensor_patch_probe_set
from traxis.gpu_mapped_tensor import build_gpu_mapped
from traxis.gpu_verification import gpu_package_identity
from traxis.highorder_verification import distribution_identity
from traxis.mapped_patches import circular_patch
from traxis.mapped_tensor import action, build_system
from traxis.resources import ResourceGuard, ResourceLimits
from traxis.storage import atomic_write, sha256_file, write_json


FREQUENCIES=(80000.,100000.,150000.)
WAVEFORM_TOLERANCE=1e-8


def reference_window(root,cpu_root,guard):
    root,cpu_root=Path(root).resolve(),Path(cpu_root).resolve()
    if not all(p.is_relative_to(PROJECT/'outputs') for p in (root,cpu_root)):
        raise ValueError('Green reference and CPU receipt must stay in project outputs')
    level=read_wave(root,guard);source=validate_saved_plan(level['plan'],root,guard)
    path=cpu_root/'cpu-window-check.json';cpu=checked_receipt(read_json(path))
    if (not cpu['completed'] or not cpu['passed'] or cpu['ready_for_training_data'] or cpu['training_updates']!=0
            or cpu['plan_fingerprint']!=level['plan']['fingerprint']
            or cpu['forward_result_fingerprint']!=level['parent']['fingerprint']
            or cpu['input_identities']!=level['input_identities'] or cpu['source_sha256']!=level['plan']['source_sha256']
            or cpu['checker_sha256']!=sha256_file(PROJECT/'examples/verify_embedded_window.py')):
        raise ValueError('Green reference requires its matching passing CPU check')
    return level,dict(directory=str(root),plan_fingerprint=level['plan']['fingerprint'],
        input_identities=level['input_identities'],source_check=source,cpu_path=str(path),cpu_sha256=sha256_file(path))


def verify(output,guard,reference_run=None,cpu_check=None):
    output=Path(output).resolve()
    if not output.is_relative_to(PROJECT/'outputs'):
        raise ValueError('Green verification must remain in project outputs')
    output.mkdir(parents=True,exist_ok=False)
    os.environ['CUPY_CACHE_DIR']=str(output/'kernel-cache')
    temporary=output/'temporary';temporary.mkdir()
    os.environ.update({key:str(temporary) for key in ('TEMP','TMP','TMPDIR')})
    if (reference_run is None)!=(cpu_check is None):
        raise ValueError('a retained Green reference needs both forward and CPU paths')
    reference,binding=reference_window(reference_run,cpu_check,guard) if reference_run is not None else (None,None)
    case=reference['case'] if reference else case_config();dt=case['duration_s']/case['steps']
    frequencies=(case['frequency_hz'],) if reference else FREQUENCIES
    analysis={name:sha256_file(PROJECT/name) for name in ('examples/discrete_green.py',
        'examples/verify_green_synthesis.py','examples/embedded_wave.py','examples/embedded_wave_evidence.py')}
    geometry={k:case[k] for k in ('plate','material','recess','order','geometry_order','angular_cells',
        'radial_cells','layers','outer_cell_size_m','source_center_xy_m','receiver_centers_xy_m','radius_m','defective')}
    plan=dict(kind='discrete_green_mechanical_verification',case=dict(geometry=geometry,
        steps=case['steps'],dt_s=dt,duration_s=case['duration_s'],impulse_force_index=1,
        impulse_force_n=1.,tone_frequencies_hz=list(frequencies),cycles=2,
        waveform_relative_l2_tolerance=WAVEFORM_TOLERANCE,operator_tolerance=1e-10,
        modified_energy_tolerance=1e-8),**identities(),analysis_sources=analysis,
        runtime=distribution_identity(guard),gpu_packages=gpu_package_identity(guard),limits=asdict(guard.limits),
        retained_reference=binding)
    snapshot_sources(output/'source-snapshot.zip',plan)
    with zipfile.ZipFile(output/'source-snapshot.zip','a',compression=zipfile.ZIP_DEFLATED) as archive:
        for name in analysis:archive.writestr(name,(PROJECT/name).read_bytes())
    plan['source_snapshot_sha256']=sha256_file(output/'source-snapshot.zip')
    write_json(output/'green-plan.json',sealed(plan))
    report=dict(completed=False,passed=False,ready_for_training_data=False,training_updates=0,
        plan_fingerprint=plan['fingerprint'],scope='Discrete-input reuse on a fixed mechanical mesh; no continuum/frequency/PZT/data-bank acceptance.')
    started=time.perf_counter()
    try:
        guard.check(force=True)
        mesh=make_embedded_recess_mesh(Plate(**case['plate']),CircularRecess(**case['recess']),
            case['order'],guard=guard,**mesh_controls(case))
        cpu=build_system(mesh,Material(**case['material']),guard,max_bytes=case['max_operator_bytes'],max_batch_vectors=1)
        system=build_gpu_mapped(cpu,guard=guard,max_device_mib=case['max_device_mib'],max_host_mib=case['max_host_mib'])
        if dt>.8*system.stable_dt_s:raise ValueError('Green step exceeds the control stability bound')
        if reference:
            patches=[SimpleNamespace(**{key:reference['patches'][f'patch{i}_{key}']
                for key in ('indices','weights','direction')}) for i in range(4)]
        else:
            patches=[circular_patch(mesh,center,case['radius_m'],guard=guard) for center in
                [case['source_center_xy_m'],*case['receiver_centers_xy_m']]]
        ns=len(mesh.points_m);load=np.zeros(3*ns)
        load.reshape(3,ns)[:,patches[0].indices]=patches[0].direction[:,None]*patches[0].weights
        for patch,center in zip(patches,[case['source_center_xy_m'],*case['receiver_centers_xy_m']]):
            if (np.max(abs(mesh.points_m[patch.indices,2]-case['plate']['thickness_m']))>1e-14
                    or abs(float(patch.weights.sum())-1)>1e-9
                    or np.max(abs(patch.weights@mesh.points_m[patch.indices,:2]-center))/case['radius_m']>1e-9):
                raise ValueError('Green source/probe physical topology or moments changed')
        probes=tensor_patch_probe_set(patches[1:],ns)
        cooling=ThermalPause(guard,76.,72.);guard.check(force=True);cooling.temperature()
        def run(force):
            cleaned=False
            def progressed(event):
                nonlocal cleaned
                if not cleaned:
                    system.stiffness.storage.cp.get_default_pinned_memory_pool().free_all_blocks();cleaned=True
                cooling.check(event['steps_completed'])
            before=time.perf_counter()
            result=gpu_explicit_simulate(system,case['steps'],dt,probes,load,force=force,
                output_stride=1,energy_stride=case['energy_stride'],guard=guard,progress=progressed,
                max_workspace_mib=case['max_workspace_mib'])
            return result,time.perf_counter()-before
        green,green_seconds=run(lambda t:1. if t==dt else 0.)
        time_s=green.time_s
        if reference and not np.array_equal(time_s,reference['window']['time_s']):
            raise ValueError('Green and retained direct windows use different time grids')
        green_force=np.zeros(len(time_s));green_force[1]=1.
        cpu_force=action(cpu,green.final_displacement_m,guard)
        gpu_force=system.stiffness@green.final_displacement_m
        relative=lambda a,b:float(np.linalg.norm(a-b)/np.linalg.norm(b))
        force_error=relative(gpu_force,cpu_force);mass_error=relative(system.mass_diagonal,cpu.mass_kg)
        cpu_probes=np.array([float(p.direction@green.final_displacement_m.reshape(3,ns)[:,p.indices]@p.weights) for p in patches[1:]])
        probe_error=relative(green.displacement_m[-1],cpu_probes)
        physical=float((green.final_velocity_m_s@(cpu.mass_kg*green.final_velocity_m_s)+green.final_displacement_m@cpu_force)/2)
        modified=physical-dt**2/8*float(cpu_force@(cpu_force/cpu.mass_kg))
        energy_errors=[abs(green.physical_energy_j[-1]/physical-1),abs(green.modified_energy_j[-1]/modified-1)]
        def energy_range(result,end):
            values=result.modified_energy_j[result.energy_time_s>end+2*dt]
            if len(values)<2 or values.mean()<=0:raise ArithmeticError('post-input energy is missing/nonpositive')
            return float(np.ptp(values)/abs(values.mean()))
        green_energy=energy_range(green,dt)
        rows=[];direct=[];replayed=[];inputs=[]
        for frequency in frequencies:
            guard.check(force=True)
            force=lambda t,frequency=frequency:tone_burst(t,frequency,2)
            if reference:
                waveform=SimpleNamespace(**reference['window']);direct_seconds=None
            else:waveform,direct_seconds=run(force)
            forcing=np.array([force(t) for t in time_s]);before=time.perf_counter()
            replay,stats=synthesize(green.displacement_m,forcing)
            replay_seconds=time.perf_counter()-before
            errors=np.linalg.norm(replay-waveform.displacement_m,axis=0)/np.linalg.norm(waveform.displacement_m,axis=0)
            variation=energy_range(waveform,2/frequency)
            passed=bool(np.isfinite(errors).all() and np.max(errors)<WAVEFORM_TOLERANCE and variation<1e-8)
            rows.append(dict(frequency_hz=frequency,relative_waveform_l2=errors.tolist(),passed=passed,
                direct_seconds=direct_seconds,replay_seconds=replay_seconds,postburst_modified_energy_relative_range=variation,
                direct_source='retained_window' if reference else 'fresh_wave_solve',
                synthesis=stats))
            direct.append(waveform.displacement_m);replayed.append(replay);inputs.append(forcing)
        atomic_write(output/'green-check-arrays.npz',lambda f:np.savez_compressed(f,time_s=time_s,
            green_displacement_m_per_n=green.displacement_m,green_force_n=green_force,
            direct_displacement_m=np.asarray(direct),replayed_displacement_m=np.asarray(replayed),force_n=np.asarray(inputs),
            green_final_displacement_m=green.final_displacement_m,green_final_velocity_m_s=green.final_velocity_m_s,
            mass_diagonal_kg=cpu.mass_kg,green_final_gpu_force_n=gpu_force,green_final_cpu_force_n=cpu_force))
        if (any(plan[k]!=v for k,v in identities().items())
                or any(sha256_file(PROJECT/name)!=digest for name,digest in analysis.items())
                or plan['runtime']!=distribution_identity(guard) or plan['gpu_packages']!=gpu_package_identity(guard)
                or sha256_file(output/'source-snapshot.zip')!=plan['source_snapshot_sha256']):
            raise ValueError('Green verification implementation/runtime changed')
        if reference and (any(sha256_file(Path(binding['directory'])/name)!=digest
                for name,digest in binding['input_identities'].items()) or sha256_file(binding['cpu_path'])!=binding['cpu_sha256']):
            raise ValueError('Green retained reference changed during verification')
        passed=all(row['passed'] for row in rows) and max(force_error,mass_error,probe_error,*energy_errors)<1e-10 and green_energy<1e-8
        report.update(completed=True,passed=bool(passed),trials=rows,green_propagation_seconds=green_seconds,
            cpu_force_relative_l2=force_error,mass_relative_l2=mass_error,probe_relative_l2=probe_error,
            energy_relative_errors=energy_errors,green_modified_energy_relative_range=green_energy,
            raw_arrays_sha256=sha256_file(output/'green-check-arrays.npz'),cooling_events=cooling.events,
            independent_force_inputs=1+len(frequencies),fresh_forward_solves=1 if reference else 1+len(frequencies),
            retained_reference=binding,training_updates=0)
    except Exception as error:report['error']=f'{type(error).__name__}: {error}';raise
    finally:
        report['wall_seconds']=time.perf_counter()-started;report['resources']=guard.report()
        write_json(output/'green-check.json',sealed(report))
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--reference-run',type=Path);parser.add_argument('--cpu-check',type=Path)
    args=parser.parse_args();guard=ResourceGuard(ResourceLimits(max_seconds=3600 if args.reference_run else 180,
        max_rss_mib=4096 if args.reference_run else 1536,
        max_system_memory_percent=89.9,min_available_mib=768))
    report=verify(args.output,guard,args.reference_run,args.cpu_check)
    print(json.dumps({k:v for k,v in report.items() if k not in ('resources','cooling_events')}))
    return 0 if report['passed'] else 2


if __name__=='__main__':raise SystemExit(main())
