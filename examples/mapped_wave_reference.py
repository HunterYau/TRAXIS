"""Complete bounded mapped-mesh pulse windows with physical/remounted disks."""
import argparse
from dataclasses import asdict, replace
import json
import math
import os
from pathlib import Path
import sys
import zipfile

import numpy as np

from examples import mapped_patch_reference as patch_reference
from examples import recess_geometry_reference as geometry_reference
from examples.tensor_wave import PROJECT, checked_receipt, identities, read_json, sealed, snapshot_sources
from traxis.config import CircularRecess, Material, Plate
from traxis.elmer_reference import run_monitored
from traxis.fem import tone_burst
from traxis.gpu_enriched_verlet import gpu_explicit_simulate
from traxis.gpu_mapped_tensor import build_gpu_mapped
from traxis.gpu_verification import gpu_package_identity
from traxis.highorder_verification import distribution_identity
from traxis.mapped_patches import scenario_patches
from traxis.mapped_tensor import action, build_system
from traxis.mounting import PZTMounting
from traxis.recess_hex import make_recess_mesh
from traxis.resources import ResourceGuard, ResourceLimits
from traxis.scenarios import AcquisitionSettings, NuisanceSettings, Scenario
from traxis.storage import sha256_file, write_json


def reference_scenario():
    return Scenario('mapped-pulse-debug', 'mapped-pulse-parent', 'train', 2026, 44,
        Plate(.04, .03, .003175), (CircularRecess(.021, .014, .008, .0015),),
        AcquisitionSettings(inset_m=.006, patch_radius_m=.003, frequencies_hz=(100000.,), samples=64, cycles=2),
        NuisanceSettings(Material(), 1., (1.,)*8, (1.,)*8, relative_noise_std=0.))


def worker(plan_path, output, guard):
    plan = checked_receipt(read_json(plan_path)); output.mkdir()
    report = dict(completed=False, passed=False, ready_for_training_data=False, training_updates=0,
                  plan_fingerprint=plan['fingerprint'])
    def source_check():
        if (any(plan[k] != v for k, v in identities().items())
                or any(sha256_file(PROJECT/name) != digest for name, digest in plan['analysis_sources'].items())
                or plan['runtime'] != distribution_identity(guard) or plan['gpu_packages'] != gpu_package_identity(guard)):
            raise ValueError('mapped pulse source/runtime changed')
    try:
        source_check(); scenario = reference_scenario()
        if (plan['kind'] != 'mapped_mechanical_pulse_reference' or Scenario.from_dict(plan['scenario']) != scenario
                or plan['order'] != 4 or plan['angular_cells'] != 4
                or plan['time_tolerance'] != 1e-3 or plan['reciprocity_tolerance'] != 1e-8
                or plan['operator_tolerance'] != 1e-10 or plan['energy_tolerance'] != 1e-8):
            raise ValueError('mapped pulse settings or acceptance criteria changed')
        systems = {}; ports = {}; native_ports = []; arrays = {}
        remounted = replace(scenario, specimen_id='mapped-remounted-intact', recesses=(),
            current_mounting=PZTMounting(((.0004, -.0003),)+((0., 0.),)*7, (1.1,)+(1.,)*7))
        for defective in (False, True):
            mesh = make_recess_mesh(scenario.plate, scenario.recesses[0], 4, angular_cells=4, defective=defective)
            cpu = build_system(mesh, scenario.nuisance.material, guard)
            systems[defective] = build_gpu_mapped(cpu, guard=guard)
            state = 'current' if defective else 'intact'
            ports[defective] = scenario_patches(mesh, scenario, state, guard=guard)
            geometry, _, model = geometry_reference.native_geometry(mesh, guard, return_model=True)
            chosen = [('nominal', ports[defective])]
            if not defective:
                ports['remounted'] = scenario_patches(mesh, remounted, 'current', guard=guard)
                chosen.append(('remounted', ports['remounted']))
            for mounting, port_list in chosen:
                for sensor in (0, 4):
                    native, dual, detail = patch_reference.native_covector(mesh, model, port_list[sensor], guard)
                    name = f'{"recess" if defective else "intact"}_{mounting}_sensor{sensor}'
                    native_ports.append(dict(name=name, native=detail, patch=port_list[sensor].report))
                    arrays[name+'_native_load'] = native; arrays[name+'_mapped_dual_load'] = dual
        duration = (scenario.acquisition.samples-1)*scenario.acquisition.sample_dt_s
        minimum_step = min(system.stable_dt_s for system in systems.values())
        substeps = 2**math.ceil(math.log2(scenario.acquisition.sample_dt_s/(.8*minimum_step)))
        if not 64 <= substeps <= 512:
            raise MemoryError('mapped pulse exceeds the bounded internal sample refinement')
        steps = (scenario.acquisition.samples-1)*substeps
        records = []; solutions = {}
        cases = [('nominal', False, False, 0, 4, 1), ('reciprocal', False, False, 4, 0, 1),
                 ('time-fine', False, False, 0, 4, 2), ('remounted', False, True, 0, 4, 1),
                 ('recess', True, False, 0, 4, 1)]
        for name, defective, moved, tx, rx, refinement in cases:
            system = systems[defective]; port_list = ports['remounted' if moved else defective]
            nscalar = len(system.mass_diagonal)//3
            load = port_list[tx].vector(nscalar); probes = port_list[rx].vector(nscalar)[None]
            result = gpu_explicit_simulate(system, steps*refinement, duration/(steps*refinement), probes, load,
                force=lambda t: tone_burst(t, 100000., 2), output_stride=64, energy_stride=64, guard=guard)
            after = result.energy_time_s >= 2/100000.
            energy = result.modified_energy_j[after]
            energy_range = float(np.ptp(energy)/abs(energy.mean()))
            cpu_force = action(system.cpu_system, result.final_displacement_m, guard)
            gpu_force = system.stiffness@result.final_displacement_m
            force_error = float(np.linalg.norm(gpu_force-cpu_force)/np.linalg.norm(cpu_force))
            complete = result.statistics['steps_completed'] == steps*refinement and abs(result.time_s[-1]-duration) < 1e-18
            records.append(dict(name=name, defective=defective, remounted=moved, transmitter=tx, receiver=rx,
                steps=steps*refinement, dt_s=duration/(steps*refinement), duration_s=duration, completed=bool(complete),
                post_burst_modified_energy_relative_range=energy_range, final_cpu_gpu_force_relative_l2=force_error,
                system=system.screen, statistics=result.statistics,
                passed=bool(complete and energy_range < 1e-8 and force_error < 1e-10)))
            solutions[name] = result
            for key, value in dict(time_s=result.time_s, trace=result.displacement_m,
                final_u=result.final_displacement_m, final_v=result.final_velocity_m_s,
                energy_time_s=result.energy_time_s, modified_energy=result.modified_energy_j,
                cpu_final_force=cpu_force, gpu_final_force=gpu_force, load=load, probe=probes).items():
                arrays[name+'_'+key] = value
            write_json(output/'progress.json', dict(completed=False, windows_completed=len(records), windows_planned=len(cases)), overwrite=True)
        baseline = solutions['nominal'].displacement_m
        def relative(values):
            return float(np.linalg.norm(values-baseline)/np.linalg.norm(baseline))
        errors = dict(reciprocity=relative(solutions['reciprocal'].displacement_m),
            time_refinement=relative(solutions['time-fine'].displacement_m[::2]),
            remount_change=relative(solutions['remounted'].displacement_m),
            recess_change=relative(solutions['recess'].displacement_m))
        np.savez_compressed(output/'pulse-proofs.npz', **arrays)
        source_check()
        report.update(completed=True, passed=bool(all(c['passed'] for c in records)
            and all(c['native']['passed'] for c in native_ports) and errors['reciprocity'] < 1e-8
            and errors['time_refinement'] < 1e-3 and min(errors['remount_change'], errors['recess_change']) > 1e-6),
            scenario=scenario.to_dict(), remounted_scenario=remounted.to_dict(), native_ports=native_ports,
            windows=records, comparisons=errors, proof_sha256=sha256_file(output/'pulse-proofs.npz'),
            scope='Complete small 31.5 us mapped mechanical pulse windows with remounted disks; no spatial/full-size/clamp/PZT-voltage validation or accepted training data.')
    except Exception as error:
        report['error'] = f'{type(error).__name__}: {error}'; raise
    finally:
        report['resources'] = guard.report(); write_json(output/'pulse-report.json', sealed(report))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--worker', action='store_true'); parser.add_argument('--plan', type=Path)
    args = parser.parse_args(); output = args.output.resolve()
    guard = ResourceGuard(ResourceLimits(max_seconds=240, max_rss_mib=1024, max_system_memory_percent=85, min_available_mib=2048))
    if args.worker: return 0 if worker(args.plan, output, guard)['passed'] else 2
    if not output.is_relative_to(PROJECT/'outputs'): raise ValueError('mapped pulse evidence must remain in project outputs')
    output.mkdir(parents=True, exist_ok=False)
    analysis = {str(Path(name).resolve().relative_to(PROJECT)).replace('\\', '/'):sha256_file(name)
                for name in (__file__, patch_reference.__file__, geometry_reference.__file__)}
    plan = dict(kind='mapped_mechanical_pulse_reference', scenario=reference_scenario().to_dict(), order=4, angular_cells=4,
        time_tolerance=1e-3, reciprocity_tolerance=1e-8, operator_tolerance=1e-10, energy_tolerance=1e-8,
        **identities(), analysis_sources=analysis, runtime=distribution_identity(guard), gpu_packages=gpu_package_identity(guard))
    snapshot_sources(output/'source-snapshot.zip', plan)
    with zipfile.ZipFile(output/'source-snapshot.zip', 'a', compression=zipfile.ZIP_DEFLATED) as archive:
        for name in analysis: archive.writestr(name, (PROJECT/name).read_bytes())
    plan['source_snapshot_sha256'] = sha256_file(output/'source-snapshot.zip'); write_json(output/'pulse-plan.json', sealed(plan))
    temporary = output/'temporary'; temporary.mkdir()
    env = {**os.environ, 'OMP_NUM_THREADS':'1', 'OPENBLAS_NUM_THREADS':'1', 'CUPY_CACHE_DIR':str(output/'kernel-cache'),
        'CUPY_CACHE_SAVE_CUDA_SOURCE':'1', **{name:str(temporary) for name in ('TEMP', 'TMP', 'TMPDIR')}}
    result = dict(completed=False, passed=False, ready_for_training_data=False, training_updates=0, plan_fingerprint=plan['fingerprint'])
    try:
        command = [sys.executable, '-m', 'examples.mapped_wave_reference', '--worker', '--plan', str(output/'pulse-plan.json'), '--output', str(output/'native')]
        result['worker_run'] = run_monitored(command, PROJECT, env, guard, output/'worker.log')
        if result['worker_run']['exit_code'] not in (0, 2): raise RuntimeError('mapped pulse worker failed; inspect worker.log')
        report = checked_receipt(read_json(output/'native/pulse-report.json'))
        if (not report['completed'] or report['plan_fingerprint'] != plan['fingerprint']
                or report['proof_sha256'] != sha256_file(output/'native/pulse-proofs.npz')):
            raise ValueError('mapped pulse evidence incomplete or changed')
        result.update(completed=True, passed=bool(report['passed'] and result['worker_run']['exit_code'] == 0),
            report_sha256=sha256_file(output/'native/pulse-report.json'))
    except Exception as error: result['error'] = f'{type(error).__name__}: {error}'
    finally:
        result['resources'] = guard.report(); write_json(output/'monitor-result.json', sealed(result))
    print(json.dumps({k:v for k,v in result.items() if k != 'resources'})); return 0 if result['passed'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
