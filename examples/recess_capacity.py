"""Measured full-size compact recess geometry/operator/port capacity."""
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

from examples import recess_geometry_reference as geometry_reference
from examples.tensor_wave import PROJECT, checked_receipt, identities, read_json, sealed, snapshot_sources
from traxis.compact_recess import make_compact_recess_mesh, storage_screen
from traxis.config import CircularRecess, Material, Plate
from traxis.embedded_recess import make_embedded_recess_mesh, storage_screen as embedded_screen
from traxis.elmer_reference import run_monitored
from traxis.gpu_mapped_tensor import build_gpu_mapped
from traxis.gpu_verification import gpu_package_identity
from traxis.highorder_verification import distribution_identity
from traxis.mapped_patches import circular_patch
from traxis.mapped_tensor import action, build_system
from traxis.resources import ResourceGuard, ResourceLimits
from traxis.storage import sha256_file, write_json


def fixed_settings(mesh_family='compact'):
    if mesh_family not in ('compact', 'embedded'):
        raise ValueError('unknown capacity mesh family')
    settings = dict(plate=asdict(Plate(.5, .5, .003175)), recess=asdict(CircularRecess(.255, .245, .015, .0015)),
        material=asdict(Material()), geometry_order=3, layers=[1, 1], mesh_family=mesh_family,
        max_mesh_bytes=512*1024**2, max_operator_bytes=1536*1024**2, max_batch_vectors=1,
        max_device_mib=3072, max_host_mib=2048, operator_tolerance=1e-10, mass_tolerance=1e-10,
        moment_tolerance=1e-9, source_xy_m=[.02, .02], receiver_xy_m=[.48, .48], patch_radius_m=.003,
        forecast_duration_s=.0005, forecast_stability_fraction=.8)
    settings.update(dict(angular_cells=64, radial_cells=[2, 32]) if mesh_family == 'compact'
                    else dict(angular_cells=12, radial_cells=[2, 6], outer_cell_size_m=.005, patch_bounds_m=None))
    return settings


def make_mesh(plan, guard):
    controls = dict(geometry_order=plan['geometry_order'], angular_cells=plan['angular_cells'],
        radial_cells=tuple(plan['radial_cells']), layers=tuple(plan['layers']), defective=plan['defective'],
        max_bytes=plan['max_mesh_bytes'], guard=guard)
    plate, recess = Plate(**plan['plate']), CircularRecess(**plan['recess'])
    if plan['mesh_family'] == 'embedded':
        return make_embedded_recess_mesh(plate, recess, plan['order'], **controls,
            outer_cell_size_m=plan['outer_cell_size_m'], patch_bounds_m=plan['patch_bounds_m'])
    return make_compact_recess_mesh(plate, recess, plan['order'], **controls)


def mesh_screen(plan):
    if plan['mesh_family'] == 'embedded':
        return embedded_screen(Plate(**plan['plate']), CircularRecess(**plan['recess']), plan['order'],
            geometry_order=plan['geometry_order'], angular_cells=plan['angular_cells'], radial_cells=tuple(plan['radial_cells']),
            layers=tuple(plan['layers']), defective=plan['defective'], max_bytes=plan['max_mesh_bytes'],
            outer_cell_size_m=plan['outer_cell_size_m'], patch_bounds_m=plan['patch_bounds_m'])
    return storage_screen(plan['order'], plan['geometry_order'], plan['angular_cells'], tuple(plan['radial_cells']),
                          tuple(plan['layers']), plan['defective'], max_bytes=plan['max_mesh_bytes'])


def source_check(plan, guard):
    if (plan['kind'] != 'full_size_mapped_capacity' or any(plan.get(k) != v for k, v in fixed_settings(plan['mesh_family']).items())
            or type(plan['order']) is not int or plan['order'] not in (3, 4, 5) or type(plan['defective']) is not bool
            or plan['mesh_screen'] != mesh_screen(plan)
            or any(plan[k] != v for k, v in identities().items())
            or any(sha256_file(PROJECT/name) != digest for name, digest in plan['analysis_sources'].items())
            or plan['runtime'] != distribution_identity(guard) or plan['gpu_packages'] != gpu_package_identity(guard)):
        raise ValueError('full-size capacity source/runtime/settings changed')


def worker(plan_path, output, guard):
    plan = checked_receipt(read_json(plan_path)); output.mkdir()
    report = dict(completed=False, passed=False, ready_for_training_data=False, training_updates=0,
                  plan_fingerprint=plan['fingerprint'], full_window_completed=False, time_steps_completed=0)
    started = time.perf_counter(); phases = {}
    def phase(name):
        guard.check(force=True); phases[name] = time.perf_counter()-started
        print(json.dumps(dict(phase=name, elapsed_seconds=phases[name])), flush=True)
    try:
        source_check(plan, guard)
        if psutil.disk_usage(str(output)).free < 2048*1024**2:
            raise MemoryError('full-size proof needs at least 2 GiB free disk')
        plate = Plate(**plan['plate']); recess = CircularRecess(**plan['recess']); material = Material(**plan['material'])
        mesh = make_mesh(plan, guard)
        phase('mesh_built')
        cpu = build_system(mesh, material, guard, max_bytes=plan['max_operator_bytes'], max_batch_vectors=1)
        phase('cpu_geometry_mass_and_stability_built')
        gpu = build_gpu_mapped(cpu, guard=guard, max_device_mib=plan['max_device_mib'], max_host_mib=plan['max_host_mib'])
        phase('gpu_operator_built')
        ns = len(mesh.points_m)
        x, y, z = (mesh.points_m[:, i]/length for i, length in enumerate((.5, .5, .003175)))
        state = np.array([np.sin(2*np.pi*x)+.17*np.cos(3*np.pi*z), .5*np.cos(np.pi*y)*np.sin(np.pi*z),
                          .25*np.sin(np.pi*(x+y))*np.cos(np.pi*z)]).ravel()
        del x, y, z
        before = time.perf_counter(); measured = gpu.stiffness@state; gpu_host_seconds = time.perf_counter()-before
        phase('gpu_field_applied')
        before = time.perf_counter(); expected = action(cpu, state, guard); cpu_seconds = time.perf_counter()-before
        operator_error = float(np.linalg.norm(measured-expected)/np.linalg.norm(expected))
        phase('cpu_field_compared')
        storage = gpu.stiffness.storage; cp = storage.cp; events = []; walls = []
        with storage.lock, cp.cuda.Device(storage.device_id):
            for _ in range(5):
                guard.check()
                begin, end = cp.cuda.Event(), cp.cuda.Event(); before = time.perf_counter()
                begin.record(); storage._launch(storage.input, storage.output, 1., 0.); end.record(); end.synchronize()
                events.append(float(cp.cuda.get_elapsed_time(begin, end)/1000)); walls.append(time.perf_counter()-before)
        mass = cpu.mass_kg[:ns]; circle_area = geometry_reference.circle_reference(recess.radius_m, 3, plan['angular_cells'])[0]
        full_volume = plate.width_m*plate.height_m*plate.thickness_m
        removed = circle_area*recess.depth_m if plan['defective'] else 0.
        expected_mass = material.density_kg_m3*(full_volume-removed)
        mass_error = abs(float(mass.sum())/expected_mass-1)
        expected_first = (full_volume*np.array([.25, .25, .003175/2])
                          -removed*np.array([recess.x_m, recess.y_m, recess.depth_m/2]))/(full_volume-removed)
        first_moment = mass@mesh.points_m/mass.sum()
        moment_error = float(np.max(abs((first_moment-expected_first)/[.5, .5, .003175])))
        shortest, longest = math.inf, 0.
        edges = ((0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7))
        for first in range(0, len(mesh.corner_elements), 512):
            guard.check(); corners = mesh.corner_points_m[mesh.corner_elements[first:first+512]]
            lengths = np.array([np.linalg.norm(corners[:, a]-corners[:, b], axis=1) for a, b in edges])
            shortest, longest = min(shortest, float(lengths.min())), max(longest, float(lengths.max()))
        phase('mass_moments_edges_and_gpu_timing_checked')
        before = time.perf_counter(); source = circular_patch(mesh, plan['source_xy_m'], plan['patch_radius_m'], guard=guard)
        source_seconds = time.perf_counter()-before; phase('physical_source_integrated')
        before = time.perf_counter(); receiver = circular_patch(mesh, plan['receiver_xy_m'], plan['patch_radius_m'], guard=guard)
        receiver_seconds = time.perf_counter()-before; phase('physical_receiver_integrated')
        step = plan['forecast_stability_fraction']*cpu.stable_dt_s
        steps = math.ceil(plan['forecast_duration_s']/step)
        np.savez_compressed(output/'capacity-proofs.npz', state=state, gpu_action=measured, cpu_action=expected,
            scalar_mass=mass, points_m=mesh.points_m, source_indices=source.indices, source_weights=source.weights,
            receiver_indices=receiver.indices, receiver_weights=receiver.weights)
        phase('proof_arrays_saved'); source_check(plan, guard)
        report.update(completed=True, passed=bool(operator_error < 1e-10 and mass_error < 1e-10 and moment_error < 1e-9),
            mesh_family=plan['mesh_family'],
            system=gpu.screen, device=gpu.device, scalar_nodes=ns, elements=len(mesh.element_nodes),
            operator_relative_l2=operator_error, gpu_host_application_seconds=gpu_host_seconds, cpu_application_seconds=cpu_seconds,
            gpu_event_seconds=events, gpu_wall_seconds=walls, mass_kg=float(mass.sum()), expected_mass_kg=expected_mass,
            mass_relative_error=mass_error, first_moment_m=first_moment.tolist(), expected_first_moment_m=expected_first.tolist(),
            normalized_first_moment_error=moment_error, shortest_corner_edge_m=shortest, longest_corner_edge_m=longest,
            source=source.report, receiver=receiver.report, source_seconds=source_seconds, receiver_seconds=receiver_seconds,
            phase_elapsed_seconds=phases, proof_sha256=sha256_file(output/'capacity-proofs.npz'),
            forecast=dict(duration_s=plan['forecast_duration_s'], step_from_stability_s=step, steps=steps,
                operator_only_device_seconds=steps*float(np.median(events)),
                scope='Operator-only extrapolation; excludes integrator/probe/guard/cooling overhead and does not choose an accuracy-qualified time step.'),
        scope='Measured full-size mapped geometry/operator/physical-port capacity; no propagation, spatial/frequency convergence, calibrated PZT/contact or accepted waveform data.')
    except Exception as error:
        report['error'] = f'{type(error).__name__}: {error}'; raise
    finally:
        report['resources'] = guard.report(); write_json(output/'capacity-report.json', sealed(report))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--order', type=int, choices=(3, 4, 5), default=4)
    parser.add_argument('--mesh', choices=('compact', 'embedded'), default='compact')
    parser.add_argument('--intact', action='store_true'); parser.add_argument('--worker', action='store_true'); parser.add_argument('--plan', type=Path)
    args = parser.parse_args(); output = args.output.resolve()
    guard = ResourceGuard(ResourceLimits(max_seconds=360, max_rss_mib=1536, max_system_memory_percent=85, min_available_mib=2048))
    if args.worker: return 0 if worker(args.plan, output, guard)['passed'] else 2
    if not output.is_relative_to(PROJECT/'outputs'): raise ValueError('full-size capacity evidence must remain in project outputs')
    output.mkdir(parents=True, exist_ok=False)
    analysis = {str(Path(name).resolve().relative_to(PROJECT)).replace('\\', '/'):sha256_file(name)
                for name in (__file__, geometry_reference.__file__)}
    plan = dict(kind='full_size_mapped_capacity', order=args.order, defective=not args.intact, **fixed_settings(args.mesh),
        **identities(), analysis_sources=analysis, runtime=distribution_identity(guard), gpu_packages=gpu_package_identity(guard))
    plan['mesh_screen'] = mesh_screen(plan)
    snapshot_sources(output/'source-snapshot.zip', plan)
    with zipfile.ZipFile(output/'source-snapshot.zip', 'a', compression=zipfile.ZIP_DEFLATED) as archive:
        for name in analysis: archive.writestr(name, (PROJECT/name).read_bytes())
    plan['source_snapshot_sha256'] = sha256_file(output/'source-snapshot.zip'); write_json(output/'capacity-plan.json', sealed(plan))
    temporary = output/'temporary'; temporary.mkdir()
    env = {**os.environ, 'OMP_NUM_THREADS':'1', 'OPENBLAS_NUM_THREADS':'1', 'CUPY_CACHE_DIR':str(output/'kernel-cache'),
        'CUPY_CACHE_SAVE_CUDA_SOURCE':'1', **{name:str(temporary) for name in ('TEMP', 'TMP', 'TMPDIR')}}
    result = dict(completed=False, passed=False, ready_for_training_data=False, training_updates=0, plan_fingerprint=plan['fingerprint'])
    try:
        command = [sys.executable, '-m', 'examples.recess_capacity', '--worker', '--plan', str(output/'capacity-plan.json'), '--output', str(output/'native')]
        result['worker_run'] = run_monitored(command, PROJECT, env, guard, output/'worker.log')
        if result['worker_run']['exit_code'] not in (0, 2): raise RuntimeError('full-size capacity worker failed; inspect worker.log')
        report = checked_receipt(read_json(output/'native/capacity-report.json'))
        if (not report['completed'] or report['plan_fingerprint'] != plan['fingerprint']
                or report['proof_sha256'] != sha256_file(output/'native/capacity-proofs.npz')):
            raise ValueError('full-size capacity evidence incomplete or changed')
        source_check(plan, guard)
        if sha256_file(output/'source-snapshot.zip') != plan['source_snapshot_sha256']: raise ValueError('capacity snapshot changed')
        result.update(completed=True, passed=bool(report['passed'] and result['worker_run']['exit_code'] == 0), report_sha256=sha256_file(output/'native/capacity-report.json'))
    except Exception as error: result['error'] = f'{type(error).__name__}: {error}'
    finally:
        result['resources'] = guard.report(); write_json(output/'monitor-result.json', sealed(result))
    print(json.dumps({k:v for k,v in result.items() if k != 'resources'})); return 0 if result['passed'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
