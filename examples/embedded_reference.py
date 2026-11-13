"""Independent native/GPU elasticity and surface-port checks on embedded grids."""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
import zipfile

import numpy as np

from examples import gpu_mapped_reference as gpu_reference
from examples import mapped_patch_reference as patch_reference
from examples import recess_geometry_reference as geometry_reference
from examples import recess_operator_reference as operator_reference
from examples.tensor_wave import PROJECT, checked_receipt, identities, read_json, sealed, snapshot_sources
from traxis.config import CircularRecess, Material, Plate
from traxis.elmer_reference import run_monitored
from traxis.embedded_recess import make_embedded_recess_mesh
from traxis.gpu_mapped_tensor import build_gpu_mapped
from traxis.gpu_verification import gpu_package_identity
from traxis.highorder_verification import distribution_identity
from traxis.mapped_patches import circular_patch
from traxis.mapped_tensor import build_system
from traxis.resources import ResourceGuard, ResourceLimits
from traxis.storage import sha256_file, write_json


def settings():
    return dict(plate=asdict(Plate(.08, .06, .003175)), recess=asdict(CircularRecess(.041, .029, .008, .0015)),
        material=asdict(Material()), geometry_order=3, angular_cells=2, radial_cells=[1, 1], layers=[1, 1],
        outer_cell_size_m=.04, operator_tolerance=1e-10, port_tolerance=1e-9,
        cases=[['corner', [.006, .006], .003], ['remounted-corner', [.0064, .0057], .0033],
               ['receiver', [.074, .054], .003], ['patch-corner', [.025, .013], .003]])


def source_check(plan, guard):
    if (plan['kind'] != 'embedded_native_reference' or any(plan.get(k) != v for k, v in settings().items())
            or type(plan['order']) is not int or plan['order'] not in (3, 4, 5) or type(plan['defective']) is not bool
            or any(plan[k] != v for k, v in identities().items())
            or any(sha256_file(PROJECT/name) != digest for name, digest in plan['analysis_sources'].items())
            or plan['runtime'] != distribution_identity(guard) or plan['gpu_packages'] != gpu_package_identity(guard)):
        raise ValueError('embedded native source/runtime/settings changed')


def worker(plan_path, output, guard):
    plan = checked_receipt(read_json(plan_path)); output.mkdir()
    report = dict(completed=False, passed=False, ready_for_training_data=False, training_updates=0,
                  plan_fingerprint=plan['fingerprint'])
    try:
        source_check(plan, guard)
        mesh = make_embedded_recess_mesh(Plate(**plan['plate']), CircularRecess(**plan['recess']), plan['order'],
            geometry_order=3, angular_cells=2, radial_cells=(1, 1), outer_cell_size_m=.04, defective=plan['defective'], guard=guard)
        cpu = build_system(mesh, Material(**plan['material']), guard)
        geometry, _, model = geometry_reference.native_geometry(mesh, guard, return_model=True)
        native = operator_reference.compare_native(cpu, model, geometry, output, guard)
        gpu = build_gpu_mapped(cpu, guard=guard); cuda = gpu_reference.compare_gpu(cpu, gpu, native, output, guard)
        ports, arrays = [], {}
        for i, (name, center, radius) in enumerate(plan['cases']):
            port = circular_patch(mesh, center, radius, guard=guard)
            coefficients, dual, detail = patch_reference.native_covector(mesh, model, port, guard)
            ports.append(dict(name=name, patch=port.report, native=detail))
            arrays[f'case{i}_native'] = coefficients; arrays[f'case{i}_mapped_dual'] = dual
            arrays[f'case{i}_nodal_indices'] = port.indices; arrays[f'case{i}_nodal_weights'] = port.weights
            arrays[f'case{i}_physical_support_m'] = mesh.points_m[port.indices]
        np.savez_compressed(output/'embedded-port-proofs.npz', **arrays)
        source_check(plan, guard)
        report.update(completed=True, passed=bool(native['passed'] and cuda['passed'] and all(p['native']['passed'] for p in ports)),
            geometry=geometry, system=gpu.screen, device=gpu.device, native=native, gpu=cuda, ports=ports,
            patch_bounds_m=list(mesh.patch_bounds_m), exterior_cells=list(mesh.exterior_cells),
            port_proof_sha256=sha256_file(output/'embedded-port-proofs.npz'),
            scope='Bounded embedded-grid native geometry/elasticity and complete mechanical covectors; no full-size wave or physical clamp acceptance.')
    except Exception as error:
        report['error'] = f'{type(error).__name__}: {error}'; raise
    finally:
        report['resources'] = guard.report(); write_json(output/'embedded-report.json', sealed(report))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--order', type=int, choices=(3, 4, 5), default=3); parser.add_argument('--intact', action='store_true')
    parser.add_argument('--worker', action='store_true'); parser.add_argument('--plan', type=Path)
    args = parser.parse_args(); output = args.output.resolve()
    guard = ResourceGuard(ResourceLimits(max_seconds=300, max_rss_mib=1024, max_system_memory_percent=85, min_available_mib=2048))
    if args.worker: return 0 if worker(args.plan, output, guard)['passed'] else 2
    if not output.is_relative_to(PROJECT/'outputs'): raise ValueError('embedded evidence must remain in project outputs')
    output.mkdir(parents=True, exist_ok=False)
    drivers = (__file__, geometry_reference.__file__, operator_reference.__file__, gpu_reference.__file__, patch_reference.__file__)
    analysis = {str(Path(name).resolve().relative_to(PROJECT)).replace('\\', '/'):sha256_file(name) for name in drivers}
    plan = dict(kind='embedded_native_reference', order=args.order, defective=not args.intact, **settings(),
        **identities(), analysis_sources=analysis, runtime=distribution_identity(guard), gpu_packages=gpu_package_identity(guard))
    snapshot_sources(output/'source-snapshot.zip', plan)
    with zipfile.ZipFile(output/'source-snapshot.zip', 'a', compression=zipfile.ZIP_DEFLATED) as archive:
        for name in analysis: archive.writestr(name, (PROJECT/name).read_bytes())
    plan['source_snapshot_sha256'] = sha256_file(output/'source-snapshot.zip'); write_json(output/'embedded-plan.json', sealed(plan))
    temporary = output/'temporary'; temporary.mkdir()
    env = {**os.environ, 'OMP_NUM_THREADS':'1', 'OPENBLAS_NUM_THREADS':'1', 'CUPY_CACHE_DIR':str(output/'kernel-cache'),
        'CUPY_CACHE_SAVE_CUDA_SOURCE':'1', **{name:str(temporary) for name in ('TEMP', 'TMP', 'TMPDIR')}}
    result = dict(completed=False, passed=False, ready_for_training_data=False, training_updates=0, plan_fingerprint=plan['fingerprint'])
    try:
        command = [sys.executable, '-m', 'examples.embedded_reference', '--worker', '--plan', str(output/'embedded-plan.json'), '--output', str(output/'native')]
        result['worker_run'] = run_monitored(command, PROJECT, env, guard, output/'worker.log')
        if result['worker_run']['exit_code'] not in (0, 2): raise RuntimeError('embedded native worker failed; inspect worker.log')
        report = checked_receipt(read_json(output/'native/embedded-report.json'))
        if not report['completed'] or report['plan_fingerprint'] != plan['fingerprint']:
            raise ValueError('embedded evidence incomplete or changed')
        paths = {item['path']:item['sha256'] for item in report['native']['element_files']}
        paths.update({'global-actions.npz':report['native']['global_actions_sha256'],
            'gpu-comparisons.npz':report['gpu']['proof_sha256'], 'embedded-port-proofs.npz':report['port_proof_sha256']})
        if any(sha256_file(output/'native'/name) != digest for name, digest in paths.items()):
            raise ValueError('embedded proof changed')
        source_check(plan, guard)
        if sha256_file(output/'source-snapshot.zip') != plan['source_snapshot_sha256']: raise ValueError('embedded snapshot changed')
        result.update(completed=True, passed=bool(report['passed'] and result['worker_run']['exit_code'] == 0),
                      report_sha256=sha256_file(output/'native/embedded-report.json'))
    except Exception as error: result['error'] = f'{type(error).__name__}: {error}'
    finally:
        result['resources'] = guard.report(); write_json(output/'monitor-result.json', sealed(result))
    print(json.dumps({k:v for k,v in result.items() if k != 'resources'})); return 0 if result['passed'] else 2


if __name__ == '__main__': raise SystemExit(main())
