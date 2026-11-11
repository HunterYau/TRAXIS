"""Select and verify circular geometry separately from wave resolution."""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
import zipfile

import numpy as np

from examples import recess_geometry_reference as reference
from examples.tensor_wave import PROJECT, checked_receipt, identities, read_json, sealed, snapshot_sources
from traxis.config import CircularRecess, Plate
from traxis.elmer_reference import run_monitored
from traxis.highorder_verification import distribution_identity
from traxis.recess_hex import make_recess_mesh
from traxis.resources import ResourceGuard, ResourceLimits
from traxis.storage import sha256_file, write_json


def worker(plan_path, output, guard):
    plan = checked_receipt(read_json(plan_path)); output.mkdir()
    def check_sources():
        if (any(plan[k] != v for k, v in identities().items())
                or any(sha256_file(PROJECT/name) != digest for name, digest in plan['analysis_sources'].items())
                or plan['runtime'] != distribution_identity(guard)):
            raise ValueError('circle shape source/runtime changed')
    report = dict(completed=False, passed=False, ready_for_training_data=False, training_updates=0,
                  plan_fingerprint=plan['fingerprint'])
    try:
        check_sources(); shape_rows, rows, native_rows, arrays = [], [], [], {}
        for cells in plan['angular_cells']:
            guard.check()
            area, radius = reference.circle_reference(1., plan['geometry_order'], cells)
            area_error = abs(area/np.pi-1)
            shape_rows.append(dict(angular_cells=cells, area_relative_error=area_error,
                radial_relative_error=radius, passed=bool(area_error < plan['area_relative_limit']
                    and radius < plan['radius_relative_limit'])))
        eligible = [r['angular_cells'] for r in shape_rows if r['passed']]
        if not eligible:
            raise ArithmeticError('no declared circular subdivision passes the geometry screen')
        selected = min(eligible)
        for case in plan['physical_cases']:
            mesh = make_recess_mesh(Plate(**case['plate']), CircularRecess(**case['recess']), 3,
                geometry_order=plan['geometry_order'], angular_cells=selected, defective=case['defective'])
            measured = reference.measure(mesh, guard)
            passed = bool(measured['passed'] and measured['circular_area_relative_error'] < plan['area_relative_limit']
                          and measured['circle_radial_relative_error'] < plan['radius_relative_limit'])
            rows.append({**case, **measured, 'passed': passed})
        for i, defective in enumerate((False, True)):
            mesh = make_recess_mesh(Plate(.04, .03, .003), CircularRecess(.021, .014, .008, .0015), 3,
                geometry_order=plan['geometry_order'], angular_cells=selected, defective=defective)
            measured = reference.measure(mesh, guard); native, proof = reference.native_geometry(mesh, guard)
            error = abs(native['native_volume_m3']/measured['volume_m3']-1)
            passed = bool(measured['passed'] and error < 1e-10
                and native['coordinate_maximum_absolute_error_m'] < 1e-12
                and native['jacobian_relative_l2'] < 1e-10 and native['minimum_native_jacobian'] > 0)
            native_rows.append({**measured, **native, 'defective': defective,
                                'native_volume_relative_error': error, 'passed': passed})
            for name, value in dict(**proof, geometry_points=mesh.geometry_points_m,
                    geometry_elements=mesh.geometry_elements, nodes=mesh.points_m,
                    element_nodes=mesh.element_nodes, cell_boxes=mesh.cell_boxes).items():
                arrays[f'case{i}_'+name] = value
        np.savez_compressed(output/'shape-proofs.npz', **arrays)
        check_sources()
        report.update(completed=True, passed=all(r['passed'] for r in rows+native_rows), shape_screen=shape_rows,
            selected_angular_cells=selected, physical_cases=rows, native_cases=native_rows,
            archive_sha256=sha256_file(output/'shape-proofs.npz'),
            scope='Circular shape and conforming/native geometry only; no spatial/time wave or PZT/data acceptance.')
    except Exception as error:
        report['error'] = f'{type(error).__name__}: {error}'; raise
    finally:
        report['resources'] = guard.report(); write_json(output/'shape-report.json', sealed(report))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--worker', action='store_true'); parser.add_argument('--plan', type=Path); args = parser.parse_args()
    output = args.output.resolve()
    guard = ResourceGuard(ResourceLimits(max_seconds=180, max_rss_mib=1024, max_system_memory_percent=85, min_available_mib=2048))
    if args.worker:
        return 0 if worker(args.plan, output, guard)['passed'] else 2
    if not output.is_relative_to(PROJECT/'outputs'):
        raise ValueError('shape study output must remain in the project')
    output.mkdir(parents=True, exist_ok=False)
    cases = [dict(plate=asdict(Plate()), recess=asdict(CircularRecess(.25, .25, diameter/2, depth)), defective=defective)
        for defective in (False, True) for diameter in (.02, .03, .04) for depth in (.00075, .0015, .002)]
    analysis = {str(Path(__file__).relative_to(PROJECT)).replace('\\','/'): sha256_file(__file__),
                'examples/recess_geometry_reference.py': sha256_file(reference.__file__)}
    plan = dict(kind='conforming_circle_shape_reference', geometry_order=3, angular_cells=[1, 2, 4, 8],
        area_relative_limit=1e-5, radius_relative_limit=1e-4, physical_cases=cases,
        **identities(), analysis_sources=analysis, runtime=distribution_identity(guard))
    snapshot_sources(output/'source-snapshot.zip', plan)
    with zipfile.ZipFile(output/'source-snapshot.zip', 'a', compression=zipfile.ZIP_DEFLATED) as archive:
        for name in analysis:
            archive.writestr(name, (PROJECT/name).read_bytes())
    plan['source_snapshot_sha256'] = sha256_file(output/'source-snapshot.zip')
    write_json(output/'shape-plan.json', sealed(plan))
    temporary = output/'temporary'; temporary.mkdir()
    env = {**os.environ, 'OMP_NUM_THREADS':'1', 'OPENBLAS_NUM_THREADS':'1',
           **{name:str(temporary) for name in ('TEMP', 'TMP', 'TMPDIR')}}
    result = dict(completed=False, passed=False, ready_for_training_data=False, training_updates=0,
                  plan_fingerprint=plan['fingerprint'])
    try:
        command = [sys.executable, '-m', 'examples.recess_shape_reference', '--worker',
                   '--plan', str(output/'shape-plan.json'), '--output', str(output/'native')]
        result['worker_run'] = run_monitored(command, PROJECT, env, guard, output/'worker.log')
        if result['worker_run']['exit_code'] not in (0, 2):
            raise RuntimeError('circle shape worker failed; inspect worker.log')
        report = checked_receipt(read_json(output/'native/shape-report.json'))
        if (not report['completed'] or report['plan_fingerprint'] != plan['fingerprint']
                or report['archive_sha256'] != sha256_file(output/'native/shape-proofs.npz')):
            raise ValueError('circle shape evidence is incomplete or changed')
        result.update(completed=True, passed=report['passed'], report_sha256=sha256_file(output/'native/shape-report.json'))
    except Exception as error:
        result['error'] = f'{type(error).__name__}: {error}'
    finally:
        result['resources'] = guard.report(); write_json(output/'monitor-result.json', sealed(result))
    print(json.dumps({k:v for k,v in result.items() if k != 'resources'}))
    return 0 if result['passed'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
