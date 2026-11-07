"""Measure a full-plate matrix-free space and operator actions, not waves."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import zipfile

import numpy as np

from traxis.cad_mesh import cad_mesh
from traxis.config import Material, Plate, positive
from traxis.elmer_plate_refinement import reference_source_hashes
from traxis.elmer_reference import run_monitored
from traxis.highorder_waveforms import HighOrderWaveSettings, waveform_runtime
from traxis.matrixfree import MatrixFreeSettings, build_matrix_free
from traxis.resources import ResourceGuard, ResourceLimits
from traxis.storage import atomic_write, canonical_hash, sha256_file, write_json


def checked_settings(values):
    if set(values) != {'plate','material','mesh_size_m','max_geometry_mib','max_tetrahedra',
                      'reference','seed','operator_repeats'}:
        raise ValueError('capacity settings have missing or unknown fields')
    plate, material = Plate(**values['plate']), Material(**values['material'])
    reference = MatrixFreeSettings(**values['reference'])
    positive('capacity mesh size',values['mesh_size_m'])
    if not math.isfinite(values['max_geometry_mib']) or not 0 < values['max_geometry_mib'] <= 2048:
        raise ValueError('invalid geometry storage screen')
    if (type(values['max_tetrahedra']) is not int or not 1 <= values['max_tetrahedra'] <= reference.max_tetrahedra
            or type(values['seed']) is not int or values['seed'] < 0
            or type(values['operator_repeats']) is not int or not 2 <= values['operator_repeats'] <= 5):
        raise ValueError('invalid capacity count/seed/repetition limit')
    return plate,material,reference


def worker(plan_path, output, guard):
    if plan_path.stat().st_size > 1024**2:
        raise MemoryError('capacity plan exceeds 1 MiB')
    plan = json.loads(plan_path.read_text())
    if (canonical_hash({k:v for k,v in plan.items() if k!='fingerprint'}) != plan['fingerprint']
            or plan['kind'] != 'matrixfree_fullplate_capacity' or reference_source_hashes() != plan['source_sha256']
            or plan['driver_sha256'] != sha256_file(Path(__file__))
            or waveform_runtime(HighOrderWaveSettings(mesh_backend='cad'),guard) != plan['runtime']):
        raise ValueError('capacity plan/source/runtime changed')
    plate,material,reference = checked_settings(plan['settings'])
    if not output.resolve().is_relative_to(plan_path.resolve().parent) or (output.exists() and any(output.iterdir())):
        raise ValueError('capacity output must be new and within its plan directory')
    output.mkdir(parents=True,exist_ok=True)
    report = {'completed':False,'ready_for_training_data':False,'wave_solves':0,'training_updates':0,
        'sparse_reference_assembled':False,'plan_fingerprint':plan['fingerprint'],'stages':[]}
    started = time.perf_counter()
    def stage(name, **fields):
        report['stages'].append({'name':name,'elapsed_seconds':time.perf_counter()-started,**fields})
        write_json(output/'progress.json',report,overwrite=True)
    try:
        mesh,geometry = cad_mesh(plate,(),plan['settings']['mesh_size_m'],
            max_geometry_mib=plan['settings']['max_geometry_mib'],max_tetrahedra=plan['settings']['max_tetrahedra'],
            algorithm=10,optimization='Netgen',seed=plan['settings']['seed'],guard=guard)
        atomic_write(output/'mesh.npz',lambda stream:np.savez_compressed(stream,points_m=mesh.points_m,tetrahedra=mesh.tetrahedra))
        if geometry['retained_volume_relative_error'] > 1e-10:
            raise ArithmeticError('full-plate intact mesh lost the declared box volume')
        stage('mesh',geometry=geometry,mesh_sha256=sha256_file(output/'mesh.npz'))
        system = build_matrix_free(mesh,material,reference,guard)
        stage('space_and_diagonals',screen=system.screen,
            native_operator_type=type(system.stiffness.form.mat).__name__)
        rng = np.random.default_rng(plan['settings']['seed'])
        first = first_k = first_m = None
        actions, symmetry_checks = [], []
        for i in range(plan['settings']['operator_repeats']):
            guard.check(force=True)
            values = rng.normal(size=system.space.ndof)*1e-9
            before = time.perf_counter()
            k_values = system.stiffness@values
            stiffness_seconds = time.perf_counter()-before
            before = time.perf_counter()
            m_values = system.mass@values
            mass_seconds = time.perf_counter()-before
            actions.append({'stiffness_seconds':stiffness_seconds,'mass_seconds':mass_seconds,
                'input_sha256':hashlib.sha256(values.tobytes()).hexdigest(),
                'stiffness_output_sha256':hashlib.sha256(k_values.tobytes()).hexdigest(),
                'mass_output_sha256':hashlib.sha256(m_values.tobytes()).hexdigest(),
                'stiffness_norm':float(np.linalg.norm(k_values)),'mass_norm':float(np.linalg.norm(m_values)),
                'stiffness_quadratic_form':float(values@k_values),'mass_quadratic_form':float(values@m_values)})
            if i == 0:
                first,first_k,first_m = values,k_values,m_values
            else:
                symmetry_k = abs(float(first@k_values-values@first_k))/max(float(np.linalg.norm(first)*np.linalg.norm(k_values)),1e-30)
                symmetry_m = abs(float(first@m_values-values@first_m))/max(float(np.linalg.norm(first)*np.linalg.norm(m_values)),1e-30)
                if max(symmetry_k,symmetry_m) > 1e-10:
                    raise ArithmeticError('full-plate matrix-free work reciprocity failed')
                symmetry_checks.append((symmetry_k,symmetry_m))
        if any(not math.isfinite(r[k]) or r[k] <= 0 for r in actions
               for k in ('mass_quadratic_form','stiffness_quadratic_form','stiffness_norm','mass_norm')):
            raise ArithmeticError('full-plate test fields have invalid energy quadratic forms')
        symmetry_k, symmetry_m = np.max(symmetry_checks,axis=0).tolist()
        stage('operator_actions',actions=actions,work_symmetry_k=symmetry_k,work_symmetry_m=symmetry_m)
        import ngsolve as ng
        field = ng.GridFunction(system.space)
        field.vec[:] = 0.
        field.components[0].Set(ng.CF(1.))
        rigid = field.vec.FV().NumPy().copy()
        mass_error = abs(float(rigid@(system.mass@rigid))/(material.density_kg_m3*geometry['tet_retained_volume_m3'])-1)
        rigid_force = float(np.linalg.norm(system.stiffness@rigid))
        relative_rigid_force = rigid_force/max(float(np.linalg.norm(first_k)/np.linalg.norm(first)*np.linalg.norm(rigid)),1e-30)
        if mass_error > 1e-10 or relative_rigid_force > 1e-10:
            raise ArithmeticError('full-plate translation mass or null-force check failed')
        if reference_source_hashes() != plan['source_sha256'] or waveform_runtime(HighOrderWaveSettings(mesh_backend='cad'),guard) != plan['runtime']:
            raise ValueError('capacity source/runtime changed during the measurement')
        guard.check(force=True)
        report.update({'completed':True,'dofs':system.space.ndof,'screen':system.screen,
            'translation_mass_relative_error':mass_error,'rigid_force_norm_n':rigid_force,
            'normalized_rigid_force':relative_rigid_force,'operator_actions':actions,
            'note':'Full physical plate geometry and vector elastic operators only. No propagation, arrival, convergence, PZT voltage or research-data acceptance.'})
    except Exception as exc:
        report['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        report['wall_seconds'] = time.perf_counter()-started
        report['resources'] = guard.report()
        report['fingerprint'] = canonical_hash(report)
        write_json(output/'capacity-report.json',report)
    return report


def measure_capacity(settings, output, guard=None):
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('capacity measurement needs a new output directory')
    checked_settings(settings)
    guard = guard or ResourceGuard(ResourceLimits(max_seconds=180,max_rss_mib=1024,
        max_system_memory_percent=85,min_available_mib=2048))
    output.mkdir(parents=True,exist_ok=True)
    result = {'completed':False,'wave_solves':0,'training_updates':0,'ready_for_training_data':False}
    try:
        plan = {'kind':'matrixfree_fullplate_capacity','settings':settings,'source_sha256':reference_source_hashes(),
            'runtime':waveform_runtime(HighOrderWaveSettings(mesh_backend='cad'),guard),
            'driver_sha256':sha256_file(Path(__file__)),'ready_for_training_data':False}
        plan['fingerprint'] = canonical_hash(plan)
        write_json(output/'capacity-plan.json',plan)
        def snapshot(stream):
            with zipfile.ZipFile(stream,'w',compression=zipfile.ZIP_DEFLATED) as archive:
                for name in plan['source_sha256']:
                    archive.writestr('traxis/'+name,(Path(__file__).resolve().parents[1]/'traxis'/name).read_bytes())
                archive.writestr('examples/matrixfree_capacity.py',Path(__file__).read_bytes())
        atomic_write(output/'source-snapshot.zip',snapshot)
        result['source_snapshot_sha256'] = sha256_file(output/'source-snapshot.zip')
        result['plan_fingerprint'] = plan['fingerprint']
        remaining = max(1.,guard.limits.max_seconds-(time.monotonic()-guard.started))
        command = [sys.executable,'-m','examples.matrixfree_capacity','--worker','--plan',str(output/'capacity-plan.json'),
            '--output',str(output/'native'),'--max-seconds',str(remaining),'--max-rss-mib',str(guard.limits.max_rss_mib)]
        result.update(run_monitored(command,Path(__file__).resolve().parents[1],
            {**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'},guard,output/'capacity.log'))
        if result['exit_code'] != 0:
            raise RuntimeError('capacity worker stopped; inspect retained native progress/report')
        native = json.loads((output/'native/capacity-report.json').read_text())
        if (canonical_hash({k:v for k,v in native.items() if k!='fingerprint'}) != native['fingerprint']
                or not native['completed'] or native['plan_fingerprint'] != plan['fingerprint']
                or reference_source_hashes() != plan['source_sha256']
                or sha256_file(Path(__file__)) != plan['driver_sha256']
                or waveform_runtime(HighOrderWaveSettings(mesh_backend='cad'),guard) != plan['runtime']
                or native['wave_solves'] != 0 or native['sparse_reference_assembled']
                or [s['name'] for s in native['stages']] != ['mesh','space_and_diagonals','operator_actions']
                or len(native['operator_actions']) != settings['operator_repeats']):
            raise ValueError('capacity worker/source receipt changed')
        mesh_sha256 = sha256_file(output/'native/mesh.npz')
        if mesh_sha256 != native['stages'][0]['mesh_sha256']:
            raise ValueError('capacity mesh archive changed')
        result.update({'completed':True,'dofs':native['dofs'],
            'mesh_sha256':mesh_sha256,
            'native_report_sha256':sha256_file(output/'native/capacity-report.json')})
    except Exception as exc:
        result['error'] = f'{type(exc).__name__}: {exc}'
    finally:
        result['resources'] = guard.report()
        result['fingerprint'] = canonical_hash(result)
        write_json(output/'monitor-result.json',result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--settings',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--plan',type=Path)
    parser.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    parser.add_argument('--max-seconds',type=float,default=180)
    parser.add_argument('--max-rss-mib',type=float,default=1024)
    args = parser.parse_args()
    guard = ResourceGuard(ResourceLimits(max_seconds=args.max_seconds,max_rss_mib=args.max_rss_mib,
        max_system_memory_percent=85,min_available_mib=2048))
    if args.worker:
        worker(args.plan,args.output,guard)
        return 0
    if not args.settings or args.settings.stat().st_size > 1024**2:
        parser.error('provide a bounded capacity settings file')
    settings = json.loads(args.settings.read_text(encoding='utf-8-sig'))
    result = measure_capacity(settings,args.output,guard)
    print(json.dumps({k:v for k,v in result.items() if k not in ('resources','fingerprint')}))
    return 0 if result['completed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
