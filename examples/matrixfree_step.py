"""Measure the first Newmark acceleration solve on a retained intact mesh."""

import argparse
import json
import math
import os
from pathlib import Path
import sys
import time
import zipfile

import numpy as np

from examples.matrixfree_capacity import checked_settings as capacity_settings
from traxis.config import positive
from traxis.elmer_plate_refinement import reference_source_hashes
from traxis.elmer_reference import run_monitored
from traxis.fem import tone_burst
from traxis.geometry import TetMesh
from traxis.highorder_patches import highorder_surface_patch, patch_polynomial_moments
from traxis.highorder_waveforms import HighOrderWaveSettings, waveform_runtime
from traxis.iterative_fem import IterativeSettings, checked_cg
from traxis.matrixfree import build_matrix_free
from traxis.resources import ResourceGuard, ResourceLimits
from traxis.storage import atomic_write, canonical_hash, sha256_file, write_json


INPUTS = ('capacity-plan.json','monitor-result.json','native/capacity-report.json','native/mesh.npz')


def checked_settings(values, plate):
    required = {'dt_s','frequency_hz','cycles','peak_patch_force_n','center_xy_m','radius_m','iterative','verify_sparse'}
    if not required.issubset(values) or set(values)-required-{'backend','gpu'}:
        raise ValueError('step settings have missing or unknown fields')
    backend = values.get('backend','native')
    if backend not in ('native','gpu_elements') or (backend == 'native' and 'gpu' in values):
        raise ValueError('select the native or explicit GPU element backend')
    if backend == 'gpu_elements':
        gpu = values.get('gpu',{})
        if (set(gpu) != {'max_device_mib','chunk_elements','device_id'}
                or not math.isfinite(gpu['max_device_mib']) or not 0 < gpu['max_device_mib'] <= 6144
                or type(gpu['chunk_elements']) is not int or not 1 <= gpu['chunk_elements'] <= 1024
                or type(gpu['device_id']) is not int or not 0 <= gpu['device_id'] <= 31):
            raise ValueError('GPU step needs explicit device/storage/chunk limits')
    for key in ('dt_s','frequency_hz','peak_patch_force_n','radius_m'):
        positive(key,values[key])
    center = np.asarray(values['center_xy_m'])
    if (np.iscomplexobj(center) or center.shape != (2,) or not np.isfinite(center).all()
            or np.any(center < values['radius_m'])
            or np.any(center > np.array([plate.width_m,plate.height_m])-values['radius_m'])
            or type(values['cycles']) is not int or not 1 <= values['cycles'] <= 10
            or values['dt_s']*values['frequency_hz'] > .05 or type(values['verify_sparse']) is not bool):
        raise ValueError('step needs an interior circular load and at least twenty points per drive cycle')
    return IterativeSettings(**values['iterative'])


def _json(path):
    if path.stat().st_size > 16*1024**2:
        raise MemoryError('step input receipt exceeds 16 MiB')
    value = json.loads(path.read_text())
    if canonical_hash({k:v for k,v in value.items() if k!='fingerprint'}) != value['fingerprint']:
        raise ValueError('step input receipt fingerprint changed')
    return value


def capacity_context(root):
    root = Path(root).resolve()
    plan, parent, native = (_json(root/path) for path in INPUTS[:3])
    plate, material, reference = capacity_settings(plan['settings'])
    if (plan['kind'] != 'matrixfree_fullplate_capacity' or parent['completed'] is not True
            or native['completed'] is not True or parent['plan_fingerprint'] != plan['fingerprint']
            or native['plan_fingerprint'] != plan['fingerprint'] or native['wave_solves'] != 0
            or native['sparse_reference_assembled'] is not False
            or sha256_file(root/INPUTS[2]) != parent['native_report_sha256']
            or sha256_file(root/INPUTS[3]) != parent['mesh_sha256']
            or native['stages'][0]['mesh_sha256'] != parent['mesh_sha256']):
        raise ValueError('step needs complete, unchanged intact capacity evidence')
    current = reference_source_hashes()
    if any(plan['source_sha256'][key] != current[key] for key in ('cad_mesh.py','config.py','geometry.py')):
        raise ValueError('retained mesh geometry implementation changed')
    return plan,native,plate,material,reference


def load_mesh(root, context):
    plan,native,plate,_,reference = context
    path = Path(root)/INPUTS[3]
    byte_limit = reference.max_vertices*24+reference.max_tetrahedra*32+4096
    if path.stat().st_size > byte_limit:
        raise MemoryError('retained mesh exceeds configured storage')
    with zipfile.ZipFile(path) as archive:
        rows = archive.infolist()
        if (len(rows) != 2 or {r.filename for r in rows} != {'points_m.npy','tetrahedra.npy'}
                or sum(r.file_size for r in rows) > byte_limit):
            raise ValueError('invalid or oversized retained mesh archive')
    with np.load(path,allow_pickle=False) as arrays:
        points,tets = arrays['points_m'],arrays['tetrahedra']
    if (points.dtype != np.float64 or points.ndim != 2 or points.shape[1] != 3
            or not 4 <= len(points) <= reference.max_vertices or not np.isfinite(points).all()
            or tets.ndim != 2 or tets.shape[1] != 4 or tets.dtype.kind not in 'iu'
            or not 1 <= len(tets) <= reference.max_tetrahedra or tets.min() < 0 or tets.max() >= len(points)):
        raise ValueError('retained mesh has invalid shape, counts or connectivity')
    bounds = np.array([plate.width_m,plate.height_m,plate.thickness_m])
    tolerance = bounds.max()*1e-10
    if (np.any(points < -tolerance) or np.any(points > bounds+tolerance)
            or not np.allclose(points.min(axis=0),0.,rtol=0,atol=tolerance)
            or not np.allclose(points.max(axis=0),bounds,rtol=0,atol=tolerance)):
        raise ValueError('retained mesh does not cover the declared intact box')
    volume = float(np.abs(np.linalg.det(points[tets[:,1:]]-points[tets[:,:1]])).sum()/6)
    if abs(volume/np.prod(bounds)-1) > 1e-10 or native['dofs'] > reference.max_dofs:
        raise ValueError('retained intact volume or degree limit changed')
    return TetMesh(points,tets,plate,0.,0.)


def driver_hashes():
    return {name:sha256_file(Path(__file__).parent/name) for name in ('matrixfree_step.py','matrixfree_capacity.py')}


def gpu_identity(settings, guard):
    if settings.get('backend','native') == 'gpu_elements':
        from traxis.gpu_verification import gpu_package_identity
        return gpu_package_identity(guard)
    return None


def worker(plan_path, output, guard):
    plan_path,output = Path(plan_path).resolve(),Path(output).resolve()
    plan = _json(plan_path)
    if (plan['kind'] != 'matrixfree_first_step' or plan['source_sha256'] != reference_source_hashes()
            or plan['driver_sha256'] != driver_hashes()
            or plan.get('gpu_packages') != gpu_identity(plan['settings'],guard)
            or plan['runtime'] != waveform_runtime(HighOrderWaveSettings(mesh_backend='cad'),guard)):
        raise ValueError('step plan/source/runtime changed')
    root = Path(plan['capacity_directory'])
    if any(sha256_file(root/name) != digest for name,digest in plan['input_sha256'].items()) or set(plan['input_sha256']) != set(INPUTS):
        raise ValueError('step retained capacity inputs changed')
    context = capacity_context(root)
    _,_,plate,material,reference = context
    iterative = checked_settings(plan['settings'],plate)
    if not output.is_relative_to(plan_path.parent) or (output.exists() and any(output.iterdir())):
        raise ValueError('step output must be new and within its plan directory')
    output.mkdir(parents=True,exist_ok=True)
    started = time.perf_counter()
    report = {'completed':False,'steps_completed':0,'ready_for_training_data':False,'training_updates':0,
        'backend':plan['settings'].get('backend','native'),
        'sparse_reference_assembled':False,
        'plan_fingerprint':plan['fingerprint'],'stages':[],'solve_progress':[],
        'scope':'First zero-initial-state mechanical Newmark step only; no arrival, convergence or PZT acceptance.'}
    def progress(**fields):
        report['stages'].append({'elapsed_seconds':time.perf_counter()-started,**fields})
        write_json(output/'progress.json',report,overwrite=True)
    try:
        mesh = load_mesh(root,context)
        progress(stage='mesh_loaded',vertices=len(mesh.points_m),tetrahedra=len(mesh.tetrahedra))
        cfg = plan['settings']
        if cfg.get('backend','native') == 'gpu_elements':
            from traxis.gpu_elements import build_gpu_elements
            last_upload_time = -math.inf
            def uploaded(event):
                nonlocal last_upload_time
                elapsed = time.perf_counter()-started
                if elapsed-last_upload_time >= 1 or event['elements_uploaded'] == event['total_elements']:
                    progress(stage='elements_uploaded',**event)
                    last_upload_time = elapsed
            system = build_gpu_elements(mesh,material,reference,guard=guard,progress=uploaded,**cfg['gpu'])
            report['device'] = system.device
        else:
            system = build_matrix_free(mesh,material,reference,guard)
        progress(stage='space_and_diagonals',dofs=system.space.ndof,screen=system.screen)
        if cfg['verify_sparse'] and system.space.ndof > 2000:
            raise MemoryError('direct first-step verification is restricted to at most 2000 coefficients')
        load,footprint = highorder_surface_patch(system,plate,cfg['center_xy_m'],cfg['radius_m'],guard=guard,
            max_scalar_dofs=reference.max_dofs//3,max_tetrahedra=reference.max_tetrahedra,max_workspace_mib=256)
        moments = patch_polynomial_moments(system,load,cfg['center_xy_m'],cfg['radius_m'],(0,0,1),'full',reference.order)
        if footprint['full_disk_area_relative_error'] > 1e-10 or moments['maximum_absolute_error'] > 1e-9:
            raise ArithmeticError('full-size circular load area or represented moments failed')
        atomic_write(output/'load.npz',lambda stream:np.savez_compressed(stream,normal_force_covector=load))
        force = cfg['peak_patch_force_n']*tone_burst(cfg['dt_s'],cfg['frequency_hz'],cfg['cycles'])
        if abs(force) <= 1e-20:
            raise ValueError('step source is too weak to measure a nonzero acceleration solve')
        progress(stage='mechanical_load',footprint=footprint,moments=moments,force_n_at_step=force,
            load_sha256=sha256_file(output/'load.npz'))
        def solve_progress(event):
            report['solve_progress'].append({'elapsed_seconds':time.perf_counter()-started,
                'stiffness_applications':system.stiffness.applications,'mass_applications':system.mass.applications,
                'effective_operator_applications':getattr(equation,'applications',None),**event})
            write_json(output/'progress.json',report,overwrite=True)
        dt = cfg['dt_s']
        equation = system.mass+(.25*dt*dt)*system.stiffness
        diagonal = system.mass_diagonal+.25*dt*dt*system.stiffness_diagonal
        before = time.perf_counter()
        result = checked_cg(equation,force*load,np.zeros(system.space.ndof),diagonal,iterative,
            guard=guard,progress=solve_progress)
        solve_seconds = time.perf_counter()-before
        u,v = .25*dt*dt*result.solution,.5*dt*result.solution
        energy = float((v@(system.mass@v)+u@(system.stiffness@u))/2)
        work = float(.5*force*(load@u))
        balance = abs(energy/work-1) if work > 0 else math.inf
        if not math.isfinite(balance) or balance > 1e-8:
            raise ArithmeticError('first-step mechanical work/energy balance failed')
        if cfg['verify_sparse']:
            from scipy.sparse.linalg import spsolve
            from traxis.highorder import assemble_highorder
            from traxis.implicit_fem import implicit_simulate
            sparse = assemble_highorder(mesh,material,reference.order,guard)
            direct_a = spsolve(sparse.mass+.25*dt*dt*sparse.stiffness,force*load)
            direct = implicit_simulate(sparse.stiffness,sparse.mass,1,dt,load[None,:],
                force=lambda t:load*cfg['peak_patch_force_n']*tone_burst(t,cfg['frequency_hz'],cfg['cycles']),guard=guard)
            error = float(np.linalg.norm(result.solution-direct_a)/np.linalg.norm(direct_a))
            probe_error = abs(float(load@u/direct.displacement_m[-1,0])-1)
            if max(error,probe_error) > 1e-7:
                raise ArithmeticError('first-step direct acceleration or Newmark probe comparison failed')
            report.update(sparse_reference_assembled=True,direct_acceleration_relative_l2=error,
                direct_probe_relative_error=probe_error)
        atomic_write(output/'step.npz',lambda stream:np.savez_compressed(stream,acceleration_m_s2=result.solution))
        if (reference_source_hashes() != plan['source_sha256'] or driver_hashes() != plan['driver_sha256']
                or gpu_identity(cfg,guard) != plan.get('gpu_packages')
                or waveform_runtime(HighOrderWaveSettings(mesh_backend='cad'),guard) != plan['runtime']):
            raise ValueError('step source/runtime changed during the solve')
        report.update(completed=True,steps_completed=1,solve_seconds=solve_seconds,iterations=result.iterations,
            true_relative_residual=result.relative_residual,source_mean_displacement_m=float(load@u),
            energy_j=energy,work_j=work,relative_energy_balance_error=balance,
            acceleration_sha256=sha256_file(output/'step.npz'))
    except Exception as exc:
        report['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        report['wall_seconds'] = time.perf_counter()-started
        report['resources'] = guard.report()
        report['fingerprint'] = canonical_hash(report)
        write_json(output/'step-report.json',report)
    return report


def measure_step(capacity_directory, settings, output, guard=None):
    root,output = Path(capacity_directory).resolve(),Path(output).resolve()
    context = capacity_context(root)
    checked_settings(settings,context[2])
    if settings['verify_sparse'] and context[1]['dofs'] > 2000:
        raise MemoryError('direct first-step verification is restricted to at most 2000 coefficients')
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('step measurement needs a fresh output directory')
    guard = guard or ResourceGuard(ResourceLimits(max_seconds=180,max_rss_mib=1024,
        max_system_memory_percent=85,min_available_mib=2048))
    output.mkdir(parents=True,exist_ok=True)
    result = {'completed':False,'steps_completed':0,'ready_for_training_data':False,'training_updates':0}
    try:
        runtime = waveform_runtime(HighOrderWaveSettings(mesh_backend='cad'),guard)
        if runtime != context[0]['runtime']:
            raise ValueError('retained mesh native runtime changed')
        plan = {'kind':'matrixfree_first_step','capacity_directory':str(root),'settings':settings,
            'input_sha256':{name:sha256_file(root/name) for name in INPUTS},'source_sha256':reference_source_hashes(),
            'driver_sha256':driver_hashes(),'runtime':runtime}
        plan['gpu_packages'] = gpu_identity(settings,guard)
        plan['fingerprint'] = canonical_hash(plan)
        write_json(output/'step-plan.json',plan)
        def snapshot(stream):
            with zipfile.ZipFile(stream,'w',compression=zipfile.ZIP_DEFLATED) as archive:
                for name in plan['source_sha256']:
                    archive.writestr('traxis/'+name,(Path(__file__).resolve().parents[1]/'traxis'/name).read_bytes())
                for name in plan['driver_sha256']:
                    archive.writestr('examples/'+name,(Path(__file__).parent/name).read_bytes())
        atomic_write(output/'source-snapshot.zip',snapshot)
        result.update(plan_fingerprint=plan['fingerprint'],source_snapshot_sha256=sha256_file(output/'source-snapshot.zip'))
        remaining = max(1.,guard.limits.max_seconds-(time.monotonic()-guard.started))
        command = [sys.executable,'-m','examples.matrixfree_step','--worker','--plan',str(output/'step-plan.json'),
            '--output',str(output/'native'),'--max-seconds',str(remaining),'--max-rss-mib',str(guard.limits.max_rss_mib)]
        temporary,cache = output/'temporary',output/'kernel-cache'
        temporary.mkdir()
        cache.mkdir()
        env = {**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','TEMP':str(temporary),'TMP':str(temporary),
            'TMPDIR':str(temporary),'CUPY_CACHE_DIR':str(cache),'CUPY_CACHE_SAVE_CUDA_SOURCE':'1'}
        result['worker_run'] = run_monitored(command,Path(__file__).resolve().parents[1],env,guard,output/'worker.log')
        if result['worker_run']['exit_code'] != 0:
            raise RuntimeError('step worker stopped; inspect retained progress/report')
        native = _json(output/'native/step-report.json')
        if (native['completed'] is not True or native['steps_completed'] != 1 or native['plan_fingerprint'] != plan['fingerprint']
                or reference_source_hashes() != plan['source_sha256'] or driver_hashes() != plan['driver_sha256']
                or gpu_identity(settings,guard) != plan.get('gpu_packages')
                or native['sparse_reference_assembled'] != settings['verify_sparse']
                or sha256_file(output/'native/step.npz') != native['acceleration_sha256']
                or sha256_file(output/'native/load.npz') != native['stages'][-1]['load_sha256']):
            raise ValueError('step worker/source/array receipt changed')
        result.update(completed=True,steps_completed=1,iterations=native['iterations'],solve_seconds=native['solve_seconds'],
            native_report_sha256=sha256_file(output/'native/step-report.json'))
    except Exception as exc:
        result['error'] = f'{type(exc).__name__}: {exc}'
    finally:
        progress_path = output/'native/progress.json'
        if progress_path.is_file():
            result['last_progress_sha256'] = sha256_file(progress_path)
        result['resources'] = guard.report()
        result['fingerprint'] = canonical_hash(result)
        write_json(output/'monitor-result.json',result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capacity',type=Path)
    parser.add_argument('--settings',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    parser.add_argument('--plan',type=Path)
    parser.add_argument('--max-seconds',type=float,default=180)
    parser.add_argument('--max-rss-mib',type=float,default=1024)
    args = parser.parse_args()
    guard = ResourceGuard(ResourceLimits(max_seconds=args.max_seconds,max_rss_mib=args.max_rss_mib,
        max_system_memory_percent=85,min_available_mib=2048))
    if args.worker:
        worker(args.plan,args.output,guard)
        return 0
    if not args.capacity or not args.settings or args.settings.stat().st_size > 1024**2:
        parser.error('provide retained capacity evidence and a bounded settings file')
    result = measure_step(args.capacity,json.loads(args.settings.read_text(encoding='utf-8-sig')),args.output,guard)
    print(json.dumps({k:v for k,v in result.items() if k not in ('resources','fingerprint')}))
    return 0 if result['completed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
