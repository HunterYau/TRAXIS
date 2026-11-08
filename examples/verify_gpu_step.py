"""Check a saved GPU first step with independent native CPU operator actions."""

import argparse
import json
import os
from pathlib import Path
import sys
import time
import zipfile

import numpy as np

from examples.matrixfree_step import _json,capacity_context,driver_hashes,load_mesh
from traxis.elmer_plate_refinement import reference_source_hashes
from traxis.elmer_reference import run_monitored
from traxis.fem import tone_burst
from traxis.highorder_patches import highorder_surface_patch
from traxis.highorder_waveforms import HighOrderWaveSettings,waveform_runtime
from traxis.matrixfree import build_matrix_free
from traxis.resources import ResourceGuard,ResourceLimits
from traxis.storage import atomic_write,canonical_hash,sha256_file,write_json


INPUTS = ('step-plan.json','monitor-result.json','native/step-report.json','native/load.npz','native/step.npz')


def gpu_context(root):
    root = Path(root).resolve()
    plan,parent,native = (_json(root/name) for name in INPUTS[:3])
    if (plan['kind'] != 'matrixfree_first_step' or native.get('backend') != 'gpu_elements'
            or parent['completed'] is not True or native['completed'] is not True
            or native['steps_completed'] != 1 or native['plan_fingerprint'] != plan['fingerprint']
            or parent['plan_fingerprint'] != plan['fingerprint']
            or parent['native_report_sha256'] != sha256_file(root/INPUTS[2])
            or native['acceleration_sha256'] != sha256_file(root/INPUTS[4])
            or native['stages'][-1]['load_sha256'] != sha256_file(root/INPUTS[3])
            or plan['source_sha256'] != reference_source_hashes() or plan['driver_sha256'] != driver_hashes()):
        raise ValueError('CPU check needs a complete unchanged GPU step and matching equation source')
    for name,digest in plan['input_sha256'].items():
        if sha256_file(Path(plan['capacity_directory'])/name) != digest:
            raise ValueError('GPU step retained geometry inputs changed')
    return plan,native


def vector(path, name, count):
    byte_limit = count*8+4096
    if path.stat().st_size > byte_limit:
        raise MemoryError('saved GPU vector exceeds its coefficient bound')
    with zipfile.ZipFile(path) as archive:
        rows = archive.infolist()
        if len(rows) != 1 or rows[0].filename != name+'.npy' or rows[0].file_size > byte_limit:
            raise ValueError('saved GPU vector has invalid members or expansion size')
    with np.load(path,allow_pickle=False) as arrays:
        result = arrays[name]
    if result.dtype != np.float64 or result.shape != (count,) or not np.isfinite(result).all():
        raise ValueError('saved GPU vector has invalid shape, dtype or values')
    return result


def worker(plan_path, output, guard):
    plan_path,output = Path(plan_path).resolve(),Path(output).resolve()
    job = _json(plan_path)
    if (job['kind'] != 'gpu_step_cpu_residual' or job['source_sha256'] != reference_source_hashes()
            or job['driver_sha256'] != sha256_file(Path(__file__))
            or job['runtime'] != waveform_runtime(HighOrderWaveSettings(mesh_backend='cad'),guard)):
        raise ValueError('CPU residual check source/runtime changed')
    root = Path(job['gpu_directory'])
    if set(job['input_sha256']) != set(INPUTS) or any(sha256_file(root/n) != d for n,d in job['input_sha256'].items()):
        raise ValueError('GPU check inputs changed')
    plan,native = gpu_context(root)
    context = capacity_context(plan['capacity_directory'])
    if not output.is_relative_to(plan_path.parent) or (output.exists() and any(output.iterdir())):
        raise ValueError('CPU residual output must be fresh and inside its plan directory')
    output.mkdir(parents=True,exist_ok=True)
    report = {'completed':False,'passed':False,'ready_for_training_data':False,'training_updates':0,
        'job_fingerprint':job['fingerprint'],'scope':'Full-size discrete GPU state checked with independent CPU actions; no continuum/PZT acceptance.'}
    started = time.perf_counter()
    try:
        mesh = load_mesh(plan['capacity_directory'],context)
        _,_,plate,material,settings = context
        system = build_matrix_free(mesh,material,settings,guard)
        count = system.space.ndof
        acceleration = vector(root/'native/step.npz','acceleration_m_s2',count)
        saved_load = vector(root/'native/load.npz','normal_force_covector',count)
        cfg = plan['settings']
        load,footprint = highorder_surface_patch(system,plate,cfg['center_xy_m'],cfg['radius_m'],guard=guard,
            max_scalar_dofs=settings.max_dofs//3,max_tetrahedra=settings.max_tetrahedra,max_workspace_mib=256)
        load_error = float(np.linalg.norm(load-saved_load)/np.linalg.norm(load))
        dt = cfg['dt_s']
        force = cfg['peak_patch_force_n']*tone_burst(dt,cfg['frequency_hz'],cfg['cycles'])
        before = time.perf_counter()
        ma,ka = system.mass@acceleration,system.stiffness@acceleration
        action_seconds = time.perf_counter()-before
        residual = float(np.linalg.norm(ma+.25*dt*dt*ka-force*load)/np.linalg.norm(force*load))
        energy = float(((.5*dt)**2*(acceleration@ma)+(.25*dt*dt)**2*(acceleration@ka))/2)
        displacement = float(.25*dt*dt*(load@acceleration))
        work = .5*force*displacement
        if energy <= 0 or work <= 0 or native['energy_j'] <= 0:
            raise ArithmeticError('CPU-check energy or source work is nonpositive')
        balance,energy_error = abs(energy/work-1),abs(energy/native['energy_j']-1)
        probe_error = abs(displacement/native['source_mean_displacement_m']-1)
        passed = (load_error <= 1e-10 and footprint['full_disk_area_relative_error'] <= 1e-10
            and residual <= cfg['iterative']['residual_limit'] and max(balance,energy_error,probe_error) <= 1e-8)
        if reference_source_hashes() != job['source_sha256'] or driver_hashes() != plan['driver_sha256']:
            raise ValueError('CPU-check source changed during execution')
        report.update(completed=True,passed=bool(passed),dofs=count,cpu_action_seconds=action_seconds,
            cpu_relative_residual=residual,gpu_saved_relative_residual=native['true_relative_residual'],
            source_covector_relative_l2=load_error,cpu_energy_j=energy,cpu_work_j=work,
            cpu_relative_work_energy_error=balance,cpu_gpu_energy_relative_error=energy_error,
            cpu_gpu_probe_relative_error=probe_error,footprint=footprint,
            sparse_reference_assembled=False,cpu_cg_solves=0)
    except Exception as exc:
        report['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        report['wall_seconds'] = time.perf_counter()-started
        report['resources'] = guard.report()
        report['fingerprint'] = canonical_hash(report)
        write_json(output/'cpu-check.json',report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu-result',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    parser.add_argument('--plan',type=Path)
    parser.add_argument('--max-seconds',type=float,default=120)
    parser.add_argument('--max-rss-mib',type=float,default=1024)
    args = parser.parse_args()
    guard = ResourceGuard(ResourceLimits(max_seconds=args.max_seconds,max_rss_mib=args.max_rss_mib,
        max_system_memory_percent=85,min_available_mib=2048))
    if args.worker:
        return 0 if worker(args.plan,args.output,guard)['passed'] else 2
    if not args.gpu_result:
        parser.error('provide a retained GPU first-step result')
    root,output = args.gpu_result.resolve(),args.output.resolve()
    gpu_context(root)
    if output.exists() and any(output.iterdir()):
        parser.error('CPU residual check needs a fresh output directory')
    output.mkdir(parents=True,exist_ok=True)
    result = {'completed':False,'passed':False,'ready_for_training_data':False,'training_updates':0}
    try:
        job = {'kind':'gpu_step_cpu_residual','gpu_directory':str(root),'source_sha256':reference_source_hashes(),
            'driver_sha256':sha256_file(Path(__file__)),'runtime':waveform_runtime(HighOrderWaveSettings(mesh_backend='cad'),guard),
            'input_sha256':{n:sha256_file(root/n) for n in INPUTS}}
        job['fingerprint'] = canonical_hash(job)
        write_json(output/'cpu-check-plan.json',job)
        def snapshot(stream):
            with zipfile.ZipFile(stream,'w',compression=zipfile.ZIP_DEFLATED) as archive:
                for name in job['source_sha256']:
                    archive.writestr('traxis/'+name,(Path(__file__).resolve().parents[1]/'traxis'/name).read_bytes())
                for name in ('matrixfree_step.py','matrixfree_capacity.py','verify_gpu_step.py'):
                    archive.writestr('examples/'+name,(Path(__file__).parent/name).read_bytes())
        atomic_write(output/'source-snapshot.zip',snapshot)
        result['source_snapshot_sha256'] = sha256_file(output/'source-snapshot.zip')
        remaining = max(1.,guard.limits.max_seconds-(time.monotonic()-guard.started))
        command = [sys.executable,'-m','examples.verify_gpu_step','--worker','--plan',str(output/'cpu-check-plan.json'),
            '--output',str(output/'native'),'--max-seconds',str(remaining),'--max-rss-mib',str(guard.limits.max_rss_mib)]
        result['worker_run'] = run_monitored(command,Path(__file__).resolve().parents[1],
            {**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'},guard,output/'worker.log')
        if result['worker_run']['exit_code'] not in (0,2):
            raise RuntimeError('CPU residual worker stopped; inspect retained report/log')
        native = _json(output/'native/cpu-check.json')
        if native['completed'] is not True or native['job_fingerprint'] != job['fingerprint']:
            raise ValueError('CPU residual receipt changed')
        result.update(completed=True,passed=native['passed'],cpu_relative_residual=native['cpu_relative_residual'],
            native_report_sha256=sha256_file(output/'native/cpu-check.json'))
    except Exception as exc:
        result['error'] = f'{type(exc).__name__}: {exc}'
    finally:
        result['resources'] = guard.report()
        result['fingerprint'] = canonical_hash(result)
        write_json(output/'monitor-result.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('resources','fingerprint')}))
    return 0 if result['passed'] else (2 if result['completed'] else 1)


if __name__ == '__main__':
    raise SystemExit(main())
