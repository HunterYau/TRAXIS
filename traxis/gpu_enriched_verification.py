"""Isolated every-column checks for factorized enriched GPU elasticity."""

import argparse
import json
import os
from pathlib import Path
import sys
import time
import zipfile

import numpy as np
from scipy.linalg import eigh

from traxis.config import Material
from traxis.elmer_plate_refinement import reference_source_hashes
from traxis.elmer_reference import run_monitored
from traxis.enriched_tet import assemble_enriched
from traxis.enriched_verification import recipes
from traxis.gpu_enriched import build_gpu_enriched
from traxis.gpu_verification import gpu_package_identity
from traxis.highorder_verification import distribution_identity,reference_mesh
from traxis.resources import ResourceGuard,ResourceLimits
from traxis.storage import atomic_write,canonical_hash,sha256_file,write_json


def worker(plan_path,output,guard):
    plan_path,output = Path(plan_path).resolve(),Path(output).resolve()
    if plan_path.stat().st_size>1024**2:
        raise MemoryError('enriched GPU plan exceeds 1 MiB')
    plan = json.loads(plan_path.read_text())
    if (plan.get('kind')!='factorized_enriched_gpu_reference' or plan.get('meshes')!=recipes()
            or plan.get('operator_tolerance')!=1e-10 or plan.get('mass_tolerance')!=1e-12
            or canonical_hash({k:v for k,v in plan.items() if k!='fingerprint'})!=plan['fingerprint']
            or plan['source_sha256']!=reference_source_hashes() or plan['runtime']!=distribution_identity(guard)
            or plan['gpu_packages']!=gpu_package_identity(guard)):
        raise ValueError('factorized GPU source/runtime/cases/criteria changed')
    if not output.is_relative_to(plan_path.parent) or (output.exists() and any(output.iterdir())):
        raise ValueError('factorized GPU reference output must be fresh and within its plan directory')
    output.mkdir(parents=True,exist_ok=True)
    started = time.perf_counter()
    report = dict(completed=False,passed=False,ready_for_training_data=False,training_updates=0,
        plan_fingerprint=plan['fingerprint'],cases=[])
    try:
        for recipe in plan['meshes']:
            mesh,material = reference_mesh(recipe),Material(**recipe['material'])
            reference = assemble_enriched(mesh,material,guard)
            gpu = build_gpu_enriched(mesh,material,guard=guard,chunk_elements=3)
            n = gpu.scalar_space.ndof
            permutation = np.concatenate([gpu.native_to_topology+c*n for c in range(3)])
            exact = reference.stiffness.toarray()[np.ix_(permutation,permutation)]
            mass = reference.mass_kg[permutation]
            columns = []
            for column in range(3*n):
                guard.check()
                unit = np.zeros(3*n)
                unit[column] = 1.
                columns.append(gpu.stiffness@unit)
            represented = np.column_stack(columns)
            error = float(np.linalg.norm(represented-exact)/np.linalg.norm(exact))
            mass_error = float(np.linalg.norm(gpu.mass_diagonal-mass)/np.linalg.norm(mass))
            diagonal_error = float(np.linalg.norm(gpu.stiffness_diagonal-exact.diagonal())/np.linalg.norm(exact.diagonal()))
            rng = np.random.default_rng(2026)
            block = rng.normal(size=(3*n,2))
            first = gpu.stiffness@block[:,0]
            kept = first.copy()
            gpu.stiffness@block[:,1]
            if not np.array_equal(first,kept):
                raise ArithmeticError('factorized GPU application overwrote returned data')
            block_error = float(np.linalg.norm(gpu.stiffness@block-exact@block)/np.linalg.norm(exact@block))
            eigenvalues = eigh(exact,np.diag(mass),eigvals_only=True)
            stability = bool(gpu.stable_dt_s<=2/np.sqrt(eigenvalues[-1])*(1+1e-12))
            case = output/recipe['name']
            case.mkdir()
            arrays = case/'comparison.npz'
            atomic_write(arrays,lambda stream:np.savez_compressed(stream,gpu_stiffness=represented,
                reference_stiffness=exact,nodal_mass_kg=gpu.mass_diagonal,reference_mass_kg=mass,
                native_to_topology=gpu.native_to_topology,points_m=gpu.points_m))
            row = dict(geometry=recipe['name'],dofs=3*n,all_columns_checked=3*n,
                stiffness_relative_frobenius=error,mass_relative_l2=mass_error,
                stiffness_diagonal_relative_l2=diagonal_error,block_relative_l2=block_error,
                native_topology_map_verified=gpu.screen['native_topology_permutation_verified'],
                stable_dt_is_conservative=stability,conservative_dt_s=gpu.stable_dt_s,
                exact_discrete_stable_dt_s=float(2/np.sqrt(eigenvalues[-1])),screen=gpu.screen,device=gpu.device,
                comparison_file=arrays.relative_to(output).as_posix(),comparison_sha256=sha256_file(arrays))
            row['passed'] = bool(max(error,diagonal_error,block_error)<plan['operator_tolerance']
                and mass_error<plan['mass_tolerance'] and stability and row['native_topology_map_verified'])
            report['cases'].append(row)
        if (reference_source_hashes()!=plan['source_sha256'] or distribution_identity(guard)!=plan['runtime']
                or gpu_package_identity(guard)!=plan['gpu_packages']):
            raise ValueError('factorized GPU source/runtime changed during checks')
        report.update(completed=True,passed=all(r['passed'] for r in report['cases']),
            scope='Every small enriched GPU stiffness coefficient and native map/mass/CFL check; no full-window or PZT acceptance.')
    except Exception as error:
        report['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        report['wall_seconds'] = time.perf_counter()-started
        report['resources'] = guard.report()
        report['fingerprint'] = canonical_hash(report)
        write_json(output/'native-report.json',report)
    return report


def verify_gpu_enriched(output,guard=None):
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('factorized GPU reference needs a fresh output directory')
    guard = guard or ResourceGuard(ResourceLimits(max_seconds=90,max_rss_mib=1024,
        max_system_memory_percent=85,min_available_mib=2048))
    output.mkdir(parents=True,exist_ok=True)
    result = dict(completed=False,passed=False,ready_for_training_data=False,training_updates=0)
    try:
        plan = dict(kind='factorized_enriched_gpu_reference',meshes=recipes(),operator_tolerance=1e-10,
            mass_tolerance=1e-12,source_sha256=reference_source_hashes(),runtime=distribution_identity(guard),
            gpu_packages=gpu_package_identity(guard))
        plan['fingerprint'] = canonical_hash(plan)
        write_json(output/'gpu-enriched-plan.json',plan)
        def snapshot(stream):
            with zipfile.ZipFile(stream,'w',compression=zipfile.ZIP_DEFLATED) as archive:
                for name in plan['source_sha256']:
                    archive.writestr('traxis/'+name,(Path(__file__).parent/name).read_bytes())
        atomic_write(output/'source-snapshot.zip',snapshot)
        result.update(plan_fingerprint=plan['fingerprint'],source_snapshot_sha256=sha256_file(output/'source-snapshot.zip'))
        temporary,cache = output/'temporary',output/'kernel-cache'
        temporary.mkdir()
        cache.mkdir()
        remaining = max(1.,guard.limits.max_seconds-(time.monotonic()-guard.started))
        command = [sys.executable,'-m','traxis.gpu_enriched_verification','--worker','--plan',str(output/'gpu-enriched-plan.json'),
            '--output',str(output/'native'),'--max-seconds',str(remaining),'--max-rss-mib',str(guard.limits.max_rss_mib)]
        env = {**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','CUPY_CACHE_DIR':str(cache),
            'CUPY_CACHE_SAVE_CUDA_SOURCE':'1','TEMP':str(temporary),'TMP':str(temporary),'TMPDIR':str(temporary)}
        result['worker_run'] = run_monitored(command,Path(__file__).resolve().parents[1],env,guard,output/'worker.log')
        if result['worker_run']['exit_code'] not in (0,2):
            raise RuntimeError('factorized GPU reference worker stopped; inspect log/report')
        native = json.loads((output/'native/native-report.json').read_text())
        if (canonical_hash({k:v for k,v in native.items() if k!='fingerprint'})!=native['fingerprint']
                or native['completed'] is not True or native['plan_fingerprint']!=plan['fingerprint']
                or [r['geometry'] for r in native['cases']]!=[r['name'] for r in plan['meshes']]
                or native['passed']!=all(r['passed'] for r in native['cases'])):
            raise ValueError('factorized GPU reference receipt changed')
        for row in native['cases']:
            path = (output/'native'/row['comparison_file']).resolve()
            if not path.is_relative_to(output/'native') or sha256_file(path)!=row['comparison_sha256']:
                raise ValueError('factorized GPU comparison archive changed')
        result.update(completed=True,passed=native['passed'],cases=native['cases'],
            native_report_sha256=sha256_file(output/'native/native-report.json'))
    except Exception as error:
        result['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        result['resources'] = guard.report()
        result['fingerprint'] = canonical_hash(result)
        write_json(output/'monitor-result.json',result)
    return result


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--plan',type=Path)
    parser.add_argument('--worker',action='store_true')
    parser.add_argument('--max-seconds',type=float,default=90)
    parser.add_argument('--max-rss-mib',type=float,default=1024)
    args = parser.parse_args()
    guard = ResourceGuard(ResourceLimits(max_seconds=args.max_seconds,max_rss_mib=args.max_rss_mib,
        max_system_memory_percent=85,min_available_mib=2048))
    result = worker(args.plan,args.output,guard) if args.worker else verify_gpu_enriched(args.output,guard)
    print(json.dumps({k:result[k] for k in ('completed','passed','ready_for_training_data')}))
    raise SystemExit(0 if result['passed'] else 2)
