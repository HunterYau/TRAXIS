"""Compare native matrix-free elasticity with the checked sparse reference."""

import argparse
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time
import zipfile

import numpy as np

from traxis.config import Material
from traxis.elmer_plate_refinement import reference_source_hashes
from traxis.elmer_reference import run_monitored
from traxis.fem import tone_burst
from traxis.highorder import assemble_highorder
from traxis.highorder_patches import highorder_surface_patch
from traxis.highorder_verification import distribution_identity, reference_mesh
from traxis.implicit_fem import implicit_simulate
from traxis.iterative_fem import IterativeSettings, iterative_simulate
from traxis.matrixfree import MatrixFreeSettings, build_matrix_free
from traxis.resources import ResourceGuard, ResourceLimits
from traxis.storage import atomic_write, canonical_hash, sha256_file, write_json


def _reference_recipes():
    return [{'name':name,'plate':{'width_m':.04,'height_m':.03,'thickness_m':.003},'cells':[2,2,1],
        'transform':transform,'material':asdict(material)} for name,transform,material in (
        ('rectangular',np.eye(3).tolist(),Material()),
        ('affine-skewed',[[1.,.12,.04],[.05,.9,.03],[.01,.02,1.1]],Material(71e9,.29,2800.)))]


def _column_error(operator, reference, guard):
    reference = reference.tocsc()
    error_squared = 0.
    for column in range(reference.shape[1]):
        guard.check()
        unit = np.zeros(reference.shape[1])
        unit[column] = 1.
        difference = operator @ unit
        start, end = reference.indptr[column:column+2]
        difference[reference.indices[start:end]] -= reference.data[start:end]
        error_squared += float(difference@difference)
    return float(np.sqrt(error_squared)/np.linalg.norm(reference.data))


def _checked_case(row, root, name, plan):
    """Check complete case evidence before reuse or parent-side acceptance."""
    geometry, degree = name.rsplit('-p',1)
    degree = int(degree)
    recipe = next(r for r in plan['meshes'] if r['name'] == geometry)
    dofs = 3*math.prod(degree*n+1 for n in recipe['cells'])
    metrics = ('stiffness_relative_frobenius','mass_relative_frobenius','mass_diagonal_relative_l2',
        'stiffness_diagonal_relative_l2','block_application_relative_l2','waveform_relative_l2',
        'postburst_relative_energy_range','maximum_relative_residual')
    counts = ('total_cg_iterations','maximum_cg_iterations','stiffness_applications','mass_applications')
    required = {'geometry','order','dofs','all_coefficient_columns_checked',*metrics,*counts,
        'native_operator_type','interface','passed','comparison_file','comparison_sha256',
        'cache_hit','computation_seconds'}
    interface = ('physical circular normal-force/mean-displacement covectors' if geometry == 'rectangular'
        else 'manufactured coefficient-space covectors on an affine block')
    if (not isinstance(row,dict) or not required.issubset(row)
            or set(row)-required-{'origin_receipt_sha256'}
            or row['geometry'] != geometry or type(row['order']) is not int or row['order'] != degree
            or type(row['dofs']) is not int or row['dofs'] != dofs
            or type(row['all_coefficient_columns_checked']) is not int or row['all_coefficient_columns_checked'] != dofs
            or type(row['passed']) is not bool or type(row['cache_hit']) is not bool
            or row['comparison_file'] != f'{name}/comparison.npz' or row['interface'] != interface
            or not isinstance(row['native_operator_type'],str) or not 1 <= len(row['native_operator_type']) <= 80
            or any(type(row[k]) not in (int,float) or not math.isfinite(row[k]) or row[k] < 0 for k in (*metrics,'computation_seconds'))
            or any(type(row[k]) is not int or row[k] < 0 for k in counts)
            or min(row['stiffness_applications'],row['mass_applications']) < dofs
            or row['maximum_relative_residual'] > plan['iterative_settings']['residual_limit']):
        raise ValueError('incomplete or invalid matrix-free case receipt')
    root = Path(root).resolve()
    path = (root/row['comparison_file']).resolve()
    if not path.is_relative_to(root) or not path.is_file() or path.stat().st_size > 1024**2:
        raise ValueError('matrix-free comparison archive is absent, escaped or oversized')
    if sha256_file(path) != row['comparison_sha256']:
        raise ValueError('matrix-free comparison arrays changed')
    names = {'time_s','direct_m','iterative_m','energy_time_s','energy_j','iterations'}
    with zipfile.ZipFile(path) as archive:
        members = archive.infolist()
        if (len(members) != len(names) or {r.filename for r in members} != {n+'.npy' for n in names}
                or sum(r.file_size for r in members) > 1024**2):
            raise ValueError('matrix-free comparison archive has invalid members or expanded size')
    with np.load(path,allow_pickle=False) as arrays:
        values = {n:arrays[n] for n in names}
    times = np.arange(33)*2.5e-7
    for key in names-{'iterations'}:
        expected_shape = (33,1) if key in {'direct_m','iterative_m'} else (33,)
        if values[key].dtype != np.float64 or values[key].shape != expected_shape or not np.isfinite(values[key]).all():
            raise ValueError('matrix-free comparison arrays have invalid shapes, dtypes or values')
    iterations = values['iterations']
    if (not np.array_equal(values['time_s'],times) or not np.array_equal(values['energy_time_s'],times)
            or iterations.shape != (33,) or iterations.dtype.kind not in 'iu'
            or np.any(iterations > plan['iterative_settings']['max_iterations']) or np.any(iterations < 0)
            or int(iterations.sum()) != row['total_cg_iterations'] or int(iterations.max()) != row['maximum_cg_iterations']
            or np.any(values['direct_m'][0]) or np.any(values['iterative_m'][0])
            or values['energy_j'][0] != 0. or np.linalg.norm(values['direct_m']) <= 1e-30):
        raise ValueError('matrix-free comparison times, iterations or initial values changed')
    waveform = float(np.linalg.norm(values['iterative_m']-values['direct_m'])/np.linalg.norm(values['direct_m']))
    after = values['energy_j'][times > 5e-6]
    if np.any(after <= 0):
        raise ValueError('matrix-free comparison has nonpositive postburst energy')
    energy = float(np.ptp(after)/abs(after.mean()))
    if (not math.isclose(waveform,row['waveform_relative_l2'],rel_tol=1e-12,abs_tol=1e-20)
            or not math.isclose(energy,row['postburst_relative_energy_range'],rel_tol=1e-12,abs_tol=1e-20)):
        raise ValueError('matrix-free saved arrays disagree with reported metrics')
    passed = (max(row[k] for k in metrics[:5]) <= plan['operator_tolerance']
        and waveform <= plan['waveform_tolerance'] and energy <= plan['energy_tolerance'])
    if row['passed'] != passed:
        raise ValueError('matrix-free saved case screen changed')
    return path


def _cached_case(cache, name, output, plan):
    receipt_path = Path(cache)/name/'case-receipt.json'
    if not receipt_path.is_file():
        return None
    if receipt_path.stat().st_size > 1024**2:
        raise MemoryError('matrix-free case receipt exceeds 1 MiB')
    receipt = json.loads(receipt_path.read_text())
    if (not isinstance(receipt,dict) or set(receipt) != {'case','plan_fingerprint','fingerprint'}
            or canonical_hash({k:v for k,v in receipt.items() if k!='fingerprint'}) != receipt['fingerprint']
            or receipt['plan_fingerprint'] != plan['fingerprint']):
        raise ValueError('cached matrix-free case plan/source/runtime changed')
    row = receipt['case']
    source = _checked_case(row,cache,name,plan)
    case = output/name
    case.mkdir()
    shutil.copy2(source,case/'comparison.npz')
    row = {**row,'cache_hit':True,'origin_receipt_sha256':sha256_file(receipt_path)}
    updated = {'plan_fingerprint':plan['fingerprint'],'case':row}
    updated['fingerprint'] = canonical_hash(updated)
    write_json(case/'case-receipt.json',updated)
    return row


def run_worker(plan_path, output, guard, cache_dir=None):
    plan_path, output = Path(plan_path).resolve(), Path(output).resolve()
    if plan_path.stat().st_size > 1024**2:
        raise MemoryError('matrix-free verification plan exceeds 1 MiB')
    plan = json.loads(plan_path.read_text())
    if (not plan.get('orders') or len(set(plan['orders'])) != len(plan['orders'])
            or any(type(p) is not int or not 1 <= p <= 4 for p in plan['orders'])
            or plan.get('meshes') != _reference_recipes()
            or plan.get('iterative_settings') != asdict(IterativeSettings(max_iterations=500))
            or plan.get('operator_tolerance') != 1e-10 or plan.get('waveform_tolerance') != 1e-7
            or plan.get('energy_tolerance') != 1e-8):
        raise ValueError('matrix-free verification cases or fixed tolerances changed')
    if (canonical_hash({k:v for k,v in plan.items() if k != 'fingerprint'}) != plan['fingerprint']
            or plan['kind'] != 'matrix_free_reference_check' or reference_source_hashes() != plan['source_sha256']
            or distribution_identity(guard) != plan['runtime']):
        raise ValueError('matrix-free plan/source/runtime changed')
    if not output.is_relative_to(plan_path.parent) or (output.exists() and any(output.iterdir())):
        raise ValueError('matrix-free output must be new and inside its plan directory')
    output.mkdir(parents=True, exist_ok=True)
    report = {'completed': False, 'passed': False, 'ready_for_training_data': False,
        'training_updates': 0, 'plan_fingerprint': plan['fingerprint'], 'cases': []}
    started = time.perf_counter()
    try:
        for recipe in plan['meshes']:
            mesh, material = reference_mesh(recipe), Material(**recipe['material'])
            for order in plan['orders']:
                name = f"{recipe['name']}-p{order}"
                if cache_dir:
                    cached = _cached_case(cache_dir,name,output,plan)
                    if cached is not None:
                        report['cases'].append(cached)
                        continue
                case_started = time.perf_counter()
                case = output/name
                case.mkdir()
                sparse = assemble_highorder(mesh,material,order,guard)
                free = build_matrix_free(mesh,material,MatrixFreeSettings(order=order),guard)
                stiffness_error = _column_error(free.stiffness,sparse.stiffness,guard)
                mass_error = _column_error(free.mass,sparse.mass,guard)
                diagonal_error = float(np.linalg.norm(free.mass_diagonal-sparse.mass.diagonal())/np.linalg.norm(sparse.mass.diagonal()))
                stiffness_diagonal_error = float(np.linalg.norm(free.stiffness_diagonal-sparse.stiffness.diagonal())/np.linalg.norm(sparse.stiffness.diagonal()))
                rng = np.random.default_rng(2026+order)
                vectors = rng.normal(size=(free.space.ndof,2))
                first = free.mass @ vectors[:,0]
                retained = first.copy()
                free.mass @ vectors[:,1]
                if not np.array_equal(first,retained):
                    raise ArithmeticError('native operator reused a returned output buffer')
                block_error = float(np.linalg.norm(free.mass@vectors-sparse.mass@vectors)/np.linalg.norm(sparse.mass@vectors))
                if recipe['name'] == 'rectangular':
                    load, _ = highorder_surface_patch(free,mesh.plate,(.01,.008),.003,guard=guard)
                    probe, _ = highorder_surface_patch(free,mesh.plate,(.03,.022),.003,guard=guard)
                    probes, interface = probe[None,:], 'physical circular normal-force/mean-displacement covectors'
                else:
                    load = vectors[:,0]/np.linalg.norm(vectors[:,0])
                    probes = (vectors[:,1]/np.linalg.norm(vectors[:,1]))[None,:]
                    interface = 'manufactured coefficient-space covectors on an affine block'
                force = lambda t:load*tone_burst(float(t),200000.,1)
                direct = implicit_simulate(sparse.stiffness,sparse.mass,32,2.5e-7,probes,force=force,guard=guard)
                iterative = iterative_simulate(free.stiffness,free.mass,free.mass_diagonal,32,2.5e-7,probes,
                    force=force,stiffness_diagonal=free.stiffness_diagonal,
                    settings=IterativeSettings(**plan['iterative_settings']),guard=guard)
                if not np.array_equal(direct.time_s,iterative.time_s):
                    raise ArithmeticError('matrix-free output time grid changed')
                waveform_error = float(np.linalg.norm(iterative.displacement_m-direct.displacement_m)/max(np.linalg.norm(direct.displacement_m),1e-30))
                after = iterative.energy_j[iterative.energy_time_s > 5e-6]
                energy_range = float(np.ptp(after)/abs(after.mean()))
                path = case/'comparison.npz'
                atomic_write(path,lambda stream:np.savez_compressed(stream,time_s=direct.time_s,
                    direct_m=direct.displacement_m,iterative_m=iterative.displacement_m,
                    energy_time_s=iterative.energy_time_s,energy_j=iterative.energy_j,iterations=iterative.iterations))
                passed = (max(stiffness_error,mass_error,diagonal_error,stiffness_diagonal_error,block_error) <= plan['operator_tolerance']
                    and waveform_error <= plan['waveform_tolerance'] and energy_range <= plan['energy_tolerance'])
                report['cases'].append({'geometry':recipe['name'],'order':order,'dofs':free.space.ndof,
                    'all_coefficient_columns_checked':free.space.ndof,'stiffness_relative_frobenius':stiffness_error,
                    'mass_relative_frobenius':mass_error,'mass_diagonal_relative_l2':diagonal_error,
                    'stiffness_diagonal_relative_l2':stiffness_diagonal_error,
                    'block_application_relative_l2':block_error,'waveform_relative_l2':waveform_error,
                    'postburst_relative_energy_range':energy_range,'maximum_relative_residual':iterative.maximum_relative_residual,
                    'total_cg_iterations':int(iterative.iterations.sum()),'maximum_cg_iterations':int(iterative.iterations.max()),
                    'stiffness_applications':free.stiffness.applications,'mass_applications':free.mass.applications,
                    'native_operator_type':type(free.stiffness.form.mat).__name__,
                    'interface':interface,'passed':bool(passed),'comparison_file':path.relative_to(output).as_posix(),
                    'comparison_sha256':sha256_file(path),'cache_hit':False,
                    'computation_seconds':time.perf_counter()-case_started})
                receipt = {'plan_fingerprint':plan['fingerprint'],'case':report['cases'][-1]}
                receipt['fingerprint'] = canonical_hash(receipt)
                write_json(case/'case-receipt.json',receipt)
        if reference_source_hashes() != plan['source_sha256'] or distribution_identity(guard) != plan['runtime']:
            raise ValueError('matrix-free source/runtime changed during verification')
        report.update({'completed':True,'passed':all(r['passed'] for r in report['cases']),
            'new_case_computations':sum(not r['cache_hit'] for r in report['cases']),
            'scope':'Discrete operator and 32-step direct/iterative equivalence on two tiny blocks; fresh cases include sparse reference assembly. No full-size/PZT/continuum acceptance.'})
    except Exception as exc:
        report['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        report['wall_seconds'] = time.perf_counter()-started
        report['resources'] = guard.report()
        report['fingerprint'] = canonical_hash(report)
        write_json(output/'native-report.json',report)
    return report


def verify_matrix_free(output, guard=None, *, orders=(1,2,3,4), cache_dir=None):
    if not orders or len(set(orders)) != len(orders) or any(type(p) is not int or not 1 <= p <= 4 for p in orders):
        raise ValueError('choose unique polynomial degrees in [1,4]')
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('matrix-free verification needs a new output directory')
    output.mkdir(parents=True,exist_ok=True)
    guard = guard or ResourceGuard(ResourceLimits(max_seconds=300,max_rss_mib=1024,
        max_system_memory_percent=85,min_available_mib=2048))
    plan = {'kind':'matrix_free_reference_check','orders':list(orders),'source_sha256':reference_source_hashes(),
        'runtime':distribution_identity(guard),'operator_tolerance':1e-10,'waveform_tolerance':1e-7,'energy_tolerance':1e-8,
        'iterative_settings':asdict(IterativeSettings(max_iterations=500)),
        'meshes':_reference_recipes(),
        'ready_for_training_data':False}
    plan['fingerprint'] = canonical_hash(plan)
    write_json(output/'matrixfree-plan.json',plan)
    def snapshot(stream):
        with zipfile.ZipFile(stream,'w',compression=zipfile.ZIP_DEFLATED) as archive:
            for name in plan['source_sha256']:
                archive.writestr('traxis/'+name,(Path(__file__).parent/name).read_bytes())
    atomic_write(output/'source-snapshot.zip',snapshot)
    result = {'completed':False,'passed':False,'ready_for_training_data':False,'training_updates':0,
        'source_snapshot_sha256':sha256_file(output/'source-snapshot.zip'),'plan_fingerprint':plan['fingerprint']}
    try:
        remaining = max(1.,guard.limits.max_seconds-(time.monotonic()-guard.started))
        command = [sys.executable,'-m','traxis.matrixfree_verification','--worker','--plan',str(output/'matrixfree-plan.json'),
            '--output',str(output/'native'),'--max-seconds',str(remaining),'--max-rss-mib',str(guard.limits.max_rss_mib)]
        if cache_dir:
            command += ['--cache-dir',str(Path(cache_dir).resolve())]
        result['worker_run'] = run_monitored(command,Path(__file__).resolve().parents[1],
            {**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'},guard,output/'worker.log')
        if result['worker_run']['exit_code'] not in (0,2):
            raise RuntimeError('matrix-free worker stopped; inspect native-report.json and worker.log')
        native = json.loads((output/'native/native-report.json').read_text())
        expected_cases = [(r['name'],p) for r in plan['meshes'] for p in plan['orders']]
        if (canonical_hash({k:v for k,v in native.items() if k!='fingerprint'}) != native['fingerprint']
                or not native['completed'] or native['plan_fingerprint'] != plan['fingerprint']
                or reference_source_hashes() != plan['source_sha256'] or distribution_identity(guard) != plan['runtime']
                or [(r['geometry'],r['order']) for r in native['cases']] != expected_cases
                or native['passed'] != all(r['passed'] for r in native['cases'])
                or native['new_case_computations'] != sum(not r['cache_hit'] for r in native['cases'])):
            raise ValueError('matrix-free worker/source receipt changed')
        for case in native['cases']:
            name = f"{case['geometry']}-p{case['order']}"
            _checked_case(case,output/'native',name,plan)
            receipt = json.loads((output/'native'/name/'case-receipt.json').read_text())
            if (receipt['case'] != case or receipt['plan_fingerprint'] != plan['fingerprint']
                    or canonical_hash({k:v for k,v in receipt.items() if k!='fingerprint'}) != receipt['fingerprint']):
                raise ValueError('matrix-free case receipt changed')
        result.update({'completed':True,'passed':native['passed'],'cases':native['cases'],
            'new_case_computations':native['new_case_computations'],
            'native_report_sha256':sha256_file(output/'native/native-report.json')})
    except Exception as exc:
        result['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        result['resources'] = guard.report()
        result['fingerprint'] = canonical_hash(result)
        write_json(output/'matrixfree-report.json',result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--plan',type=Path)
    parser.add_argument('--orders',type=int,nargs='+',default=[1,2,3,4])
    parser.add_argument('--worker',action='store_true')
    parser.add_argument('--cache-dir',type=Path)
    parser.add_argument('--max-seconds',type=float,default=300)
    parser.add_argument('--max-rss-mib',type=float,default=1024)
    args = parser.parse_args()
    guard = ResourceGuard(ResourceLimits(max_seconds=args.max_seconds,max_rss_mib=args.max_rss_mib,
        max_system_memory_percent=85,min_available_mib=2048))
    result = run_worker(args.plan,args.output,guard,args.cache_dir) if args.worker else verify_matrix_free(args.output,guard,orders=tuple(args.orders),cache_dir=args.cache_dir)
    print(json.dumps({k:result[k] for k in ('completed','passed','ready_for_training_data')}))
    raise SystemExit(0 if result['passed'] else 2)
