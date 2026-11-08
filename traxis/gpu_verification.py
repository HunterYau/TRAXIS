"""Check optional CUDA element operators against the native sparse reference."""

import argparse
from dataclasses import asdict
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import sys
import time
import zipfile

import numpy as np

from traxis.config import Material
from traxis.elmer_plate_refinement import reference_source_hashes
from traxis.elmer_reference import run_monitored
from traxis.fem import tone_burst
from traxis.gpu_elements import build_gpu_elements, GPUElementOperator
from traxis.highorder import assemble_highorder,highorder_forms
from traxis.highorder_patches import highorder_surface_patch
from traxis.highorder_verification import distribution_identity,reference_mesh
from traxis.implicit_fem import implicit_simulate
from traxis.iterative_fem import IterativeSettings,iterative_simulate
from traxis.matrixfree import MatrixFreeSettings
from traxis.matrixfree_verification import _column_error,_reference_recipes
from traxis.resources import ResourceGuard,ResourceLimits
from traxis.storage import atomic_write,canonical_hash,sha256_file,write_json


def gpu_package_identity(guard=None):
    result = {}
    for name in ('cupy-cuda12x','cuda-pathfinder','nvidia-cuda-runtime-cu12','nvidia-cuda-nvrtc-cu12','nvidia-cuda-cccl-cu12'):
        distribution = metadata.distribution(name)
        files = {}
        for entry in distribution.files or ():
            if entry.suffix.lower() in ('.dll','.pyd','.so','.dylib'):
                if guard:
                    guard.check()
                files[str(entry).replace('\\','/')] = sha256_file(distribution.locate_file(entry))
        result[name] = {'version':distribution.version,'native_sha256':files,
            'record_sha256':hashlib.sha256((distribution.read_text('RECORD') or '').encode()).hexdigest()}
    if not result['cupy-cuda12x']['native_sha256'] or not result['nvidia-cuda-nvrtc-cu12']['native_sha256']:
        raise ValueError('GPU package identity lacks CuPy or NVRTC native binaries')
    return result


def schwarz_reference(system, material, scale, guard):
    """Tiny CPU reference using independent local dense solves and scatter."""
    import ngsolve as ng
    n = system.space.ndof
    if n > 2000:
        raise MemoryError('explicit Schwarz verification is restricted to 2000 coefficients')
    stiffness,_ = highorder_forms(system.space,material,matrix_free=True)
    u,v = system.scalar_space.TnT()
    mass = ng.BilinearForm(system.scalar_space,nonassemble=True)
    mass += material.density_kg_m3*u*v*ng.dx(bonus_intorder=2)
    result,coverage = np.zeros((n,n)),np.zeros(n)
    for element in system.mesh.Elements(ng.VOL):
        guard.check()
        eid = ng.ElementId(ng.VOL,element.nr)
        indices = np.asarray(system.space.GetDofNrs(eid),dtype=int)
        transformation = system.mesh.GetTrafo(eid)
        k = sum(np.asarray(i.CalcElementMatrix(system.space.GetFE(eid),transformation)) for i in stiffness.integrators)
        m = sum(np.asarray(i.CalcElementMatrix(system.scalar_space.GetFE(eid),transformation)) for i in mass.integrators)
        local = scale*k+np.kron(np.eye(3),m)
        result[np.ix_(indices,indices)] += np.linalg.solve(local,np.eye(len(local)))
        coverage[indices] += 1
    if np.any(coverage <= 0):
        raise ArithmeticError('Schwarz reference contains an uncovered coefficient')
    result /= coverage[:,None]*coverage[None,:]
    return result


def worker(plan_path, output, guard):
    plan_path,output = Path(plan_path).resolve(),Path(output).resolve()
    if plan_path.stat().st_size > 1024**2:
        raise MemoryError('GPU reference plan exceeds 1 MiB')
    plan = json.loads(plan_path.read_text())
    if (plan.get('kind') != 'gpu_element_reference' or plan.get('meshes') != _reference_recipes()
            or not plan.get('orders') or len(set(plan['orders'])) != len(plan['orders'])
            or any(type(p) is not int or not 1 <= p <= 4 for p in plan['orders'])
            or plan.get('operator_tolerance') != 1e-10 or plan.get('waveform_tolerance') != 1e-7
            or plan.get('energy_tolerance') != 1e-8
            or plan.get('preconditioner') not in ('diagonal','element_schwarz')
            or plan.get('iterative_settings') != asdict(IterativeSettings(max_iterations=500))
            or canonical_hash({k:v for k,v in plan.items() if k!='fingerprint'}) != plan['fingerprint']
            or plan['source_sha256'] != reference_source_hashes()
            or plan['reference_runtime'] != distribution_identity(guard)
            or plan['gpu_packages'] != gpu_package_identity(guard)):
        raise ValueError('GPU reference cases, source, runtime or criteria changed')
    if not output.is_relative_to(plan_path.parent) or (output.exists() and any(output.iterdir())):
        raise ValueError('GPU reference output must be fresh and inside its plan directory')
    output.mkdir(parents=True,exist_ok=True)
    report = {'completed':False,'passed':False,'ready_for_training_data':False,'training_updates':0,
        'plan_fingerprint':plan['fingerprint'],'cases':[]}
    started = time.perf_counter()
    try:
        for recipe in plan['meshes']:
            mesh,material = reference_mesh(recipe),Material(**recipe['material'])
            for degree in plan['orders']:
                case_started = time.perf_counter()
                case = output/f"{recipe['name']}-p{degree}"
                case.mkdir()
                sparse = assemble_highorder(mesh,material,degree,guard)
                dt = 2.5e-7
                use_schwarz = plan['preconditioner'] == 'element_schwarz'
                gpu = build_gpu_elements(mesh,material,MatrixFreeSettings(order=degree),guard=guard,
                    preconditioner_scale=.25*dt*dt if use_schwarz else None)
                k_error,m_error = _column_error(gpu.stiffness,sparse.stiffness,guard),_column_error(gpu.mass,sparse.mass,guard)
                kdiag = float(np.linalg.norm(gpu.stiffness_diagonal-sparse.stiffness.diagonal())/np.linalg.norm(sparse.stiffness.diagonal()))
                mdiag = float(np.linalg.norm(gpu.mass_diagonal-sparse.mass.diagonal())/np.linalg.norm(sparse.mass.diagonal()))
                effective = gpu.mass+(.25*dt*dt)*gpu.stiffness
                if not isinstance(effective,GPUElementOperator) or effective.storage is not gpu.stiffness.storage:
                    raise ArithmeticError('GPU Newmark combination failed to fuse element application')
                a_error = _column_error(effective,sparse.mass+(.25*dt*dt)*sparse.stiffness,guard)
                preconditioner_metrics = {}
                if use_schwarz:
                    reference = schwarz_reference(gpu,material,.25*dt*dt,guard)
                    columns = []
                    for column in range(gpu.space.ndof):
                        guard.check()
                        unit = np.zeros(gpu.space.ndof)
                        unit[column] = 1.
                        columns.append(gpu.preconditioner@unit)
                    represented = np.column_stack(columns)
                    b_error = float(np.linalg.norm(represented-reference)/np.linalg.norm(reference))
                    b_symmetry = float(np.linalg.norm(represented-represented.T)/np.linalg.norm(represented))
                    factor = np.linalg.cholesky((represented+represented.T)*.5)
                    if max(b_error,b_symmetry) > plan['operator_tolerance']:
                        raise ArithmeticError('Schwarz reference or symmetry check failed')
                    preconditioner_metrics = {'preconditioner_relative_frobenius':b_error,
                        'preconditioner_relative_skew':b_symmetry,
                        'preconditioner_minimum_cholesky_diagonal':float(factor.diagonal().min()),
                        'preconditioner_columns_checked':gpu.space.ndof}
                rng = np.random.default_rng(2026+degree)
                vectors = rng.normal(size=(gpu.space.ndof,2))
                first = gpu.mass@vectors[:,0]
                retained = first.copy()
                gpu.stiffness@vectors[:,1]
                if not np.array_equal(first,retained):
                    raise ArithmeticError('GPU application overwrote a returned host vector')
                block_error = float(np.linalg.norm(gpu.mass@vectors-sparse.mass@vectors)/np.linalg.norm(sparse.mass@vectors))
                if recipe['name'] == 'rectangular':
                    load,_ = highorder_surface_patch(gpu,mesh.plate,(.01,.008),.003,guard=guard)
                    probe,_ = highorder_surface_patch(gpu,mesh.plate,(.03,.022),.003,guard=guard)
                    probes = probe[None,:]
                else:
                    load = vectors[:,0]/np.linalg.norm(vectors[:,0])
                    probes = (vectors[:,1]/np.linalg.norm(vectors[:,1]))[None,:]
                force = lambda t:load*tone_burst(float(t),200000.,1)
                direct = implicit_simulate(sparse.stiffness,sparse.mass,32,dt,probes,force=force,guard=guard)
                result = iterative_simulate(gpu.stiffness,gpu.mass,gpu.mass_diagonal,32,dt,probes,
                    force=force,stiffness_diagonal=gpu.stiffness_diagonal,
                    settings=IterativeSettings(**plan['iterative_settings']),guard=guard,
                    effective_preconditioner=gpu.preconditioner)
                if not np.array_equal(result.time_s,direct.time_s):
                    raise ArithmeticError('GPU transient time grid changed')
                waveform = float(np.linalg.norm(result.displacement_m-direct.displacement_m)/np.linalg.norm(direct.displacement_m))
                after = result.energy_j[result.energy_time_s > 5e-6]
                energy = float(np.ptp(after)/abs(after.mean()))
                arrays = case/'comparison.npz'
                atomic_write(arrays,lambda stream:np.savez_compressed(stream,time_s=result.time_s,
                    direct_m=direct.displacement_m,gpu_m=result.displacement_m,energy_time_s=result.energy_time_s,
                    energy_j=result.energy_j,iterations=result.iterations))
                passed = (max(k_error,m_error,a_error,kdiag,mdiag,block_error) <= plan['operator_tolerance']
                    and waveform <= plan['waveform_tolerance'] and energy <= plan['energy_tolerance'])
                row = {'geometry':recipe['name'],'order':degree,'dofs':gpu.space.ndof,
                    'all_coefficient_columns_checked':gpu.space.ndof,'stiffness_relative_frobenius':k_error,
                    'mass_relative_frobenius':m_error,'effective_relative_frobenius':a_error,
                    'stiffness_diagonal_relative_l2':kdiag,'mass_diagonal_relative_l2':mdiag,
                    'block_application_relative_l2':block_error,'waveform_relative_l2':waveform,
                    'postburst_relative_energy_range':energy,'maximum_relative_residual':result.maximum_relative_residual,
                    'maximum_cg_iterations':int(result.iterations.max()),'total_cg_iterations':int(result.iterations.sum()),
                    'preconditioner':plan['preconditioner'],**preconditioner_metrics,
                    'fused_effective_operator':True,'device':gpu.device,'screen':gpu.screen,'passed':bool(passed),
                    'comparison_file':arrays.relative_to(output).as_posix(),'comparison_sha256':sha256_file(arrays),
                    'wall_seconds':time.perf_counter()-case_started}
                report['cases'].append(row)
                write_json(case/'case-receipt.json',{'plan_fingerprint':plan['fingerprint'],'case':row})
        if (reference_source_hashes() != plan['source_sha256'] or gpu_package_identity(guard) != plan['gpu_packages']
                or distribution_identity(guard) != plan['reference_runtime']):
            raise ValueError('GPU reference source/runtime changed during verification')
        report.update(completed=True,passed=all(r['passed'] for r in report['cases']),
            scope='Every small coefficient column and 32-step sparse/float64-CUDA equivalence. No full-size/PZT/continuum acceptance.')
    except Exception as exc:
        report['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        report['wall_seconds'] = time.perf_counter()-started
        report['resources'] = guard.report()
        report['fingerprint'] = canonical_hash(report)
        write_json(output/'native-report.json',report)
    return report


def verify_gpu(output, guard=None, *, orders=(1,2,3,4), preconditioner='diagonal'):
    if not orders or len(set(orders)) != len(orders) or any(type(p) is not int or not 1 <= p <= 4 for p in orders):
        raise ValueError('choose unique GPU reference degrees in [1,4]')
    if preconditioner not in ('diagonal','element_schwarz'):
        raise ValueError('choose diagonal or element_schwarz preconditioning')
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('GPU verification needs a fresh output directory')
    output.mkdir(parents=True,exist_ok=True)
    guard = guard or ResourceGuard(ResourceLimits(max_seconds=180,max_rss_mib=1024,
        max_system_memory_percent=85,min_available_mib=2048))
    result = {'completed':False,'passed':False,'ready_for_training_data':False,'training_updates':0}
    try:
        plan = {'kind':'gpu_element_reference','meshes':_reference_recipes(),'orders':list(orders),
            'source_sha256':reference_source_hashes(),'reference_runtime':distribution_identity(guard),
            'gpu_packages':gpu_package_identity(guard),'operator_tolerance':1e-10,'waveform_tolerance':1e-7,
            'energy_tolerance':1e-8,'iterative_settings':asdict(IterativeSettings(max_iterations=500)),
            'preconditioner':preconditioner}
        plan['fingerprint'] = canonical_hash(plan)
        write_json(output/'gpu-plan.json',plan)
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
        command = [sys.executable,'-m','traxis.gpu_verification','--worker','--plan',str(output/'gpu-plan.json'),
            '--output',str(output/'native'),'--max-seconds',str(remaining),'--max-rss-mib',str(guard.limits.max_rss_mib)]
        env = {**os.environ,'CUPY_CACHE_DIR':str(cache),'CUPY_CACHE_SAVE_CUDA_SOURCE':'1',
            'TEMP':str(temporary),'TMP':str(temporary),'TMPDIR':str(temporary),'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'}
        result['worker_run'] = run_monitored(command,Path(__file__).resolve().parents[1],env,guard,output/'worker.log')
        if result['worker_run']['exit_code'] not in (0,2):
            raise RuntimeError('GPU reference worker stopped; inspect its report and log')
        native = json.loads((output/'native/native-report.json').read_text())
        expected = [(r['name'],p) for r in plan['meshes'] for p in plan['orders']]
        if (canonical_hash({k:v for k,v in native.items() if k!='fingerprint'}) != native['fingerprint']
                or native['completed'] is not True or native['plan_fingerprint'] != plan['fingerprint']
                or [(r['geometry'],r['order']) for r in native['cases']] != expected
                or native['passed'] != all(r['passed'] for r in native['cases'])
                or reference_source_hashes() != plan['source_sha256'] or gpu_package_identity(guard) != plan['gpu_packages']):
            raise ValueError('GPU reference receipt/source/runtime changed')
        for row in native['cases']:
            path = (output/'native'/row['comparison_file']).resolve()
            if not path.is_relative_to(output/'native') or sha256_file(path) != row['comparison_sha256']:
                raise ValueError('GPU reference arrays changed')
        result.update(completed=True,passed=native['passed'],cases=native['cases'],
            native_report_sha256=sha256_file(output/'native/native-report.json'),
            kernel_cache_sha256={p.relative_to(cache).as_posix():sha256_file(p) for p in sorted(cache.rglob('*')) if p.is_file()})
    except Exception as exc:
        result['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        result['resources'] = guard.report()
        result['fingerprint'] = canonical_hash(result)
        write_json(output/'gpu-report.json',result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--plan',type=Path)
    parser.add_argument('--orders',type=int,nargs='+',default=[1,2,3,4])
    parser.add_argument('--preconditioner',choices=('diagonal','element_schwarz'),default='diagonal')
    parser.add_argument('--worker',action='store_true')
    parser.add_argument('--max-seconds',type=float,default=180)
    parser.add_argument('--max-rss-mib',type=float,default=1024)
    args = parser.parse_args()
    guard = ResourceGuard(ResourceLimits(max_seconds=args.max_seconds,max_rss_mib=args.max_rss_mib,
        max_system_memory_percent=85,min_available_mib=2048))
    result = worker(args.plan,args.output,guard) if args.worker else verify_gpu(args.output,guard,
        orders=tuple(args.orders),preconditioner=args.preconditioner)
    print(json.dumps({k:result[k] for k in ('completed','passed','ready_for_training_data')}))
    raise SystemExit(0 if result['passed'] else 2)
