"""Measure enriched-space and positive-mass capacity on a retained intact mesh."""

import argparse
import json
import os
from pathlib import Path
import sys
import time
import zipfile

import numpy as np

from examples.matrixfree_step import INPUTS,_json,capacity_context,driver_hashes,load_mesh
from traxis.elmer_plate_refinement import reference_source_hashes
from traxis.elmer_reference import run_monitored
from traxis.enriched_tet import WEIGHTS,build_topology
from traxis.highorder import _native_space,highorder_dofs
from traxis.highorder_verification import distribution_identity
from traxis.resources import ResourceGuard,ResourceLimits
from traxis.storage import atomic_write,canonical_hash,sha256_file,write_json


def identity():
    return {**driver_hashes(),'enriched_capacity.py':sha256_file(Path(__file__))}


def worker(plan_path,output,guard):
    plan_path,output = Path(plan_path).resolve(),Path(output).resolve()
    plan = _json(plan_path)
    if (plan['kind']!='enriched_space_capacity' or plan['source_sha256']!=reference_source_hashes()
            or plan['driver_sha256']!=identity() or plan['runtime']!=distribution_identity(guard)):
        raise ValueError('enriched capacity source/runtime changed')
    root = Path(plan['capacity_directory'])
    if set(plan['input_sha256'])!=set(INPUTS) or any(sha256_file(root/n)!=h for n,h in plan['input_sha256'].items()):
        raise ValueError('retained capacity inputs changed')
    if not output.is_relative_to(plan_path.parent) or (output.exists() and any(output.iterdir())):
        raise ValueError('enriched output must be fresh and inside its plan directory')
    output.mkdir(parents=True,exist_ok=True)
    started = time.perf_counter()
    report = dict(completed=False,ready_for_training_data=False,training_updates=0,time_steps=0,
        stiffness_applications=0,cuda_allocation=False,sparse_assembly=False,plan_fingerprint=plan['fingerprint'])
    try:
        context = capacity_context(root)
        _,_,plate,material,settings = context
        geometry = load_mesh(root,context)
        topology = build_topology(geometry,max_vertices=settings.max_vertices,max_tetrahedra=settings.max_tetrahedra,
            max_dofs=settings.max_dofs,max_workspace_mib=settings.max_topology_mib)
        screen = highorder_dofs(geometry,1,max_vertices=settings.max_vertices,max_tetrahedra=settings.max_tetrahedra,
            max_topology_mib=settings.max_topology_mib)
        native,_,_ = _native_space(geometry,1,screen,guard)
        import ngsolve as ng
        scalar = ng.H1LumpingFESpace(native,order=2)
        space = ng.FESpace([scalar,scalar,scalar])
        if scalar.ndof!=len(topology.points_m) or space.ndof!=topology.screen['vector_dofs']:
            raise ArithmeticError('native enriched counts differ from topology')
        corners = geometry.points_m[geometry.tetrahedra]
        determinants = abs(np.linalg.det((corners[:,:3]-corners[:,3:4]).transpose(0,2,1)))
        mass = np.zeros(scalar.ndof)
        np.add.at(mass,topology.element_nodes.ravel(),
            (material.density_kg_m3*determinants[:,None]*WEIGHTS).ravel())
        if np.any(mass<=0) or not np.isfinite(mass).all():
            raise ArithmeticError('full-size enriched nodal mass is not finite positive')
        lengths = np.array([plate.width_m,plate.height_m,plate.thickness_m])
        total_mass = material.density_kg_m3*np.prod(lengths)
        moments = []
        for degree in range(3):
            for i in range(degree+1):
                for j in range(degree-i+1):
                    powers = np.array([i,j,degree-i-j])
                    measured = float(mass@np.prod(topology.points_m**powers,axis=1))
                    expected = float(total_mass*np.prod(lengths**powers/(powers+1)))
                    moments.append(dict(powers=powers.tolist(),measured=measured,expected=expected,
                        relative_error=abs(measured/expected-1)))
        if max(m['relative_error'] for m in moments)>1e-10:
            raise ArithmeticError('full-size enriched volume/moment check failed')
        atomic_write(output/'nodal-mass.npz',lambda stream:np.savez_compressed(stream,nodal_mass_kg=mass))
        if (plan['source_sha256']!=reference_source_hashes() or plan['driver_sha256']!=identity()
                or plan['runtime']!=distribution_identity(guard)):
            raise ValueError('enriched capacity source changed during measurement')
        report.update(completed=True,scalar_dofs=scalar.ndof,vector_dofs=space.ndof,screen=topology.screen,
            native_local_scalar_dofs=scalar.GetFE(ng.ElementId(ng.VOL,0)).ndof,
            positive_mass_minimum_kg=float(mass.min()),positive_mass_maximum_kg=float(mass.max()),
            total_mass_kg=float(mass.sum()),moments=moments,mass_sha256=sha256_file(output/'nodal-mass.npz'),
            mass_layout='enriched_topology_nodal',native_coefficient_permutation_verified=False,
            estimated_local_stiffness_mib=len(geometry.tetrahedra)*45**2*8/1024**2,
            scope='Actual native space counts and topology-order positive mass only; no stiffness, time propagation or native/topology permutation acceptance.')
    except Exception as error:
        report['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        report['wall_seconds'] = time.perf_counter()-started
        report['resources'] = guard.report()
        report['fingerprint'] = canonical_hash(report)
        write_json(output/'capacity-report.json',report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capacity',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--plan',type=Path)
    parser.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    parser.add_argument('--max-seconds',type=float,default=90)
    parser.add_argument('--max-rss-mib',type=float,default=1024)
    args = parser.parse_args()
    guard = ResourceGuard(ResourceLimits(max_seconds=args.max_seconds,max_rss_mib=args.max_rss_mib))
    if args.worker:
        return 0 if worker(args.plan,args.output,guard)['completed'] else 2
    if not args.capacity:
        parser.error('provide a retained intact capacity run')
    root,output = args.capacity.resolve(),args.output.resolve()
    capacity_context(root)
    if output.exists() and any(output.iterdir()):
        parser.error('enriched capacity needs a fresh output directory')
    output.mkdir(parents=True,exist_ok=True)
    result = dict(completed=False,ready_for_training_data=False,training_updates=0)
    try:
        plan = dict(kind='enriched_space_capacity',capacity_directory=str(root),source_sha256=reference_source_hashes(),
            driver_sha256=identity(),runtime=distribution_identity(guard),input_sha256={n:sha256_file(root/n) for n in INPUTS})
        plan['fingerprint'] = canonical_hash(plan)
        write_json(output/'enriched-capacity-plan.json',plan)
        def snapshot(stream):
            with zipfile.ZipFile(stream,'w',compression=zipfile.ZIP_DEFLATED) as archive:
                for name in plan['source_sha256']:
                    archive.writestr('traxis/'+name,(Path(__file__).resolve().parents[1]/'traxis'/name).read_bytes())
                for name in plan['driver_sha256']:
                    archive.writestr('examples/'+name,(Path(__file__).parent/name).read_bytes())
        atomic_write(output/'source-snapshot.zip',snapshot)
        result.update(plan_fingerprint=plan['fingerprint'],source_snapshot_sha256=sha256_file(output/'source-snapshot.zip'))
        temporary = output/'temporary'
        temporary.mkdir()
        remaining = max(1.,guard.limits.max_seconds-(time.monotonic()-guard.started))
        command = [sys.executable,'-m','examples.enriched_capacity','--worker','--plan',str(output/'enriched-capacity-plan.json'),
            '--output',str(output/'native'),'--max-seconds',str(remaining),'--max-rss-mib',str(guard.limits.max_rss_mib)]
        result['worker_run'] = run_monitored(command,Path(__file__).resolve().parents[1],
            {**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','TEMP':str(temporary),
                'TMP':str(temporary),'TMPDIR':str(temporary)},guard,output/'worker.log')
        if result['worker_run']['exit_code']!=0:
            raise RuntimeError('enriched capacity worker stopped; inspect log/report')
        native = _json(output/'native/capacity-report.json')
        if native['completed'] is not True or native['plan_fingerprint']!=plan['fingerprint']:
            raise ValueError('enriched capacity receipt changed')
        if sha256_file(output/'native/nodal-mass.npz')!=native['mass_sha256']:
            raise ValueError('enriched nodal mass changed')
        if (plan['source_sha256']!=reference_source_hashes() or plan['driver_sha256']!=identity()
                or plan['runtime']!=distribution_identity(guard)):
            raise ValueError('enriched capacity parent source/runtime changed')
        result.update(completed=True,vector_dofs=native['vector_dofs'],native_report_sha256=sha256_file(output/'native/capacity-report.json'))
    except Exception as error:
        result['error'] = f'{type(error).__name__}: {error}'
    finally:
        result['resources'] = guard.report()
        result['fingerprint'] = canonical_hash(result)
        write_json(output/'monitor-result.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('resources','fingerprint')}))
    return 0 if result['completed'] else 1


if __name__=='__main__':
    raise SystemExit(main())
