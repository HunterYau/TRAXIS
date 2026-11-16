"""Compare cached geometry assembly with the actual archived uncached source."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
from types import ModuleType
from unittest.mock import patch
import zipfile

import numpy as np

from examples.embedded_wave_evidence import check_snapshot
from examples.tensor_wave import PROJECT, checked_receipt, identities, read_json, sealed
from traxis.config import CircularRecess, Material, Plate
from traxis.embedded_recess import make_embedded_recess_mesh
from traxis.highorder_verification import distribution_identity
from traxis.mapped_tensor import build_system
import traxis.recess_hex as recess_hex
from traxis.resources import ResourceGuard, ResourceLimits
from traxis.storage import sha256_file, write_json


UNCACHED_SHA256 = 'de20a3763cbbec9c22cfc1aabfd3c5afc41af45da14054a42ad28fdbd7c6f9bf'


def array_hash(array):
    return hashlib.sha256(array.tobytes(order='A')).hexdigest()


def assemble(basis, order, defective, guard):
    started = time.perf_counter()
    with patch.object(recess_hex, 'lagrange_values', basis):
        mesh = make_embedded_recess_mesh(Plate(.08,.06,.003175), CircularRecess(.041,.029,.008,.0015),
            order, geometry_order=3, angular_cells=4, radial_cells=(1,2), layers=(2,2),
            outer_cell_size_m=.02, defective=defective, guard=guard)
        cpu = build_system(mesh, Material(), guard, max_bytes=512*1024**2, max_batch_vectors=1)
    arrays = {name:getattr(mesh,name) for name in ('points_m','element_nodes','geometry_points_m',
        'geometry_elements','cell_faces','cell_boxes')}
    arrays.update({name:getattr(cpu,name) for name in ('inverse_jacobians','volume_weights_m3',
        'mass_kg','stiffness_majorant_kg_s2')})
    return dict(arrays={name:array_hash(a) for name,a in arrays.items()},
        stable_dt_s=cpu.stable_dt_s, screen=cpu.screen, boundary_faces=list(mesh.boundary_faces),
        shared_faces=mesh.shared_faces, shared_coordinate_error_m=mesh.shared_coordinate_error_m,
        wall_seconds=time.perf_counter()-started)


def verify(reference, output, guard):
    reference,output = Path(reference).resolve(),Path(output).resolve()
    if not output.is_relative_to(PROJECT/'outputs'):
        raise ValueError('geometry comparison output must remain in project outputs')
    output.mkdir(parents=True,exist_ok=False)
    sources = identities()
    checker = sha256_file(__file__)
    runtime = distribution_identity(guard)
    report = dict(kind='geometry_basis_cache_equivalence', completed=False, passed=False,
        ready_for_training_data=False, training_updates=0, sources=sources, checker_sha256=checker,
        runtime=runtime, scope='Exact archived/current basis and small coupled-XYZ CPU assembly equivalence; no wave convergence or physical validation.')
    try:
        plan = checked_receipt(read_json(reference/'embedded-wave-plan.json'))
        snapshot = reference/'source-snapshot.zip'
        check_snapshot(plan,snapshot)
        with zipfile.ZipFile(snapshot) as archive:
            source = archive.read('traxis/recess_hex.py')
        if hashlib.sha256(source).hexdigest()!=UNCACHED_SHA256:
            raise ValueError('reference is not the reviewed original uncached geometry source')
        name = '_traxis_uncached_recess_geometry'
        old = ModuleType(name);sys.modules[name]=old
        try:
            exec(compile(source,str(snapshot)+'/traxis/recess_hex.py','exec'),old.__dict__)
        finally:
            sys.modules.pop(name,None)
        cached = recess_hex.lagrange_values
        basis_comparisons = 0
        rng = np.random.default_rng(20261115)
        for order in range(1,6):
            for points in (np.empty(0),np.linspace(-1.,1.,125),rng.uniform(-1.,1.,127),
                           np.linspace(-1.,1.,2051),np.linspace(-1.,1.,2051)[::3]):
                for a,b in zip(old.lagrange_values(order,points),cached(order,points)):
                    if a.dtype!=b.dtype or a.shape!=b.shape or array_hash(a)!=array_hash(b):
                        raise ArithmeticError('cached basis differs from actual archived polynomial evaluation')
                    basis_comparisons += 1
        cases = []
        for order in (3,4,5):
            for defective in (False,True):
                guard.check(force=True)
                previous = assemble(old.lagrange_values,order,defective,guard)
                current = assemble(cached,order,defective,guard)
                if {k:v for k,v in previous.items() if k!='wall_seconds'}!={k:v for k,v in current.items() if k!='wall_seconds'}:
                    raise ArithmeticError('cached mesh/operator assembly differs from archived geometry')
                cases.append(dict(order=order,defective=defective,passed=True,
                    vector_dofs=current['screen']['vector_dofs'],array_sha256=current['arrays'],
                    stable_dt_s=current['stable_dt_s'],uncached_wall_seconds=previous['wall_seconds'],
                    cached_wall_seconds=current['wall_seconds']))
        if identities()!=sources or sha256_file(__file__)!=checker or distribution_identity(guard)!=runtime:
            raise ValueError('geometry check source or runtime changed during verification')
        guard.check(force=True)
        report.update(completed=True,passed=True,basis_array_comparisons=basis_comparisons,cases=cases,
            reference_plan_fingerprint=plan['fingerprint'],reference_snapshot_sha256=sha256_file(snapshot),
            archived_geometry_sha256=UNCACHED_SHA256,
            current_geometry_sha256=sha256_file(PROJECT/'traxis/recess_hex.py'))
    except Exception as error:
        report['error']=f'{type(error).__name__}: {error}'
        raise
    finally:
        report['resources']=guard.report()
        write_json(output/'geometry-cache-check.json',sealed(report))
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    guard=ResourceGuard(ResourceLimits(max_seconds=90,max_rss_mib=1024,
        max_system_memory_percent=89.9,min_available_mib=768))
    report=verify(args.reference_run,args.output,guard)
    print(json.dumps({k:report[k] for k in ('completed','passed','basis_array_comparisons','fingerprint')}))
    return 0 if report['passed'] else 2


if __name__=='__main__':
    raise SystemExit(main())
