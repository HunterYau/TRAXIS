"""Bounded intact layered geometry, without sparse or native solver allocation."""

import argparse
import json
import math
import os
from pathlib import Path
import sys
import time
import zipfile

import numpy as np

from examples.matrixfree_step import _json
from traxis.config import Material,Plate
from traxis.elmer_plate_refinement import reference_source_hashes
from traxis.elmer_reference import run_monitored
from traxis.enriched_tet import build_topology
from traxis.geometry import TetMesh
from traxis.matrixfree import MatrixFreeSettings
from traxis.resources import ResourceGuard,ResourceLimits
from traxis.storage import atomic_write,canonical_hash,sha256_file,write_json


INPUTS = ('mesh-plan.json','monitor-result.json','geometry/mesh-report.json','geometry/mesh.npz')
PATTERN = np.array([[0,1,3,7],[0,3,2,7],[0,2,6,7],[0,6,4,7],[0,4,5,7],[0,5,1,7]])
OFFSETS = np.array([[0,0,0],[1,0,0],[0,1,0],[1,1,0],[0,0,1],[1,0,1],[0,1,1],[1,1,1]])


def dependencies():
    sources = reference_source_hashes()
    return {**{name:sources[name] for name in ('config.py','geometry.py','highorder.py','enriched_tet.py')},
        'examples/layered_mesh.py':sha256_file(Path(__file__))}


def checked_spec(values):
    if set(values)!={'plate','material','cells','max_geometry_mib','reference'}:
        raise ValueError('layered mesh specification has missing or unknown fields')
    plate,material,reference = Plate(**values['plate']),Material(**values['material']),MatrixFreeSettings(**values['reference'])
    cells = values['cells']
    if (not isinstance(cells,(tuple,list)) or len(cells)!=3 or any(type(n) is not int or not 1<=n<=2048 for n in cells)
            or not math.isfinite(values['max_geometry_mib']) or not 0<values['max_geometry_mib']<=2048
            or reference.order!=2):
        raise ValueError('layered geometry needs bounded integer cells and the degree-two enriched candidate')
    x,y,z = cells
    vertices = (x+1)*(y+1)*(z+1)
    tetrahedra = 6*x*y*z
    edges = x*(y+1)*(z+1)+y*(x+1)*(z+1)+z*(x+1)*(y+1)
    edges += x*y*(z+1)+y*z*(x+1)+z*x*(y+1)+x*y*z
    faces = 2*tetrahedra+2*(x*y+y*z+z*x)
    scalar = vertices+edges+faces+tetrahedra
    geometry_mib = (vertices*72+x*y*z*512+tetrahedra*64)/1024**2
    if (vertices>reference.max_vertices or tetrahedra>reference.max_tetrahedra
            or 3*scalar>reference.max_dofs or geometry_mib>values['max_geometry_mib']):
        raise MemoryError('layered geometry exceeds its pre-allocation count/storage limits')
    screen = dict(vertices=vertices,edges=edges,faces=faces,tetrahedra=tetrahedra,scalar_dofs=scalar,
        vector_dofs=3*scalar,geometry_workspace_screen_mib=geometry_mib,
        boundary_triangles=4*(x*y+y*z+z*x),cells=list(cells),spacing_m=(np.array([plate.width_m,plate.height_m,plate.thickness_m])/cells).tolist())
    return plate,material,reference,screen


def make_geometry(values,guard=None):
    plate,_,_,screen = checked_spec(values)
    nx,ny,nz = values['cells']
    if guard:
        guard.check(force=True)
    axes = [np.linspace(0,length,n+1) for length,n in zip((plate.width_m,plate.height_m,plate.thickness_m),(nx,ny,nz))]
    points = np.column_stack([a.ravel() for a in np.meshgrid(*axes,indexing='ij')])
    i,j,k = np.indices((nx,ny,nz)).reshape(3,-1)
    bases = i*(ny+1)*(nz+1)+j*(nz+1)+k
    offsets = OFFSETS@np.array([(ny+1)*(nz+1),nz+1,1])
    tetrahedra = (bases[:,None]+offsets)[...,PATTERN].reshape(-1,4)
    if points.shape!=(screen['vertices'],3) or tetrahedra.shape!=(screen['tetrahedra'],4):
        raise ArithmeticError('layered geometry counts differ from preflight')
    volume = 0.
    for start in range(0,len(tetrahedra),16384):
        if guard:
            guard.check()
        corners = points[tetrahedra[start:start+16384]]
        determinants = abs(np.linalg.det(corners[:,1:]-corners[:,:1]))
        if not np.isfinite(determinants).all() or np.any(determinants<=0):
            raise ArithmeticError('layered geometry contains an invalid tetrahedron')
        volume += float(determinants.sum()/6)
    expected = plate.width_m*plate.height_m*plate.thickness_m
    if abs(volume/expected-1)>1e-11:
        raise ArithmeticError('layered geometry lost intact box volume')
    return TetMesh(points,tetrahedra,plate,0.,0.),screen,volume


def worker(plan_path,output,guard):
    plan_path,output = Path(plan_path).resolve(),Path(output).resolve()
    plan = _json(plan_path)
    if plan['kind']!='layered_intact_geometry' or plan['geometry_dependencies']!=dependencies():
        raise ValueError('layered geometry source/plan changed')
    if not output.is_relative_to(plan_path.parent) or (output.exists() and any(output.iterdir())):
        raise ValueError('layered geometry output must be fresh and inside its plan')
    output.mkdir(parents=True,exist_ok=True)
    report = dict(completed=False,ready_for_training_data=False,training_updates=0,time_steps=0,
        native_space_constructed=False,cuda_allocation=False,sparse_assembly=False,plan_fingerprint=plan['fingerprint'])
    try:
        mesh,screen,volume = make_geometry(plan['specification'],guard)
        _,_,reference,_ = checked_spec(plan['specification'])
        topology = build_topology(mesh,max_vertices=reference.max_vertices,max_tetrahedra=reference.max_tetrahedra,
            max_dofs=reference.max_dofs,max_workspace_mib=reference.max_topology_mib)
        if any(topology.screen[k]!=screen[k] for k in ('vertices','edges','faces','tetrahedra','scalar_dofs','vector_dofs')):
            raise ArithmeticError('layered closed-form counts disagree with actual enriched topology')
        faces = np.sort(mesh.tetrahedra[:,((0,1,2),(0,1,3),(0,2,3),(1,2,3))],axis=2).reshape(-1,3)
        unique,counts = np.unique(faces,axis=0,return_counts=True)
        if counts.max()>2 or np.count_nonzero(counts==1)!=screen['boundary_triangles']:
            raise ArithmeticError('layered tetrahedra have an invalid shared-face topology')
        boundary = mesh.points_m[unique[counts==1]]
        bounds = np.array([mesh.plate.width_m,mesh.plate.height_m,mesh.plate.thickness_m])
        if not np.all(np.any(np.all(boundary==0.,axis=1)|np.all(boundary==bounds,axis=1),axis=1)):
            raise ArithmeticError('layered boundary face is not on the intact box')
        atomic_write(output/'mesh.npz',lambda stream:np.savez_compressed(stream,points_m=mesh.points_m,tetrahedra=mesh.tetrahedra))
        if plan['geometry_dependencies']!=dependencies():
            raise ValueError('layered geometry source changed during construction')
        report.update(completed=True,screen=screen,topology_screen=topology.screen,volume_m3=volume,
            mesh_sha256=sha256_file(output/'mesh.npz'),scope='Intact layered geometry and exact topology only; no native/GPU stiffness or wave propagation.')
    except Exception as error:
        report['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        report['resources'] = guard.report()
        report['fingerprint'] = canonical_hash(report)
        write_json(output/'mesh-report.json',report)
    return report


def bundle_context(root):
    root = Path(root).resolve()
    plan,parent,report = (_json(root/p) for p in INPUTS[:3])
    plate,material,reference,screen = checked_spec(plan['specification'])
    if (plan['kind']!='layered_intact_geometry' or plan['geometry_dependencies']!=dependencies()
            or parent['completed'] is not True or parent['worker_run']['exit_code']!=0
            or report['completed'] is not True or report['native_space_constructed'] is not False
            or report['cuda_allocation'] is not False or report['time_steps']!=0
            or report['sparse_assembly'] is not False or report['training_updates']!=0
            or report['plan_fingerprint']!=plan['fingerprint'] or parent['plan_fingerprint']!=plan['fingerprint']
            or sha256_file(root/INPUTS[2])!=parent['geometry_report_sha256']
            or sha256_file(root/INPUTS[3])!=parent['mesh_sha256'] or report['mesh_sha256']!=parent['mesh_sha256']
            or report['screen']!=screen):
        raise ValueError('layered mesh bundle is incomplete or changed')
    return plan,report,plate,material,reference


def load_bundle_mesh(root,context,guard=None):
    plan,report,_,_,_ = context
    path = Path(root)/INPUTS[3]
    screen = report['screen']
    limit = screen['vertices']*24+screen['tetrahedra']*32+4096
    if path.stat().st_size>limit:
        raise MemoryError('layered mesh archive exceeds its declared size')
    with zipfile.ZipFile(path) as archive:
        rows = archive.infolist()
        if len(rows)!=2 or {r.filename for r in rows}!={'points_m.npy','tetrahedra.npy'} or sum(r.file_size for r in rows)>limit:
            raise ValueError('layered mesh archive has invalid entries or expansion size')
    with np.load(path,allow_pickle=False) as arrays:
        points,tets = arrays['points_m'],arrays['tetrahedra']
    expected,_,_ = make_geometry(plan['specification'],guard)
    if points.dtype!=np.float64 or tets.dtype!=np.int64 or not np.array_equal(points,expected.points_m) or not np.array_equal(tets,expected.tetrahedra):
        raise ValueError('saved mesh differs from the declared exact layered construction')
    return TetMesh(points,tets,expected.plate,0.,0.)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--plan',type=Path)
    parser.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    parser.add_argument('--max-seconds',type=float,default=120)
    parser.add_argument('--max-rss-mib',type=float,default=1024)
    args = parser.parse_args()
    guard = ResourceGuard(ResourceLimits(max_seconds=args.max_seconds,max_rss_mib=args.max_rss_mib,
        max_system_memory_percent=85,min_available_mib=2048))
    if args.worker:
        return 0 if worker(args.plan,args.output,guard)['completed'] else 2
    if not args.spec or args.spec.stat().st_size>1024**2:
        parser.error('provide a bounded layered mesh specification')
    specification = json.loads(args.spec.read_text())
    checked_spec(specification)
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        parser.error('layered geometry needs a fresh output directory')
    output.mkdir(parents=True,exist_ok=True)
    result = dict(completed=False,ready_for_training_data=False,training_updates=0)
    try:
        plan = dict(kind='layered_intact_geometry',specification=specification,geometry_dependencies=dependencies(),
            source_sha256=reference_source_hashes(),numpy_version=np.__version__)
        plan['fingerprint'] = canonical_hash(plan)
        write_json(output/'mesh-plan.json',plan)
        def snapshot(stream):
            with zipfile.ZipFile(stream,'w',compression=zipfile.ZIP_DEFLATED) as archive:
                for name in plan['source_sha256']:
                    archive.writestr('traxis/'+name,(Path(__file__).resolve().parents[1]/'traxis'/name).read_bytes())
                archive.writestr('examples/layered_mesh.py',Path(__file__).read_bytes())
        atomic_write(output/'source-snapshot.zip',snapshot)
        result.update(plan_fingerprint=plan['fingerprint'],source_snapshot_sha256=sha256_file(output/'source-snapshot.zip'))
        temporary = output/'temporary'
        temporary.mkdir()
        remaining = max(1.,guard.limits.max_seconds-(time.monotonic()-guard.started))
        command = [sys.executable,'-m','examples.layered_mesh','--worker','--plan',str(output/'mesh-plan.json'),
            '--output',str(output/'geometry'),'--max-seconds',str(remaining),'--max-rss-mib',str(guard.limits.max_rss_mib)]
        env = {**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','TEMP':str(temporary),'TMP':str(temporary),'TMPDIR':str(temporary)}
        result['worker_run'] = run_monitored(command,Path(__file__).resolve().parents[1],env,guard,output/'worker.log')
        if result['worker_run']['exit_code']!=0:
            raise RuntimeError('layered geometry worker stopped; inspect its report/log')
        report = _json(output/'geometry/mesh-report.json')
        if report['completed'] is not True or report['plan_fingerprint']!=plan['fingerprint'] or dependencies()!=plan['geometry_dependencies']:
            raise ValueError('layered geometry report/source changed')
        if report['mesh_sha256']!=sha256_file(output/'geometry/mesh.npz'):
            raise ValueError('layered geometry archive changed')
        result.update(completed=True,geometry_report_sha256=sha256_file(output/'geometry/mesh-report.json'),mesh_sha256=sha256_file(output/'geometry/mesh.npz'))
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
