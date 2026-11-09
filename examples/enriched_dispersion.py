"""Bounded enriched-element Lamb-wave checks against Rayleigh-Lamb roots."""

import argparse
import json
import math
import os
from pathlib import Path
import sys
import time
import zipfile

import numpy as np
from scipy import sparse
from scipy.linalg import eigh

from examples.matrixfree_step import _json
from traxis.config import Material,Plate
from traxis.dispersion import wavenumber_rad_m
from traxis.elmer_plate_refinement import reference_source_hashes
from traxis.elmer_reference import run_monitored
from traxis.enriched_tet import assemble_enriched
from traxis.geometry import structured_mesh
from traxis.highorder_verification import distribution_identity
from traxis.resources import ResourceGuard,ResourceLimits
from traxis.storage import atomic_write,canonical_hash,sha256_file,write_json


def periodic_projection(points,plate,thickness_cells,wavenumber):
    """Tie only opposite faces; leave enriched interior coefficients free."""
    if (type(thickness_cells) is not int or not 1<=thickness_cells<=4
            or not math.isfinite(wavenumber) or wavenumber<=0):
        raise ValueError('invalid bounded enriched periodic cell')
    points = np.asarray(points)
    if points.ndim!=2 or points.shape[1]!=3 or not 15<=len(points)<=650 or not np.isfinite(points).all():
        raise ValueError('invalid enriched cell coordinates')
    # Vertices, midpoints, thirds and quarters lie on this integer lattice.
    scaled = points*np.array([12/plate.width_m,12/plate.height_m,12*thickness_cells/plate.thickness_m])
    integer = np.rint(scaled).astype(np.int64)
    if (not np.allclose(integer,scaled,atol=1e-9,rtol=0) or np.any(integer<0)
            or np.any(integer>[12,12,12*thickness_cells]) or len(np.unique(integer,axis=0))!=len(points)):
        raise ValueError('enriched nodes escaped the exact cell lattice')
    available = {tuple(row) for row in integer}
    for row in integer:
        for axis in (0,1):
            if row[axis] in (0,12):
                partner = row.copy()
                partner[axis] = 12-row[axis]
                if tuple(partner) not in available:
                    raise ValueError('enriched periodic face has no matching node')
    canonical = integer.copy()
    canonical[:,:2] %= 12
    keys,inverse = np.unique(canonical,axis=0,return_inverse=True)
    n = len(points)
    rows = np.r_[np.arange(n),np.arange(n)+2*n]
    columns = np.r_[inverse,inverse+len(keys)]
    phases = np.tile(np.exp(1j*wavenumber*plate.width_m*(integer[:,0]==12)),2)
    return sparse.coo_matrix((phases,(rows,columns)),shape=(3*n,2*len(keys))).tocsr()


def identity():
    return {'enriched_dispersion.py':sha256_file(Path(__file__))}


def worker(plan_path,output,guard):
    plan_path,output = Path(plan_path).resolve(),Path(output).resolve()
    plan = _json(plan_path)
    if (plan['kind']!='enriched_lamb_dispersion' or plan['finest_frequency_tolerance']!=.02
            or plan['source_sha256']!=reference_source_hashes() or plan['driver_sha256']!=identity()
            or plan['runtime']!=distribution_identity(guard)):
        raise ValueError('enriched dispersion source/runtime/criteria changed')
    if not output.is_relative_to(plan_path.parent) or (output.exists() and any(output.iterdir())):
        raise ValueError('enriched dispersion output must be fresh and within its plan')
    output.mkdir(parents=True,exist_ok=True)
    report = dict(completed=False,passed=False,ready_for_training_data=False,training_updates=0,
        plan_fingerprint=plan['fingerprint'],cases=[])
    started = time.perf_counter()
    try:
        material = Material()
        for thickness in (.0015,.003):
            for frequency in (100000.,200000.):
                for mode,index in (('A0',0),('S0',1)):
                    k = wavenumber_rad_m(frequency,thickness,mode,material)
                    for lateral in (4,8,16):
                        for depth in (1,2,4):
                            guard.check()
                            width = 2*math.pi/k/lateral
                            plate = Plate(width,width,thickness)
                            system = assemble_enriched(structured_mesh(plate,(1,1,depth)),material,guard)
                            projection = periodic_projection(system.topology.points_m,plate,depth,k)
                            kh = (projection.conj().T@system.stiffness@projection).toarray()
                            mh = (projection.conj().T@sparse.diags(system.mass_kg)@projection).toarray()
                            symmetry = max(np.linalg.norm(a-a.conj().T)/np.linalg.norm(a) for a in (kh,mh))
                            if symmetry>1e-12:
                                raise ArithmeticError('enriched Bloch operators are not Hermitian')
                            eigenvalues = eigh(kh,mh,eigvals_only=True)
                            if not np.isfinite(eigenvalues).all() or eigenvalues[0]<=0:
                                raise ArithmeticError('enriched Bloch eigenvalues are not positive')
                            numerical = float(np.sqrt(eigenvalues[index])/(2*math.pi))
                            name = f't{thickness:.4f}-f{frequency:.0f}-{mode}-xy{lateral}-z{depth}.npz'
                            atomic_write(output/name,lambda stream:np.savez_compressed(stream,
                                stiffness=kh,mass=mh,eigenvalues=eigenvalues,points_m=system.topology.points_m))
                            report['cases'].append(dict(thickness_m=thickness,target_frequency_hz=frequency,mode=mode,
                                cells_per_wavelength=lateral,thickness_cells=depth,wavenumber_rad_m=k,
                                full_dofs=system.ndof,condensed_dofs=projection.shape[1],hermitian_relative_error=float(symmetry),
                                numerical_frequency_hz=numerical,relative_frequency_error=abs(numerical/frequency-1),
                                comparison_file=name,comparison_sha256=sha256_file(output/name)))
                            write_json(output/'progress.json',report,overwrite=True)
        if (plan['source_sha256']!=reference_source_hashes() or plan['driver_sha256']!=identity()
                or plan['runtime']!=distribution_identity(guard)):
            raise ValueError('enriched dispersion source/runtime changed during measurement')
        finest = [r for r in report['cases'] if r['cells_per_wavelength']==16 and r['thickness_cells']==4]
        report.update(completed=True,passed=len(finest)==8 and all(r['relative_frequency_error']<.02 for r in finest),
            maximum_finest_relative_frequency_error=max(r['relative_frequency_error'] for r in finest),
            scope='Structured periodic x/z Lamb sector only; no finite-plate waveform, unstructured mesh or PZT acceptance.')
    except Exception as error:
        report['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        report['wall_seconds'] = time.perf_counter()-started
        report['resources'] = guard.report()
        report['fingerprint'] = canonical_hash(report)
        write_json(output/'dispersion-report.json',report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--plan',type=Path)
    parser.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    parser.add_argument('--max-seconds',type=float,default=90)
    parser.add_argument('--max-rss-mib',type=float,default=1024)
    args = parser.parse_args()
    guard = ResourceGuard(ResourceLimits(max_seconds=args.max_seconds,max_rss_mib=args.max_rss_mib,
        max_system_memory_percent=85,min_available_mib=2048))
    if args.worker:
        return 0 if worker(args.plan,args.output,guard)['passed'] else 2
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        parser.error('use a fresh enriched dispersion directory')
    output.mkdir(parents=True,exist_ok=True)
    result = dict(completed=False,passed=False,ready_for_training_data=False,training_updates=0)
    try:
        plan = dict(kind='enriched_lamb_dispersion',finest_frequency_tolerance=.02,
            source_sha256=reference_source_hashes(),driver_sha256=identity(),runtime=distribution_identity(guard))
        plan['fingerprint'] = canonical_hash(plan)
        write_json(output/'dispersion-plan.json',plan)
        def snapshot(stream):
            with zipfile.ZipFile(stream,'w',compression=zipfile.ZIP_DEFLATED) as archive:
                for name in plan['source_sha256']:
                    archive.writestr('traxis/'+name,(Path(__file__).resolve().parents[1]/'traxis'/name).read_bytes())
                archive.writestr('examples/enriched_dispersion.py',Path(__file__).read_bytes())
        atomic_write(output/'source-snapshot.zip',snapshot)
        result.update(plan_fingerprint=plan['fingerprint'],source_snapshot_sha256=sha256_file(output/'source-snapshot.zip'))
        temporary = output/'temporary'
        temporary.mkdir()
        remaining = max(1.,guard.limits.max_seconds-(time.monotonic()-guard.started))
        command = [sys.executable,'-m','examples.enriched_dispersion','--worker','--plan',str(output/'dispersion-plan.json'),
            '--output',str(output/'native'),'--max-seconds',str(remaining),'--max-rss-mib',str(guard.limits.max_rss_mib)]
        env = {**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','TEMP':str(temporary),'TMP':str(temporary),'TMPDIR':str(temporary)}
        result['worker_run'] = run_monitored(command,Path(__file__).resolve().parents[1],env,guard,output/'worker.log')
        if result['worker_run']['exit_code'] not in (0,2):
            raise RuntimeError('enriched dispersion worker stopped; inspect its log')
        native = _json(output/'native/dispersion-report.json')
        if native['completed'] is not True or native['plan_fingerprint']!=plan['fingerprint'] or len(native['cases'])!=72:
            raise ValueError('enriched dispersion receipt is incomplete or changed')
        for row in native['cases']:
            path = (output/'native'/row['comparison_file']).resolve()
            if not path.is_relative_to(output/'native') or sha256_file(path)!=row['comparison_sha256']:
                raise ValueError('enriched dispersion arrays changed')
        if (plan['source_sha256']!=reference_source_hashes() or plan['driver_sha256']!=identity()
                or plan['runtime']!=distribution_identity(guard)):
            raise ValueError('enriched dispersion parent source/runtime changed')
        result.update(completed=True,passed=native['passed'],maximum_finest_relative_frequency_error=native['maximum_finest_relative_frequency_error'],
            native_report_sha256=sha256_file(output/'native/dispersion-report.json'))
    except Exception as error:
        result['error'] = f'{type(error).__name__}: {error}'
    finally:
        result['resources'] = guard.report()
        result['fingerprint'] = canonical_hash(result)
        write_json(output/'monitor-result.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('resources','fingerprint')}))
    return 0 if result['passed'] else 2 if result['completed'] else 1


if __name__=='__main__':
    raise SystemExit(main())
