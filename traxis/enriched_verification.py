"""Isolated native and temporal checks for the 15-node enriched candidate."""

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
from traxis.enriched_tet import NODES,WEIGHTS,basis,assemble_enriched
from traxis.fem import simulate
from traxis.highorder import assemble_highorder
from traxis.highorder_verification import distribution_identity,reference_mesh
from traxis.matrixfree_verification import _reference_recipes
from traxis.resources import ResourceGuard,ResourceLimits
from traxis.storage import atomic_write,canonical_hash,sha256_file,write_json


def recipes():
    return [{**r,'cells':[1,1,1]} for r in _reference_recipes()]


def embedding(native_mesh,scalar,topology,reference_nodes):
    """Represent the enriched nodal functions in an independent native basis."""
    if scalar.ndof>1000 or len(topology.points_m)>500:
        raise MemoryError('explicit basis embedding exceeds its small-reference bound')
    import ngsolve as ng
    original = topology.geometry
    owners = {tuple(sorted(tet)):i for i,tet in enumerate(original.tetrahedra)}
    transform = np.zeros((scalar.ndof,len(topology.points_m)))
    seen = np.zeros(scalar.ndof,dtype=bool)
    maximum_overlap = 0.
    for element in native_mesh.Elements(ng.VOL):
        vertices = [v.nr for v in element.vertices]
        owner = owners[tuple(sorted(vertices))]
        eid = ng.ElementId(ng.VOL,element.nr)
        fe = scalar.GetFE(eid)
        native_values = np.array([fe.CalcShape(*p[:3]) for p in reference_nodes])
        physical = reference_nodes@original.points_m[vertices]
        corners = original.points_m[original.tetrahedra[owner]]
        xyz = np.linalg.solve((corners[:3]-corners[3]).T,(physical-corners[3]).T).T
        barycentric = np.column_stack((xyz,1-xyz.sum(axis=1)))
        local = np.linalg.solve(native_values,basis(barycentric))
        rows = np.array(scalar.GetDofNrs(eid),dtype=int)
        if native_values.shape!=(len(rows),len(rows)) or np.any(rows<0):
            raise ValueError('native embedding needs complete unconstrained local functions')
        for row,coefficients in zip(rows,local):
            represented = np.zeros(len(topology.points_m))
            represented[topology.element_nodes[owner]] = coefficients
            if seen[row]:
                maximum_overlap = max(maximum_overlap,float(np.max(abs(transform[row]-represented))))
            else:
                transform[row] = represented
                seen[row] = True
    if not seen.all() or maximum_overlap>1e-10:
        raise ArithmeticError('native basis embedding violates shared coefficient continuity')
    return np.kron(np.eye(3),transform),maximum_overlap


def native_enriched(mesh,material,bonus):
    import ngsolve as ng
    scalar = ng.H1LumpingFESpace(mesh,order=2)
    space = ng.FESpace([scalar,scalar,scalar])
    u,v = space.TnT()
    du,dv = [[ng.grad(field[i]) for i in range(3)] for field in (u,v)]
    lam,mu = material.lame
    elastic = mu/2*sum((du[i][j]+du[j][i])*(dv[i][j]+dv[j][i]) for i in range(3) for j in range(3))
    elastic += lam*sum(du[i][i] for i in range(3))*sum(dv[i][i] for i in range(3))
    k,m,q = (ng.BilinearForm(space,symmetric=False) for _ in range(3))
    k += elastic*ng.dx(bonus_intorder=bonus)
    expression = material.density_kg_m3*sum(u[i]*v[i] for i in range(3))
    m += expression*ng.dx(bonus_intorder=bonus)
    rule = ng.IntegrationRule(points=NODES[:,:3].tolist(),weights=WEIGHTS.tolist())
    q += expression*ng.dx(intrules={ng.ET.TET:rule})
    from scipy.sparse import csr_matrix
    def copy(form):
        form.Assemble()
        values,indices,pointers = form.mat.CSR()
        return csr_matrix((np.array(values),np.array(indices),np.array(pointers)),shape=(space.ndof,space.ndof))
    return scalar,space,copy(k),copy(m),copy(q)


def worker(plan_path,output,guard):
    plan_path,output = Path(plan_path).resolve(),Path(output).resolve()
    if plan_path.stat().st_size>1024**2:
        raise MemoryError('enriched reference plan exceeds 1 MiB')
    plan = json.loads(plan_path.read_text())
    if (plan.get('kind')!='enriched_tetrahedron_reference' or plan.get('meshes')!=recipes()
            or plan.get('operator_tolerance')!=1e-10 or plan.get('quadrature_bonus')!=6
            or plan.get('modified_energy_tolerance')!=1e-9
            or plan.get('temporal_steps')!=[64,128,256] or plan.get('diagnostic_steps')!=32
            or canonical_hash({k:v for k,v in plan.items() if k!='fingerprint'})!=plan['fingerprint']
            or plan['source_sha256']!=reference_source_hashes() or plan['runtime']!=distribution_identity(guard)):
        raise ValueError('enriched reference source, runtime, cases or criteria changed')
    if not output.is_relative_to(plan_path.parent) or (output.exists() and any(output.iterdir())):
        raise ValueError('enriched reference output must be new and within its plan directory')
    output.mkdir(parents=True,exist_ok=True)
    report = dict(completed=False,passed=False,ready_for_training_data=False,training_updates=0,
        plan_fingerprint=plan['fingerprint'],cases=[])
    started = time.perf_counter()
    try:
        import ngsolve as ng
        p4_nodes = np.array([(i,j,k,4-i-j-k) for i in range(5) for j in range(5-i) for k in range(5-i-j)])/4
        for recipe in plan['meshes']:
            guard.check(force=True)
            mesh,material = reference_mesh(recipe),Material(**recipe['material'])
            system = assemble_enriched(mesh,material,guard)
            p4 = assemble_highorder(mesh,material,4,guard)
            scalar,space,nk,nm,nq = native_enriched(p4.mesh,material,plan['quadrature_bonus'])
            native_transform,overlap = embedding(p4.mesh,scalar,system.topology,NODES)
            p4_transform,p4_overlap = embedding(p4.mesh,p4.scalar_space,system.topology,p4_nodes)
            k,m,d = system.stiffness.toarray(),system.consistent_mass.toarray(),system.mass_kg
            nk,nm,nq = (native_transform.T@(a@native_transform) for a in (nk,nm,nq))
            pk,pm = (p4_transform.T@(a@p4_transform) for a in (p4.stiffness,p4.mass))
            difference = lambda a,b:float(np.linalg.norm(a-b)/np.linalg.norm(b))
            metrics = dict(native_stiffness_relative_frobenius=difference(k,nk),
                native_consistent_mass_relative_frobenius=difference(m,nm),
                native_nodal_mass_relative_frobenius=difference(np.diag(d),nq),
                p4_stiffness_relative_frobenius=difference(k,pk),p4_consistent_mass_relative_frobenius=difference(m,pm))
            ranks = [int(np.linalg.matrix_rank(t)) for t in (native_transform,p4_transform)]
            points = system.topology.points_m
            affine_mass_errors = []
            for polynomial in (np.ones(len(points)),*points.T):
                q = np.r_[polynomial,np.zeros(2*len(points))]
                affine_mass_errors.append(abs(float(q@(d*q)/(q@(m@q)))-1))
            eigenvalues,modes = eigh(k,np.diag(d))
            normalized = eigenvalues/eigenvalues[-1]
            six_rigid = bool(np.max(abs(normalized[:6]))<1e-10 and normalized[6]>1e-10)
            actual_limit = 2/np.sqrt(eigenvalues[-1])
            stable = bool(system.stable_dt_s<=actual_limit*(1+1e-12))
            initial = modes[:,-4]*1e-9
            probe = (d*modes[:,-4])[None,:]
            omega = float(np.sqrt(eigenvalues[-4]))
            base_dt = .5*system.stable_dt_s
            arrays = dict(stiffness=k,consistent_mass=m,nodal_mass_kg=d,native_stiffness=nk,
                native_consistent_mass=nm,native_nodal_mass=nq,p4_stiffness=pk,p4_consistent_mass=pm,
                native_embedding=native_transform,p4_embedding=p4_transform,points_m=system.topology.points_m)
            errors,energy_ranges = [],[]
            for steps in (plan['diagnostic_steps'],*plan['temporal_steps']):
                dt = base_dt*32/steps
                result = simulate(system,steps,dt,probe,initial_displacement=initial,guard=guard)
                truth = 1e-9*np.cos(omega*result.time_s)
                error = float(np.linalg.norm(result.displacement_m[:,0]-truth)/np.linalg.norm(truth))
                errors.append(error)
                energy_ranges.append(float(np.ptp(result.modified_energy_j)/abs(result.modified_energy_j.mean())))
                arrays.update({f'time_{steps}_s':result.time_s,f'trace_{steps}_m':result.displacement_m,
                    f'modified_energy_{steps}_j':result.modified_energy_j})
            all_ratios = [a/b for a,b in zip(errors,errors[1:])]
            ratios = all_ratios[-2:]
            rule_difference = difference(np.diag(d),m)
            case = output/recipe['name']
            case.mkdir()
            atomic_write(case/'comparison.npz',lambda stream:np.savez_compressed(stream,**arrays))
            row = dict(geometry=recipe['name'],dofs=system.ndof,scalar_dofs=len(d)//3,
                all_coefficient_columns_checked=system.ndof,**metrics,
                native_api_local_scalar_dofs=scalar.GetFE(ng.ElementId(ng.VOL,0)).ndof,
                native_api_local_vector_dofs=space.GetFE(ng.ElementId(ng.VOL,0)).ndof,
                native_embedding_overlap_error=overlap,p4_embedding_overlap_error=p4_overlap,
                native_embedding_rank=ranks[0],p4_embedding_rank=ranks[1],
                affine_mass_relative_errors=affine_mass_errors,topology_screen=system.topology.screen,
                six_rigid_modes=six_rigid,first_six_normalized_eigenvalues=normalized[:6].tolist(),
                seventh_normalized_eigenvalue=float(normalized[6]),stability_bound_valid=stable,
                conservative_dt_s=system.stable_dt_s,exact_discrete_stability_dt_s=float(actual_limit),
                manufactured_mode_omega_rad_s=omega,base_dt_s=base_dt,temporal_errors=errors,
                temporal_step_counts=[plan['diagnostic_steps'],*plan['temporal_steps']],
                temporal_error_ratios=ratios,coarse_temporal_error_ratio=all_ratios[0],
                coarse_ratio_in_screen=3.8<all_ratios[0]<4.2,modified_energy_relative_ranges=energy_ranges,
                intentional_lumped_vs_consistent_relative_frobenius=rule_difference,
                comparison_file=(case/'comparison.npz').relative_to(output).as_posix(),
                comparison_sha256=sha256_file(case/'comparison.npz'))
            row['passed'] = bool(all(np.isfinite(x) for x in metrics.values())
                and max(metrics.values())<=plan['operator_tolerance'] and six_rigid and stable
                and ranks==[system.ndof]*2 and max(affine_mass_errors)<1e-10
                and all(3.8<x<4.2 for x in ratios) and max(energy_ranges)<plan['modified_energy_tolerance'])
            report['cases'].append(row)
            write_json(case/'case-receipt.json',dict(plan_fingerprint=plan['fingerprint'],case=row))
        if reference_source_hashes()!=plan['source_sha256'] or distribution_identity(guard)!=plan['runtime']:
            raise ValueError('enriched reference source/runtime changed during verification')
        report.update(completed=True,passed=all(r['passed'] for r in report['cases']),
            scope='Small enriched/native/P4 operator agreement and discrete explicit time checks; no full-size wave/PZT acceptance.')
    except Exception as error:
        report['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        report['wall_seconds'] = time.perf_counter()-started
        report['resources'] = guard.report()
        report['fingerprint'] = canonical_hash(report)
        write_json(output/'native-report.json',report)
    return report


def verify_enriched(output,guard=None):
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('enriched verification needs a fresh output directory')
    guard = guard or ResourceGuard(ResourceLimits(max_seconds=90,max_rss_mib=1024))
    output.mkdir(parents=True,exist_ok=True)
    result = dict(completed=False,passed=False,ready_for_training_data=False,training_updates=0)
    try:
        plan = dict(kind='enriched_tetrahedron_reference',meshes=recipes(),operator_tolerance=1e-10,
            quadrature_bonus=6,modified_energy_tolerance=1e-9,temporal_steps=[64,128,256],diagnostic_steps=32,
            source_sha256=reference_source_hashes(),
            runtime=distribution_identity(guard))
        plan['fingerprint'] = canonical_hash(plan)
        write_json(output/'enriched-plan.json',plan)
        def snapshot(stream):
            with zipfile.ZipFile(stream,'w',compression=zipfile.ZIP_DEFLATED) as archive:
                for name in plan['source_sha256']:
                    archive.writestr('traxis/'+name,(Path(__file__).parent/name).read_bytes())
        atomic_write(output/'source-snapshot.zip',snapshot)
        result.update(plan_fingerprint=plan['fingerprint'],source_snapshot_sha256=sha256_file(output/'source-snapshot.zip'))
        temporary = output/'temporary'
        temporary.mkdir()
        remaining = max(1.,guard.limits.max_seconds-(time.monotonic()-guard.started))
        command = [sys.executable,'-m','traxis.enriched_verification','--worker','--plan',str(output/'enriched-plan.json'),
            '--output',str(output/'native'),'--max-seconds',str(remaining),'--max-rss-mib',str(guard.limits.max_rss_mib)]
        env = {**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1',
            'TEMP':str(temporary),'TMP':str(temporary),'TMPDIR':str(temporary)}
        result['worker_run'] = run_monitored(command,Path(__file__).resolve().parents[1],env,guard,output/'worker.log')
        if result['worker_run']['exit_code'] not in (0,2):
            raise RuntimeError('enriched reference worker stopped; inspect report/log')
        native = json.loads((output/'native/native-report.json').read_text())
        if (canonical_hash({k:v for k,v in native.items() if k!='fingerprint'})!=native['fingerprint']
                or native['completed'] is not True or native['plan_fingerprint']!=plan['fingerprint']
                or [r['geometry'] for r in native['cases']]!=[r['name'] for r in plan['meshes']]
                or native['passed']!=all(r['passed'] for r in native['cases'])
                or reference_source_hashes()!=plan['source_sha256'] or distribution_identity(guard)!=plan['runtime']):
            raise ValueError('enriched receipt, case counts, source or runtime changed')
        for row in native['cases']:
            path = (output/'native'/row['comparison_file']).resolve()
            if not path.is_relative_to(output/'native') or sha256_file(path)!=row['comparison_sha256']:
                raise ValueError('enriched comparison archive changed')
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
    guard = ResourceGuard(ResourceLimits(max_seconds=args.max_seconds,max_rss_mib=args.max_rss_mib))
    result = worker(args.plan,args.output,guard) if args.worker else verify_enriched(args.output,guard)
    print(json.dumps({k:result[k] for k in ('completed','passed','ready_for_training_data')}))
    raise SystemExit(0 if result['passed'] else 2)
