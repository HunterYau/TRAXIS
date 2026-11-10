"""Native operator and parity-separated Lamb checks for tensor elements."""
import argparse
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import sys
import zipfile
import numpy as np
from scipy import sparse
from scipy.linalg import eigh
from traxis.config import Plate,Material
from traxis.dispersion import wavenumber_rad_m
from traxis.tensor_hex import line_rule,make_grid,assemble_tensor
from traxis.elmer_plate_refinement import reference_source_hashes
from traxis.elmer_reference import run_monitored
from traxis.highorder_verification import distribution_identity
from traxis.resources import ResourceGuard,ResourceLimits
from traxis.storage import canonical_hash,sha256_file,write_json


def native_operators(grid,material,guard):
    import ngsolve as ng
    import netgen.meshing as nm
    ng.SetNumThreads(1)
    native = nm.Mesh(dim=3)
    vertices = [native.Add(nm.MeshPoint(nm.Pnt(*p))) for p in grid.corner_points_m]
    region = native.AddRegion('solid',dim=3)
    for element in grid.corner_elements:
        native.Add(nm.Element3D(region,[vertices[i] for i in element]))
    mesh = ng.Mesh(native)
    scalar = ng.H1(mesh,order=grid.order)
    n = len(grid.points_m)
    if scalar.ndof!=n:
        raise ArithmeticError('native hexahedron is not the expected complete tensor space')
    space = ng.FESpace([scalar]*3)
    if any(list(space.Range(c))!=list(range(c*n,(c+1)*n)) for c in range(3)):
        raise ArithmeticError('native tensor component order changed')
    u,v = space.TnT()
    du,dv = [[ng.grad(field[i]) for i in range(3)] for field in (u,v)]
    lam,mu = material.lame
    expression = mu/2*sum((du[i][j]+du[j][i])*(dv[i][j]+dv[j][i]) for i in range(3) for j in range(3))
    expression += lam*sum(du[i][i] for i in range(3))*sum(dv[i][i] for i in range(3))
    mass_expression = material.density_kg_m3*sum(u[i]*v[i] for i in range(3))
    nodes,weights,*_ = line_rule(grid.order)
    points = np.stack(np.meshgrid(*[(nodes+1)/2]*3,indexing='ij'),axis=-1).reshape(-1,3)
    quadrature_weights = np.kron(np.kron(weights/2,weights/2),weights/2)
    rule = ng.IntegrationRule(points=points.tolist(),weights=quadrature_weights.tolist())
    integration = [ng.dx(bonus_intorder=2*grid.order+2),ng.dx(intrules={ng.ET.HEX:rule})]
    forms = []
    for dx in integration:
        for expression_here in (expression,mass_expression):
            guard.check()
            form = ng.BilinearForm(space,symmetric=False)
            form += expression_here*dx
            form.Assemble()
            values,indices,pointers = form.mat.CSR()
            forms.append(sparse.csr_matrix((np.array(values),np.array(indices),np.array(pointers)),shape=(3*n,3*n)).toarray())
    transform = np.zeros((n,n)); seen = np.zeros(n,dtype=bool); overlap = 0.
    for element in mesh.Elements(ng.VOL):
        eid = ng.ElementId(ng.VOL,element.nr)
        dofs = np.asarray(scalar.GetDofNrs(eid),int)
        fe,trafo = scalar.GetFE(eid),mesh.GetTrafo(eid)
        if len(dofs)!=(grid.order+1)**3 or np.any(dofs<0):
            raise ArithmeticError('native hexahedron local space changed')
        for point,ip in zip(points,rule):
            physical = np.array(trafo(ip).point)
            distances = np.linalg.norm(grid.points_m-physical,axis=1)
            node = int(np.argmin(distances))
            if distances[node]>1e-12:
                raise ArithmeticError('native tensor point is absent from the independent grid')
            row = np.zeros(n); row[dofs] = fe.CalcShape(*point)
            if seen[node]:
                overlap = max(overlap,float(np.max(abs(row-transform[node]))))
            else:
                transform[node] = row; seen[node] = True
    if not seen.all() or overlap>1e-11:
        raise ArithmeticError('native tensor nodal transform violates shared continuity')
    inverse = np.linalg.inv(transform)
    if np.linalg.norm(transform@inverse-np.eye(n))>1e-10:
        raise ArithmeticError('native tensor evaluation transform is singular')
    vector = np.kron(np.eye(3),inverse)
    return [vector.T@a@vector for a in forms],transform,overlap


def operator_cases(output,guard):
    material = Material(); records = []; arrays = {}
    for name,transform in (('rectangular',np.eye(3)),('affine-skewed',np.array([[1,.13,.07],[.04,1,.11],[.02,.03,1.]]))):
        for order in range(1,5):
            guard.check(force=True)
            grid = make_grid(Plate(width_m=.04,height_m=.03,thickness_m=.003),(2,1,1),order,transform)
            exact,gll = (assemble_tensor(grid,material,rule,guard) for rule in ('exact','gll'))
            native,evaluation,overlap = native_operators(grid,material,guard)
            independent = [exact.stiffness.toarray(),exact.consistent_mass.toarray(),gll.stiffness.toarray(),np.diag(gll.mass_kg)]
            labels = ['exact_stiffness','consistent_mass','gll_stiffness','gll_mass']
            errors = {label:float(np.linalg.norm(a-b)/np.linalg.norm(a)) for label,a,b in zip(labels,independent,native)}
            eigenvalues = eigh(independent[2],independent[3],eigvals_only=True)
            normalized = eigenvalues/eigenvalues[-1]
            rigid = bool(max(abs(normalized[:6]))<1e-12 and normalized[6]>1e-12)
            row = dict(geometry=name,order=order,cells=list(grid.cells),scalar_dofs=len(grid.points_m),vector_dofs=len(gll.mass_kg),
                operator_relative_l2=errors,maximum_shared_transform_difference=overlap,six_rigid_modes=rigid,
                smallest_six_eigenvalues_over_largest=normalized[:6].tolist(),seventh_eigenvalue_over_largest=float(normalized[6]),
                gll_vs_exact_stiffness_relative_l2=float(np.linalg.norm(independent[2]-independent[0])/np.linalg.norm(independent[0])),
                stable_dt_s=gll.stable_dt_s,positive_gll_mass=bool(np.all(gll.mass_kg>0)),
                passed=bool(max(errors.values())<1e-10 and rigid))
            records.append(row)
            key=f'{name}_p{order}_'
            for label,a,b in zip(labels,independent,native):
                arrays[key+'independent_'+label],arrays[key+'native_'+label] = a,b
            arrays[key+'evaluation_transform'] = evaluation
            arrays[key+'points_m'] = grid.points_m
    np.savez_compressed(output/'operator-comparisons.npz',**arrays)
    return records


def parity_projection(grid,wavenumber,mode):
    """Tie x/y faces, then impose physical midplane Lamb parity."""
    if grid.cells[:2]!=(1,1) or mode not in ('A0','S0') or not math.isfinite(wavenumber) or wavenumber<=0:
        raise ValueError('tensor Lamb projection needs one lateral cell and a positive mode wavenumber')
    p,nz = grid.order,grid.cells[2]*grid.order+1
    shape = (p+1,p+1,nz); n = len(grid.points_m)
    ix,iy,iz = np.indices(shape).reshape(3,-1)
    ids = ((ix%p)*p+iy%p)*nz+iz
    ns = p*p*nz
    phase = np.exp(1j*wavenumber*grid.plate.width_m*(ix==p))
    periodic = sparse.coo_matrix((np.tile(phase,2),(np.r_[np.arange(n),np.arange(n)+2*n],np.r_[ids,ids+ns])),shape=(3*n,2*ns)).tocsr()
    rows,columns,values = [],[],[]; column = 0
    for component,parity in ((0,-1 if mode=='A0' else 1),(1,1 if mode=='A0' else -1)):
        for xy in range(p*p):
            for z in range((nz+1)//2):
                reflected = nz-1-z
                if reflected==z and parity==-1:
                    continue
                base = component*ns+xy*nz
                rows.append(base+z); columns.append(column); values.append(1. if reflected==z else 1/math.sqrt(2))
                if reflected!=z:
                    rows.append(base+reflected); columns.append(column); values.append(parity/math.sqrt(2))
                column += 1
    parity_map = sparse.coo_matrix((values,(rows,columns)),shape=(2*ns,column)).tocsr()
    return periodic@parity_map


def dispersion_cases(output,guard):
    material = Material(); records = []; proofs = {}
    waves = {(h,f,mode):wavenumber_rad_m(f,h,mode,material) for h in (.003,.001) for f in (100000.,200000.) for mode in ('A0','S0')}
    for thickness in (.003,.001):
        for width in (.005,.005/1.5,.0025):
            for order in (2,3,4):
                for depth in (1,2,4):
                    guard.check(force=True)
                    grid = make_grid(Plate(width,width,thickness),(1,1,depth),order)
                    for rule in ('gll','exact'):
                        system = assemble_tensor(grid,material,rule,guard)
                        for frequency in (100000.,200000.):
                            for mode in ('A0','S0'):
                                wave = waves[thickness,frequency,mode]
                                row = dict(thickness_m=thickness,lateral_spacing_m=width,order=order,thickness_cells=depth,
                                    stiffness_rule=rule,target_frequency_hz=frequency,mode=mode,wavenumber_rad_m=wave,
                                    cells_per_wavelength=2*math.pi/wave/width,within_first_brillouin_zone=bool(wave*width<math.pi),
                                    vector_dofs=len(system.mass_kg),stable_dt_s=system.stable_dt_s)
                                if wave*width>=math.pi:
                                    row.update(branch_identified=False,reason='Target phase outside first cell Brillouin zone; lowest-branch comparison is withheld.')
                                else:
                                    projection = parity_projection(grid,wave,mode)
                                    opposite = parity_projection(grid,wave,'S0' if mode=='A0' else 'A0')
                                    k = (projection.conj().T@system.stiffness@projection).toarray()
                                    m = (projection.conj().T@sparse.diags(system.mass_kg)@projection).toarray()
                                    cross = (opposite.conj().T@system.stiffness@projection).toarray()
                                    cross_error = float(np.linalg.norm(cross)/np.linalg.norm(k))
                                    hermitian = max(float(np.linalg.norm(a-a.conj().T)/np.linalg.norm(a)) for a in (k,m))
                                    if max(cross_error,hermitian)>1e-10:
                                        raise ArithmeticError('tensor physical parity or Hermitian reduction failed')
                                    eigenvalues = eigh(k,m,eigvals_only=True)
                                    if not np.isfinite(eigenvalues).all() or eigenvalues[0]<=0:
                                        raise ArithmeticError('tensor Lamb branch is nonpositive')
                                    numerical = float(np.sqrt(eigenvalues[0])/(2*math.pi))
                                    error = numerical/frequency-1
                                    row.update(branch_identified=True,numerical_frequency_hz=numerical,
                                        signed_relative_frequency_error=error,design_screen_passed=bool(abs(error)<.0005),
                                        parity_cross_relative_l2=cross_error,hermitian_relative_error=hermitian,condensed_dofs=len(k))
                                    # Retain all modes of the prospective finest case for audit.
                                    if order==4 and depth==4 and width==.0025:
                                        key=f'h{thickness}_f{frequency}_{mode}_{rule}'
                                        row['proof_key'] = key
                                        proofs[key+'_stiffness'],proofs[key+'_mass'],proofs[key+'_eigenvalues'] = k,m,eigenvalues
                                records.append(row)
    np.savez_compressed(output/'finest-dispersion.npz',**proofs)
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=('operators','dispersion'),required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--worker',action='store_true')
    parser.add_argument('--plan',type=Path)
    args = parser.parse_args(); output = args.output.resolve()
    guard = ResourceGuard(ResourceLimits(max_seconds=180,max_rss_mib=512,max_system_memory_percent=85,min_available_mib=2048))
    identity = dict(source_sha256=reference_source_hashes(),driver_sha256=sha256_file(Path(__file__)))
    if args.worker:
        plan = json.loads(args.plan.read_text())
        if (canonical_hash({k:v for k,v in plan.items() if k!='fingerprint'})!=plan['fingerprint']
                or any(plan[k]!=v for k,v in identity.items()) or plan['mode']!=args.mode
                or plan['runtime']!=distribution_identity(guard)):
            raise ValueError('tensor reference source/runtime/plan changed')
        output.mkdir()
        report = dict(completed=False,passed=False,ready_for_training_data=False,training_updates=0,mode=args.mode,
            plan_fingerprint=plan['fingerprint'],cases=[])
        try:
            report['cases'] = operator_cases(output,guard) if args.mode=='operators' else dispersion_cases(output,guard)
            if args.mode=='operators':
                passed = all(c['passed'] for c in report['cases']); file = 'operator-comparisons.npz'
            else:
                finest = [c for c in report['cases'] if c['order']==4 and c['thickness_cells']==4 and c['lateral_spacing_m']==.0025]
                passed = len(finest)==16 and all(c.get('design_screen_passed',False) for c in finest); file = 'finest-dispersion.npz'
            if (reference_source_hashes()!=plan['source_sha256'] or sha256_file(Path(__file__))!=plan['driver_sha256']
                    or distribution_identity(guard)!=plan['runtime']):
                raise ValueError('tensor reference source/runtime changed during verification')
            report.update(completed=True,passed=bool(passed),archive=file,archive_sha256=sha256_file(output/file),
                scope='Small affine tensor references and periodic-cell design checks; no finite-plate wave, recess/PZT or physical acceptance.')
        except Exception as error:
            report['error'] = f'{type(error).__name__}: {error}'
            raise
        finally:
            report['resources'] = guard.report(); report['fingerprint'] = canonical_hash(report)
            write_json(output/'reference-report.json',report)
        return 0 if report['passed'] else 2
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('tensor reference needs a fresh output directory')
    output.mkdir(parents=True,exist_ok=True)
    plan = {**identity,'mode':args.mode,'material':asdict(Material()),'runtime':distribution_identity(guard),
        'operator_relative_tolerance':1e-10,'finest_frequency_design_screen':.0005,'limits':guard.report()['limits']}
    with zipfile.ZipFile(output/'source-snapshot.zip','w',compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(Path(__file__),'examples/tensor_reference.py')
        for name in plan['source_sha256']:
            archive.write(Path(__file__).resolve().parents[1]/'traxis'/name,'traxis/'+name)
    plan['source_snapshot_sha256'] = sha256_file(output/'source-snapshot.zip'); plan['fingerprint'] = canonical_hash(plan)
    path = output/'reference-plan.json'; write_json(path,plan)
    env = os.environ.copy(); env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    temporary = output/'temporary-files'; temporary.mkdir()
    env.update({name:str(temporary) for name in ('TEMP','TMP','TMPDIR')})
    command = [sys.executable,'-m','examples.tensor_reference','--worker','--plan',str(path),'--mode',args.mode,'--output',str(output/'native')]
    report = dict(completed=False,passed=False,ready_for_training_data=False,training_updates=0,plan_fingerprint=plan['fingerprint'])
    try:
        worker = run_monitored(command,Path(__file__).resolve().parents[1],env,guard,output/'worker.log')
        report['worker_run'] = worker
        if worker['exit_code'] not in (0,2) or not (output/'native/reference-report.json').is_file():
            raise RuntimeError(f"tensor worker exited {worker['exit_code']}; see worker.log")
        native = json.loads((output/'native/reference-report.json').read_text())
        if (native['plan_fingerprint']!=plan['fingerprint'] or canonical_hash({k:v for k,v in native.items() if k!='fingerprint'})!=native['fingerprint']
                or sha256_file(output/'native'/native['archive'])!=native['archive_sha256']
                or reference_source_hashes()!=plan['source_sha256'] or sha256_file(Path(__file__))!=plan['driver_sha256']):
            raise ValueError('tensor reference receipt or source changed')
        report.update(completed=True,passed=worker['exit_code']==0 and native['passed'],
            native_report_sha256=sha256_file(output/'native/reference-report.json'))
    except Exception as error:
        report['error'] = f'{type(error).__name__}: {error}'
    finally:
        report['resources'] = guard.report();report['fingerprint'] = canonical_hash(report)
        write_json(output/'monitor-result.json',report)
    print(json.dumps({k:v for k,v in report.items() if k!='resources'}))
    return 0 if report['passed'] else 2


if __name__=='__main__':
    raise SystemExit(main())
