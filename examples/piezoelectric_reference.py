"""Independent small coupled piezoelectric/electrode/circuit verification.

The material is synthetic verification data, not an identified PZT product.
This example does not replace the frozen full-plate mechanical experiments.
"""

import argparse
import json
import math
import os
from pathlib import Path
import sys
import zipfile

import numpy as np
from scipy.linalg import eigh, expm, solve

from traxis.config import Plate
from traxis.geometry import structured_mesh
from traxis.highorder import _native_space
from traxis.highorder_verification import distribution_identity
from traxis.elmer_plate_refinement import reference_source_hashes
from traxis.elmer_reference import run_monitored
from traxis.resources import ResourceGuard, ResourceLimits
from traxis.storage import canonical_hash, sha256_file, write_json


SOURCES = [
    'https://doc.comsol.com/6.3/doc/com.comsol.help.sme/sme_ug_theory.06.043.html',
    'https://doc.comsol.com/6.4/doc/com.comsol.help.acdc/acdc_ug_electric_fields.07.020.html',
]


def synthetic_material():
    c = np.array([[110,55,50,0,0,0],[55,110,50,0,0,0],[50,50,100,0,0,0],
                  [0,0,0,25,0,0],[0,0,0,0,25,0],[0,0,0,0,0,27.5]])*1e9
    e = np.array([[0,0,0,0,10,0],[0,0,0,10,0,0],[-4,-4,12,0,0,0]],float)
    return c,e,np.diag([12.,12.,10.])*1e-9,7800.


def checked_material(c,e,epsilon,density):
    arrays = (c,e,epsilon)
    if any(np.iscomplexobj(a) for a in arrays):
        raise ValueError('reference material must be real')
    c,e,epsilon = (np.asarray(a,float) for a in arrays)
    if (c.shape!=(6,6) or e.shape!=(3,6) or epsilon.shape!=(3,3)
            or not all(np.isfinite(a).all() for a in (c,e,epsilon))
            or not math.isfinite(density) or density<=0):
        raise ValueError('invalid stress-charge material dimensions or values')
    for a in (c,epsilon):
        if np.linalg.norm(a-a.T)>1e-12*np.linalg.norm(a):
            raise ValueError('elasticity and clamped permittivity must be symmetric')
        np.linalg.cholesky(a)
    return c,e,epsilon,float(density)


def strain_matrix(g):
    """Engineering strain order: xx, yy, zz, yz, xz, xy."""
    b = np.zeros((6,12))
    b[0,:4],b[1,4:8],b[2,8:] = g
    b[3,4:8],b[3,8:] = g[2],g[1]
    b[4,:4],b[4,8:] = g[2],g[0]
    b[5,:4],b[5,4:8] = g[1],g[0]
    return b


def assemble(mesh,material):
    c,e,epsilon,density = checked_material(*material)
    n = len(mesh.points_m)
    if not 4<=n<=64 or not 1<=len(mesh.tetrahedra)<=128:
        raise MemoryError('dense coupled reference is limited to 64 vertices/128 tetrahedra')
    k,m,coupling,dielectric = np.zeros((3*n,3*n)),np.zeros((3*n,3*n)),np.zeros((3*n,n)),np.zeros((n,n))
    volume = 0.
    for tet in mesh.tetrahedra:
        points = mesh.points_m[tet]
        affine = np.column_stack((np.ones(4),points))
        v = abs(np.linalg.det(affine))/6
        if not np.isfinite(v) or v<=0:
            raise ValueError('coupled reference contains a degenerate element')
        g = np.linalg.inv(affine)[1:]
        b = strain_matrix(g)
        ids = np.concatenate([tet+i*n for i in range(3)])
        k[np.ix_(ids,ids)] += v*b.T@c@b
        coupling[np.ix_(ids,tet)] += v*b.T@e.T@g
        dielectric[np.ix_(tet,tet)] += v*g.T@epsilon@g
        m[np.ix_(ids,ids)] += np.kron(np.eye(3),density*v/20*(np.ones((4,4))+np.eye(4)))
        volume += v
    return k,m,coupling,dielectric,volume


def native_operators(mesh,material,guard):
    """Assemble four independent native fields, then verify their nodal map."""
    import ngsolve as ng
    from scipy.sparse import csr_matrix
    c,e,epsilon,density = checked_material(*material)
    n = len(mesh.points_m)
    native,scalar,_ = _native_space(mesh,1,dict(scalar_dofs=n,vector_dofs=3*n),guard)
    space = ng.FESpace([scalar]*4)
    if any(list(space.Range(i))!=list(range(i*n,(i+1)*n)) for i in range(4)):
        raise ArithmeticError('native coupled component order changed')
    u,v = space.TnT()
    def strain(f):
        a = [ng.grad(f[i]) for i in range(3)]
        return [a[0][0],a[1][1],a[2][2],a[1][2]+a[2][1],a[0][2]+a[2][0],a[0][1]+a[1][0]]
    su,sv,gu,gv = strain(u),strain(v),ng.grad(u[3]),ng.grad(v[3])
    expression = sum(c[i,j]*sv[i]*su[j] for i in range(6) for j in range(6))
    expression += sum(e[j,i]*(sv[i]*gu[j]+su[i]*gv[j]) for i in range(6) for j in range(3))
    expression -= sum(epsilon[i,j]*gv[i]*gu[j] for i in range(3) for j in range(3))
    form,mass = ng.BilinearForm(space,symmetric=False),ng.BilinearForm(space,symmetric=False)
    form += expression*ng.dx(bonus_intorder=4)
    mass += density*sum(u[i]*v[i] for i in range(3))*ng.dx(bonus_intorder=4)
    def dense(f):
        f.Assemble()
        values,indices,pointers = f.mat.CSR()
        return csr_matrix((np.array(values),np.array(indices),np.array(pointers)),shape=(4*n,4*n)).toarray()
    a,mm = dense(form),dense(mass)
    mapping = np.full(n,-1,dtype=int)
    maximum_shape_error = 0.
    for element in native.Elements(ng.VOL):
        eid = ng.ElementId(ng.VOL,element.nr)
        fe,dofs = scalar.GetFE(eid),np.asarray(scalar.GetDofNrs(eid),int)
        shapes = np.array([fe.CalcShape(*p[:3]) for p in np.eye(4)])
        permutation = np.argmax(shapes,axis=1)
        maximum_shape_error = max(maximum_shape_error,float(np.max(abs(shapes-np.eye(4)[permutation]))))
        if len(set(permutation))!=4 or len(dofs)!=4 or np.any(dofs<0):
            raise ArithmeticError('native reference is not a complete linear nodal tetrahedron')
        for dof,vertex in zip(dofs[permutation],[p.nr for p in element.vertices]):
            if mapping[dof] not in (-1,vertex):
                raise ArithmeticError('native coupled nodal continuity changed')
            mapping[dof] = vertex
    if maximum_shape_error>1e-13 or sorted(mapping.tolist())!=list(range(n)):
        raise ArithmeticError('native coupled nodal map is incomplete')
    order = np.concatenate([np.argsort(mapping)+i*n for i in range(4)])
    a,mm = a[np.ix_(order,order)],mm[np.ix_(order,order)]
    return a[:3*n,:3*n],mm[:3*n,:3*n],a[:3*n,3*n:],-a[3*n:,3*n:],mapping


def electrode_port(k,m,b,d,bottom,top):
    """Ground the bottom; top is equipotential V; sides are charge-free.

    M u'' + Ksc u + g V = F, Q = Cp V - g.T u.
    """
    n = d.shape[0]
    bottom,top = np.asarray(bottom,int),np.asarray(top,int)
    if (not len(bottom) or not len(top) or np.intersect1d(bottom,top).size
            or len(np.unique(np.r_[bottom,top]))!=len(bottom)+len(top)
            or np.any(np.r_[bottom,top]<0) or np.any(np.r_[bottom,top]>=n)):
        raise ValueError('electrodes must be disjoint, nonempty vertex sets')
    interior = np.setdiff1d(np.arange(n),np.r_[bottom,top])
    psi = np.zeros(n)
    psi[top] = 1.
    correction = np.zeros((len(interior),3*n))
    if len(interior):
        di = d[np.ix_(interior,interior)]
        psi[interior] = -solve(di,d[interior]@psi,assume_a='pos')
        correction = solve(di,b[:,interior].T,assume_a='pos')
    ksc,g,cp = k+b[:,interior]@correction,b@psi,float(psi@d@psi)
    if not math.isfinite(cp) or cp<=0:
        raise ArithmeticError('electrode capacitance must be positive')
    free = np.setdiff1d(np.arange(3*n),np.concatenate([bottom+i*n for i in range(3)]))
    return dict(k=ksc[np.ix_(free,free)],m=m[np.ix_(free,free)],g=g[free],cp=cp,
                free=free,interior=interior,psi=psi,correction=correction)


def balanced_solve(a,rhs):
    scale = np.sqrt(abs(np.diag(a)))
    if np.any(scale==0):
        raise ArithmeticError('coupled reference has an unconstrained zero diagonal')
    return solve(a/scale[:,None]/scale[None,:],rhs/scale,assume_a='sym')/scale


def port_checks(independent,native,bottom,top):
    k,m,b,d,_ = independent
    port = electrode_port(k,m,b,d,bottom,top)
    free,inside,psi = port['free'],port['interior'],port['psi']
    nk,nm,nb,nd,_ = native
    # Independent native mixed equations with prescribed top voltage.
    boundary = np.zeros(len(d)); boundary[top] = 1.
    full = np.block([[nk[np.ix_(free,free)],nb[np.ix_(free,inside)]],
                     [nb[np.ix_(free,inside)].T,-nd[np.ix_(inside,inside)]]])
    rhs = np.r_[-nb[free]@boundary,nd[inside]@boundary]
    direct = balanced_solve(full,rhs)
    u = -solve(port['k'],port['g'],assume_a='pos')
    uf = np.zeros(len(k)); uf[free] = u
    phi = psi.copy(); phi[inside] += port['correction']@uf
    charge = float((d@phi-b.T@uf)[top].sum())
    expected_charge = port['cp']-port['g']@u
    # Independent native mixed equations with an uncharged floating top.
    p = np.zeros((len(d),len(inside)+1))
    p[inside,np.arange(len(inside))] = 1.; p[top,-1] = 1.
    force = np.zeros(len(free)); force[-1] = 1.
    floating = np.block([[nk[np.ix_(free,free)],nb[free]@p],
                         [(nb[free]@p).T,-p.T@nd@p]])
    result = balanced_solve(floating,np.r_[force,np.zeros(p.shape[1])])
    koc = port['k']+np.outer(port['g'],port['g'])/port['cp']
    oc = solve(koc,force,assume_a='pos')
    voltage = float(port['g']@oc/port['cp'])
    def relative(a,b):
        return float(np.linalg.norm(a-b)/np.linalg.norm(b))
    checks = dict(capacitance_f=port['cp'],voltage_actuation_relative_l2=relative(u,direct[:len(free)]),
        internal_potential_relative_l2=relative(phi[inside],direct[len(free):]),
        electrode_charge_relative_error=abs(charge/expected_charge-1),
        open_circuit_displacement_relative_l2=relative(oc,result[:len(free)]),
        open_circuit_voltage_relative_error=abs(voltage/result[-1]-1),
        open_circuit_voltage_v=voltage,short_circuit_voltage_v=0.,
        open_minus_short_min_eigenvalue=float(np.linalg.eigvalsh(koc-port['k']).min()))
    if np.linalg.eigvalsh(port['k']).min()<=0 or np.linalg.eigvalsh(port['m']).min()<=0:
        raise ArithmeticError('clamped coupled mechanical operators must be positive')
    return port,checks


def circuit_check(port,conductance,current,steps):
    """Energy-scaled full modal system, compared with a matrix exponential."""
    if not math.isfinite(conductance) or conductance<0 or not math.isfinite(current) or steps not in (32,64,128):
        raise ValueError('invalid bounded passive-circuit reference settings')
    eigenvalues,phi = eigh(port['k'],port['m'])
    omega = np.sqrt(eigenvalues)
    n = len(omega)
    h = phi.T@port['g']/math.sqrt(port['cp'])
    a = np.zeros((2*n+1,2*n+1))
    a[:n,n:2*n],a[n:2*n,:n] = np.diag(omega),-np.diag(omega)
    a[n:2*n,-1],a[-1,n:2*n] = -h,h
    a[-1,-1] = -conductance/port['cp']
    forcing = np.zeros(2*n+1); forcing[-1] = current/math.sqrt(port['cp'])
    duration = 4/omega[-1]
    initial = np.zeros(2*n+1); initial[-1] = math.sqrt(2e-12)
    extended = np.zeros((len(a)+1,len(a)+1)); extended[:-1,:-1],extended[:-1,-1] = a,forcing
    exact = (expm(duration*extended)@np.r_[initial,1.])[:-1]
    dt,eye = duration/steps,np.eye(len(a))
    lhs = eye-dt*a/2
    advance,drive = solve(lhs,eye+dt*a/2),solve(lhs,dt*forcing)
    x = initial.copy(); energy0 = float(x@x/2)
    maximum_balance,maximum_charge = 0.,0.
    def charge(y):
        return math.sqrt(port['cp'])*y[-1]-port['g']@(phi@(y[:n]/omega))
    q0 = charge(x)
    dissipated,work = 0.,0.
    voltages = [x[-1]/math.sqrt(port['cp'])]
    for _ in range(steps):
        next_x = advance@x+drive
        mid = (x+next_x)/2
        loss = dt*conductance/port['cp']*mid[-1]**2
        supplied = dt*mid@forcing
        maximum_balance = max(maximum_balance,abs((next_x@next_x-x@x)/2+loss-supplied)/energy0)
        maximum_charge = max(maximum_charge,abs(charge(next_x)/q0-1))
        dissipated += loss; work += supplied
        x = next_x
        voltages.append(x[-1]/math.sqrt(port['cp']))
    result = dict(steps=steps,duration_s=duration,conductance_s=conductance,current_a=current,
        final_relative_l2=float(np.linalg.norm(x-exact)/np.linalg.norm(exact)),
        maximum_step_energy_balance_relative_error=maximum_balance,
        total_energy_balance_relative_error=abs((x@x/2-energy0)+dissipated-work)/energy0,
        initial_energy_j=energy0,final_energy_j=float(x@x/2),dissipated_energy_j=dissipated,
        supplied_energy_j=work,isolated_charge_relative_range=maximum_charge if conductance==0 and current==0 else None)
    proof = dict(time_s=np.linspace(0,duration,steps+1),voltage_v=np.array(voltages),
        initial_state=initial,final_state=x,exact_final_state=exact,system_matrix=a,forcing=forcing,
        modal_vectors=phi,angular_frequencies_rad_s=omega)
    return result,proof


def verify(output,guard):
    output = Path(output)
    material = synthetic_material()
    report = dict(completed=False,passed=False,ready_for_training_data=False,training_updates=0,
        material_status='Synthetic positive-definite verification coefficients; no hardware identification.',
        strain_order=['xx','yy','zz','yz','xz','xy'],sources=SOURCES,cases=[])
    material_dict = dict(elasticity_pa=material[0].tolist(),coupling_c_m2=material[1].tolist(),
        clamped_permittivity_f_m=material[2].tolist(),density_kg_m3=material[3])
    report['material'] = material_dict
    arrays = {}
    for name,transform in (('rectangular',np.eye(3)),('affine-skewed',np.array([[1,.13,.07],[.04,1,.11],[.02,.03,1.]]))):
        guard.check(force=True)
        mesh = structured_mesh(Plate(width_m=.04,height_m=.03,thickness_m=.003),(2,2,2))
        bottom,top = np.flatnonzero(mesh.points_m[:,2]==0),np.flatnonzero(mesh.points_m[:,2]==.003)
        mesh.points_m = mesh.points_m@transform.T
        independent = assemble(mesh,material)
        native = native_operators(mesh,material,guard)
        errors = {label:float(np.linalg.norm(a-b)/np.linalg.norm(a)) for label,a,b in
            zip(('stiffness','consistent_mass','coupling','dielectric'),independent[:4],native[:4])}
        k,m,b,d,volume = independent
        gradient = np.array([[2,-.3,.7],[.2,-1,.4],[-.1,.8,.5]])*1e-6
        field = np.array([30.,-20.,100.])
        displacement = (mesh.points_m@gradient.T).T.ravel()
        potential = -mesh.points_m@field
        strain = np.r_[np.diag(gradient),gradient[1,2]+gradient[2,1],gradient[0,2]+gradient[2,0],gradient[0,1]+gradient[1,0]]
        exact = [volume*strain@material[0]@strain,-volume*field@material[1]@strain,volume*field@material[2]@field]
        measured = [displacement@k@displacement,displacement@b@potential,potential@d@potential]
        affine_errors = [float(abs(a/z-1)) for a,z in zip(measured,exact)]
        gauge = float(np.linalg.norm(d@np.ones(len(d)))/(np.linalg.norm(d)*math.sqrt(len(d))))
        rigid = []
        for axis in np.eye(3):
            rigid.extend([np.tile(axis,(len(d),1)).T.ravel(),np.cross(np.broadcast_to(axis,mesh.points_m.shape),mesh.points_m).T.ravel()])
        rigid_error = max(float(np.linalg.norm(k@v)/(np.linalg.norm(k)*np.linalg.norm(v))) for v in rigid)
        port,checks = port_checks(independent,native,bottom,top)
        if name=='rectangular':
            checks['analytic_clamped_capacitance_relative_error'] = abs(port['cp']/(material[2][2,2]*.04*.03/.003)-1)
        omega_max = math.sqrt(eigh(port['k'],port['m'],eigvals_only=True)[-1])
        circuits = []
        for mode,g,i in (('isolated',0.,0.),('resistive',port['cp']*omega_max/3,0.),
                         ('current-driven',port['cp']*omega_max/3,math.sqrt(port['cp'])*math.sqrt(2e-12)*omega_max*.1)):
            rows = []
            for steps in (32,64,128):
                row,proof = circuit_check(port,g,i,steps)
                rows.append(row)
                for label,value in proof.items():
                    arrays[f'{name}_{mode}_{steps}_{label}'] = value
            ratios = [rows[j]['final_relative_l2']/rows[j+1]['final_relative_l2'] for j in (0,1)]
            circuits.append(dict(mode=mode,levels=rows,error_ratios=ratios,
                passed=bool(all(3.8<r<4.2 for r in ratios) and all(r['maximum_step_energy_balance_relative_error']<1e-10
                and r['total_energy_balance_relative_error']<1e-10 for r in rows)
                and (mode!='isolated' or max(r['isolated_charge_relative_range'] for r in rows)<1e-10))))
        check_errors = [v for label,v in checks.items() if 'relative' in label]
        passed = bool(max(errors.values())<1e-11 and max(affine_errors)<1e-11 and max(gauge,rigid_error)<1e-12
            and max(check_errors)<1e-9 and all(r['passed'] for r in circuits)
            and checks['open_minus_short_min_eigenvalue']>-1e-12*np.linalg.norm(port['k']))
        report['cases'].append(dict(name=name,transform=transform.tolist(),vertices=len(d),tetrahedra=len(mesh.tetrahedra),
            volume_m3=volume,operator_relative_l2=errors,affine_energy_relative_errors=affine_errors,
            gauge_relative_residual=gauge,rigid_relative_residual=rigid_error,port=checks,circuits=circuits,passed=passed))
        for label,value in zip(('stiffness','mass','coupling','dielectric'),independent[:4]):
            arrays[f'{name}_independent_{label}'] = value
        for label,value in zip(('stiffness','mass','coupling','dielectric'),native[:4]):
            arrays[f'{name}_native_{label}'] = value
        arrays[f'{name}_native_to_vertex'] = native[-1]
    archive = output/'coupled-reference.npz'
    np.savez_compressed(archive,**arrays)
    report.update(completed=True,passed=all(c['passed'] for c in report['cases']),archive_sha256=sha256_file(archive),
        scope='Small 3D constitutive/electrode/circuit reference; no bonded plate, PZT product, wave convergence or physical acceptance.')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--worker',action='store_true')
    parser.add_argument('--plan',type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    guard = ResourceGuard(ResourceLimits(max_seconds=120,max_rss_mib=512,max_system_memory_percent=85,min_available_mib=2048))
    source = dict(driver_sha256=sha256_file(Path(__file__)),source_sha256=reference_source_hashes())
    if args.worker:
        plan = json.loads(args.plan.read_text())
        if (canonical_hash({k:v for k,v in plan.items() if k!='fingerprint'})!=plan['fingerprint']
                or any(plan[k]!=v for k,v in source.items()) or distribution_identity(guard)!=plan['runtime']):
            raise ValueError('coupled reference plan/source/runtime changed')
        output.mkdir()
        report = verify(output,guard)
        if (any(plan[k]!=v for k,v in dict(driver_sha256=sha256_file(Path(__file__)),source_sha256=reference_source_hashes()).items())
                or distribution_identity(guard)!=plan['runtime']):
            raise ValueError('coupled reference source changed during verification')
        report.update(plan_fingerprint=plan['fingerprint'],resources=guard.report())
        report['fingerprint'] = canonical_hash(report)
        write_json(output/'reference-report.json',report)
        print(json.dumps({k:v for k,v in report.items() if k not in ('resources','cases','material')}))
        return 0 if report['passed'] else 2
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('coupled reference needs a fresh output directory')
    output.mkdir(parents=True,exist_ok=True)
    plan = {**source,'runtime':distribution_identity(guard),'limits':guard.report()['limits']}
    with zipfile.ZipFile(output/'source-snapshot.zip','w',compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(Path(__file__),'examples/piezoelectric_reference.py')
        for name in plan['source_sha256']:
            archive.write(Path(__file__).resolve().parents[1]/'traxis'/name,'traxis/'+name)
    plan['source_snapshot_sha256'] = sha256_file(output/'source-snapshot.zip')
    plan['fingerprint'] = canonical_hash(plan)
    path = output/'reference-plan.json'; write_json(path,plan)
    env = os.environ.copy(); env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    temporary = output/'temporary-files'; temporary.mkdir()
    env.update({name:str(temporary) for name in ('TEMP','TMP','TMPDIR')})
    command = [sys.executable,'-m','examples.piezoelectric_reference','--worker','--plan',str(path),'--output',str(output/'native')]
    report = dict(completed=False,passed=False,ready_for_training_data=False,training_updates=0,plan_fingerprint=plan['fingerprint'])
    try:
        worker = run_monitored(command,Path(__file__).resolve().parents[1],env,guard,output/'worker.log')
        report['worker_run'] = worker
        if worker['exit_code'] not in (0,2) or not (output/'native/reference-report.json').is_file():
            raise RuntimeError(f"coupled reference worker exited {worker['exit_code']}; see worker.log")
        native = json.loads((output/'native/reference-report.json').read_text())
        if (any(plan[k]!=v for k,v in dict(driver_sha256=sha256_file(Path(__file__)),source_sha256=reference_source_hashes()).items())
                or native['plan_fingerprint']!=plan['fingerprint']
                or canonical_hash({k:v for k,v in native.items() if k!='fingerprint'})!=native['fingerprint']
                or sha256_file(output/'native/coupled-reference.npz')!=native['archive_sha256']
                or sha256_file(output/'source-snapshot.zip')!=plan['source_snapshot_sha256']):
            raise ValueError('coupled reference source changed before receipt verification')
        report.update(completed=True,passed=worker['exit_code']==0 and native['passed'],worker_run=worker,
            native_report_sha256=sha256_file(output/'native/reference-report.json'))
    except Exception as error:
        report['error'] = f'{type(error).__name__}: {error}'
    finally:
        report['resources'] = guard.report(); report['fingerprint'] = canonical_hash(report)
        write_json(output/'monitor-result.json',report)
    print(json.dumps({k:v for k,v in report.items() if k!='resources'}))
    return 0 if report['passed'] else 2


if __name__=='__main__':
    raise SystemExit(main())
