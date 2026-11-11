"""Bounded nonaffine hexahedron operators against native deformed geometry."""
import argparse
import json
import os
from pathlib import Path
import sys
import zipfile

import numpy as np
from numpy.polynomial.legendre import leggauss

from examples.tensor_wave import PROJECT,identities,read_json,checked_receipt,sealed,snapshot_sources
from traxis.config import Material,Plate
from traxis.elmer_reference import run_monitored
from traxis.highorder_verification import distribution_identity
from traxis.resources import ResourceGuard,ResourceLimits
from traxis.storage import sha256_file,write_json
from traxis.tensor_hex import make_grid,line_rule


LENGTHS=np.array([.04,.03,.003])


def geometry(points):
    x,y,z=np.asarray(points).T;w,h,t=LENGTHS
    physical=np.column_stack([x+.1*y*z/t,y+.05*x*x/w,z*(1+.12*x/w-.08*y/h+.04*x*y/(w*h))])
    jac=np.zeros((len(x),3,3));jac[:,0,0]=1; jac[:,0,1]=.1*z/t; jac[:,0,2]=.1*y/t
    jac[:,1,0]=.1*x/w;jac[:,1,1]=1
    jac[:,2,0]=z*(.12/w+.04*y/(w*h));jac[:,2,1]=z*(-.08/h+.04*x/(w*h))
    jac[:,2,2]=1+.12*x/w-.08*y/h+.04*x*y/(w*h)
    return physical,jac


def independent_operators(grid,material,guard):
    q=grid.order+1;nodes,w,d,*_=line_rule(grid.order);ns=len(grid.points_m);n=q**3
    ijk=np.indices((q,q,q)).reshape(3,-1).T;grad=np.zeros((n,n,3))
    for axis in range(3):
        value=d[ijk[:,axis,None],ijk[None,:,axis]]
        for other in range(3):
            if other!=axis:value=value*(ijk[:,other,None]==ijk[None,:,other])
        grad[:,:,axis]=value
    mass=np.zeros(ns);stiffness=np.zeros((3*ns,3*ns));determinants=[];conditions=[]
    lengths=np.array([grid.plate.width_m,grid.plate.height_m,grid.plate.thickness_m])
    base_jac=grid.transform@np.diag(lengths/(2*np.array(grid.cells)))
    for ids in grid.element_nodes:
        guard.check();points=grid.points_m[ids];_,map_jac=geometry(points)
        jac=map_jac@base_jac;det=np.linalg.det(jac);condition=np.linalg.cond(jac)
        if np.any(det<=0) or np.any(condition>1e4):raise ValueError('mapped fixture has invalid geometric Jacobians')
        g=np.einsum('qia,qad->qid',grad,np.linalg.inv(jac))
        volume=np.kron(np.kron(w,w),w)*det;np.add.at(mass,ids,material.density_kg_m3*volume)
        lam,mu=material.lame;dot=np.einsum('qid,qjd,q->ij',g,g,volume,optimize=True)
        local=np.empty((3*n,3*n))
        for a in range(3):
            for b in range(3):
                block=lam*(g[:,:,a].T@(volume[:,None]*g[:,:,b]))+mu*(g[:,:,b].T@(volume[:,None]*g[:,:,a]))
                if a==b:block+=mu*dot
                local[a*n:(a+1)*n,b*n:(b+1)*n]=block
        vector_ids=np.concatenate([ids+c*ns for c in range(3)])
        stiffness[np.ix_(vector_ids,vector_ids)]+=local
        determinants.extend(det.tolist());conditions.extend(condition.tolist())
    return stiffness,np.tile(mass,3),dict(minimum_jacobian_determinant=min(determinants),maximum_jacobian_condition=max(conditions))


def native_operators(grid,material,guard):
    import ngsolve as ng
    import netgen.meshing as nm
    from scipy import sparse
    ng.SetNumThreads(1);raw=nm.Mesh(dim=3)
    vertices=[raw.Add(nm.MeshPoint(nm.Pnt(*p))) for p in grid.corner_points_m]
    region=raw.AddRegion('solid',dim=3)
    for element in grid.corner_elements:raw.Add(nm.Element3D(region,[vertices[i] for i in element]))
    mesh=ng.Mesh(raw);deformation=ng.GridFunction(ng.VectorH1(mesh,order=2))
    x,y,z=ng.x,ng.y,ng.z;w,h,t=LENGTHS
    deformation.Set(ng.CoefficientFunction((.1*y*z/t,.05*x*x/w,z*(.12*x/w-.08*y/h+.04*x*y/(w*h)))))
    mesh.SetDeformation(deformation)
    scalar=ng.H1(mesh,order=grid.order);ns=len(grid.points_m)
    if scalar.ndof!=ns:raise ArithmeticError('native mapped scalar count changed')
    space=ng.FESpace([scalar]*3);u,v=space.TnT();du,dv=([ng.grad(f[i]) for i in range(3)] for f in (u,v))
    lam,mu=material.lame
    expression=mu/2*sum((du[i][j]+du[j][i])*(dv[i][j]+dv[j][i]) for i in range(3) for j in range(3))
    expression+=lam*sum(du[i][i] for i in range(3))*sum(dv[i][i] for i in range(3))
    nodes,weights,*_=line_rule(grid.order)
    points=np.stack(np.meshgrid(*[(nodes+1)/2]*3,indexing='ij'),axis=-1).reshape(-1,3)
    rule=ng.IntegrationRule(points=points.tolist(),weights=np.kron(np.kron(weights/2,weights/2),weights/2).tolist())
    forms=[]
    for expression_here in (expression,material.density_kg_m3*sum(u[i]*v[i] for i in range(3))):
        guard.check();form=ng.BilinearForm(space,symmetric=False);form+=expression_here*ng.dx(intrules={ng.ET.HEX:rule});form.Assemble()
        values,indices,pointers=form.mat.CSR();forms.append(sparse.csr_matrix((np.array(values),np.array(indices),np.array(pointers)),shape=(3*ns,3*ns)).toarray())
    physical=geometry(grid.points_m)[0];transform=np.zeros((ns,ns));seen=np.zeros(ns,dtype=bool);overlap=0.;coordinate_error=0.
    for element in mesh.Elements(ng.VOL):
        guard.check();eid=ng.ElementId(ng.VOL,element.nr);dofs=np.asarray(scalar.GetDofNrs(eid),int)
        fe,trafo=scalar.GetFE(eid),mesh.GetTrafo(eid)
        for point,ip in zip(points,rule):
            measured=np.array(trafo(ip).point);distance=np.linalg.norm(physical-measured,axis=1);node=int(np.argmin(distance))
            coordinate_error=max(coordinate_error,float(distance[node]));row=np.zeros(ns);row[dofs]=fe.CalcShape(*point)
            if seen[node]:overlap=max(overlap,float(np.max(abs(row-transform[node]))))
            else:transform[node]=row;seen[node]=True
    if not seen.all() or coordinate_error>1e-12 or overlap>1e-11:raise ArithmeticError('native mapped geometry/shared nodal transform failed')
    inverse=np.linalg.inv(transform);vector=np.kron(np.eye(3),inverse)
    return [vector.T@a@vector for a in forms],dict(physical_coordinate_maximum_absolute_error_m=coordinate_error,shared_transform_maximum_error=overlap)


def worker(plan_path,output,guard):
    from scipy.linalg import eigh
    plan=checked_receipt(read_json(plan_path))
    if (any(plan[k]!=v for k,v in identities().items()) or plan['driver_sha256_self']!=sha256_file(__file__)
            or plan['runtime']!=distribution_identity(guard)):raise ValueError('mapped reference source/runtime changed')
    output.mkdir();report=dict(completed=False,passed=False,ready_for_training_data=False,training_updates=0,plan_fingerprint=plan['fingerprint'])
    try:
        material=Material();arrays={};rows=[]
        a,w=leggauss(7);points=(np.stack(np.meshgrid(a,a,a,indexing='ij'),axis=-1).reshape(-1,3)+1)*LENGTHS/2
        exact_volume=float(np.kron(np.kron(w,w),w)@np.linalg.det(geometry(points)[1])*np.prod(LENGTHS)/8)
        for p in plan['orders']:
            grid=make_grid(Plate(*LENGTHS),(2,1,1),p);k,mass,details=independent_operators(grid,material,guard)
            native,geometry_checks=native_operators(grid,material,guard);physical=geometry(grid.points_m)[0];ns=len(physical)
            k_error=float(np.linalg.norm(k-native[0])/np.linalg.norm(native[0]))
            m_error=float(np.linalg.norm(np.diag(mass)-native[1])/np.linalg.norm(native[1]))
            rigid=[]
            for axis in np.eye(3):
                rigid.append(np.broadcast_to(axis,(ns,3)).T.ravel())
                rigid.append(np.cross(axis,physical-physical.mean(axis=0)).T.ravel())
            rigid_error=max(float(np.linalg.norm(k@r)/(np.linalg.norm(k)*np.linalg.norm(r))) for r in rigid)
            gradient=np.array([[.13,-.07,.02],[.04,.09,-.03],[-.05,.08,.11]])
            displacement=(physical@gradient.T).T.ravel();strain=(gradient+gradient.T)/2;lam,mu=material.lame
            expected=exact_volume*(mu*np.sum(strain*strain)+lam*np.trace(strain)**2/2)
            energy=float(displacement@k@displacement/2);energy_error=abs(energy/expected-1)
            volume_error=abs(mass[:ns].sum()/(material.density_kg_m3*exact_volume)-1)
            if not np.isfinite(mass).all() or np.any(mass<=0):raise ArithmeticError('mapped GLL mass is not finite and positive')
            upper=float(np.max(np.sum(abs(k),axis=1)/mass))
            smallest=eigh(k,np.diag(mass),eigvals_only=True,subset_by_index=(0,6))
            normalized=smallest/upper
            only_six=bool(np.max(abs(normalized[:6]))<1e-12 and normalized[6]>1e-12)
            rows.append(dict(order=p,vector_dofs=len(mass),native_gll_stiffness_relative_l2=k_error,native_gll_mass_relative_l2=m_error,
                six_rigid_motion_maximum_relative_residual=rigid_error,affine_energy_relative_error=energy_error,
                total_volume_relative_error=volume_error,exact_volume_m3=exact_volume,positive_mass=True,
                generalized_gershgorin_upper=upper,smallest_eigenvalues=smallest.tolist(),only_six_rigid_modes=only_six,
                **details,**geometry_checks,
                passed=bool(max(k_error,m_error,rigid_error,energy_error,volume_error)<1e-10 and only_six)))
            for key,value in dict(independent_stiffness=k,native_stiffness=native[0],mass=mass,native_mass=native[1],
                    physical_nodes=physical,rigid_fields=np.asarray(rigid),affine_displacement=displacement,
                    smallest_eigenvalues=smallest).items():arrays[f'p{p}_'+key]=value
        np.savez_compressed(output/'mapped-proofs.npz',**arrays)
        if (any(plan[k]!=v for k,v in identities().items()) or plan['driver_sha256_self']!=sha256_file(__file__)
                or plan['runtime']!=distribution_identity(guard)):raise ValueError('mapped reference source/runtime changed during verification')
        report.update(completed=True,passed=all(r['passed'] for r in rows),cases=rows,archive_sha256=sha256_file(output/'mapped-proofs.npz'),
            scope='Small connected nonaffine polynomial geometry and native GLL operators only; no mapped CUDA, recess mesh or wave acceptance.')
    except Exception as error:report['error']=f'{type(error).__name__}: {error}';raise
    finally:
        report['resources']=guard.report();write_json(output/'mapped-report.json',sealed(report))
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--worker',action='store_true');parser.add_argument('--plan',type=Path);args=parser.parse_args()
    output=args.output.resolve();guard=ResourceGuard(ResourceLimits(max_seconds=120,max_rss_mib=1024,max_system_memory_percent=85,min_available_mib=2048))
    if args.worker:return 0 if worker(args.plan,output,guard)['passed'] else 2
    if not output.is_relative_to(PROJECT/'outputs'):raise ValueError('mapped reference output must remain in the project')
    output.mkdir(exist_ok=False,parents=True)
    plan=dict(kind='mapped_tensor_native_reference',orders=[3,4,5],cells=[2,1,1],lengths_m=LENGTHS.tolist(),**identities(),
        driver_sha256_self=sha256_file(__file__),runtime=distribution_identity(guard),criteria=1e-10)
    snapshot_sources(output/'source-snapshot.zip',plan)
    with zipfile.ZipFile(output/'source-snapshot.zip','a',compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr('examples/mapped_tensor_reference.py',Path(__file__).read_bytes())
    plan['source_snapshot_sha256']=sha256_file(output/'source-snapshot.zip');write_json(output/'mapped-plan.json',sealed(plan))
    temporary=output/'temporary';temporary.mkdir()
    env={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1',**{name:str(temporary) for name in ('TEMP','TMP','TMPDIR')}}
    result=dict(completed=False,passed=False,ready_for_training_data=False,training_updates=0,plan_fingerprint=plan['fingerprint'])
    try:
        command=[sys.executable,'-m','examples.mapped_tensor_reference','--worker','--plan',str(output/'mapped-plan.json'),'--output',str(output/'native')]
        result['worker_run']=run_monitored(command,PROJECT,env,guard,output/'worker.log')
        if result['worker_run']['exit_code'] not in (0,2):raise RuntimeError('mapped native worker failed; inspect worker.log')
        native=checked_receipt(read_json(output/'native/mapped-report.json'))
        if (not native['completed'] or native['plan_fingerprint']!=plan['fingerprint']
                or native['archive_sha256']!=sha256_file(output/'native/mapped-proofs.npz')):raise ValueError('mapped native evidence changed')
        result.update(completed=True,passed=native['passed'],native_report_sha256=sha256_file(output/'native/mapped-report.json'))
    except Exception as error:result['error']=f'{type(error).__name__}: {error}'
    finally:
        result['resources']=guard.report();write_json(output/'monitor-result.json',sealed(result))
    print(json.dumps({k:v for k,v in result.items() if k!='resources'}));return 0 if result['passed'] else 2


if __name__=='__main__':raise SystemExit(main())
