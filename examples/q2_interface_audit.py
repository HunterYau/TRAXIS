"""Read-only physics investigation: fixed 8-ring Q2, small local plate models."""
import os
os.environ['OPENBLAS_NUM_THREADS']='2'
os.environ['OMP_NUM_THREADS']='2'
import gc,json,time
from pathlib import Path
import numpy as np
import psutil
from scipy import sparse
from scipy.sparse.linalg import splu,eigsh,LinearOperator
from traxis.config import Plate,Material
from traxis.gpu_tensor import make_layout,TensorLayout
from traxis.tensor_hex import line_rule
from traxis.tensor_hex import element_matrices as host_element
from traxis.spectral_piezoelectric import disk_mesh,assemble,element_matrices,basis
from traxis.quadratic_piezoelectric import electrical_reduction
from traxis.piezoelectric import PiezoMaterial,read_profile
from traxis.pzt_receiver import DiskGeometry,_point_weights
from traxis.lamb_profiles import fundamental_profile

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/q2-interface-audit-002'
FREQ=[50000,100000,150000,200000,250000,300000]


def save(name,data):
    p=OUT/name;t=p.with_suffix('.tmp');t.write_text(json.dumps(data,indent=2));t.replace(p)


def mapping(layout,xy):
    ns=layout.screen['scalar_dofs'];rr=[];cc=[];vv=[]
    for row,point in enumerate(xy):
        ids,w=_point_weights(layout,(*point,layout.plate.thickness_m))
        nz=layout.cells[2]*layout.order+1;keep=ids%nz==nz-1;ids,w=ids[keep],w[keep]
        rr.extend([row]*len(ids));cc.extend(ids);vv.extend(w)
    return sparse.coo_matrix((vv,(rr,cc)),shape=(len(xy),ns)).tocsr()


def coordinates(layout,ids):
    idx=np.array(np.unravel_index(ids,tuple(c*layout.order+1 for c in layout.cells))).T
    return np.column_stack([layout.axes_m[j][idx[:,j]] for j in range(3)])


def profile(f,mode,xyz,thickness,mat):
    p=fundamental_profile(f,thickness,mode,mat)
    # fields needs a vector of centered z coordinates.
    u=p.fields(xyz[:,2]-thickness/2)[0]
    return (1e-9*u*np.exp(1j*p.wavenumber_rad_m*xyz[:,0])[None]).ravel()


def main():
    OUT.mkdir(exist_ok=True);started=time.perf_counter()
    config=json.loads((ROOT/'outputs/simulation-runs/spectral-q2-convergence-001/plan.json').read_text())['configuration']
    hostmat=Material(**config['host_material']);mat=PiezoMaterial.from_profile(read_profile(ROOT/config['material_profile']))
    mesh=disk_mesh(DiskGeometry((.5,.5),.005,.0005,'top',1),radial_cells=8)
    print('Fixed 8-ring PZT assembly; available RAM MiB',psutil.virtual_memory().available/2**20,flush=True)
    a=assemble(mesh,mat,quadrature_order=6,max_workspace_mib=1536)
    n=len(mesh.points_m);BR,DR,bottom,top=electrical_reduction(a,mesh.points_m)
    scale=np.sqrt(np.median(a.stiffness.diagonal())/np.median(DR.diagonal()))
    def port_response(factor,count,frequency):
        # Identical 1-pC port charge excitation; normalize by solved voltage.
        rhs=np.zeros(count+DR.shape[0]);rhs[-1]=-1e-12*scale
        sol=factor.solve(rhs);voltage=sol[-1]*scale
        return dict(frequency_hz=frequency,dynamic_capacitance_pf=float(1e-12/voltage*1e12),
            free_mechanical_displacement_norm_m_per_v=float(np.linalg.norm(sol[:count])/abs(voltage)))
    psi=np.zeros(n);psi[top]=1;inside=np.setdiff1d(np.arange(n),np.r_[bottom,top])
    psi[inside]=-splu(a.dielectric[inside][:,inside].tocsc()).solve(a.dielectric[inside]@psi)
    g=a.coupling@psi;cap=float(psi@a.dielectric@psi)
    element_g=[];grad_centers=[]
    for ids in mesh.hexahedra:
        _,_,B,_,_=element_matrices(mesh.points_m[ids],mat,2,6)
        element_g.append(B@psi[ids])
        derivative=basis(2,3)[1][13]
        grad_centers.append(np.linalg.solve(derivative@mesh.points_m[ids],derivative))
    element_g=np.array(element_g)
    def strain_charge(u):
        u=u.reshape(3,n);values=np.array([np.r_[u[0,ids],u[1,ids],u[2,ids]] for ids in mesh.hexahedra])
        charge=np.sum(values*element_g,axis=1);net=charge.sum();unsigned=np.abs(charge).sum()
        gradient=np.einsum('eji,eai->eaj',np.array(grad_centers),u[:,mesh.hexahedra].transpose(1,0,2))
        strains=np.stack([gradient[:,0,0],gradient[:,1,1],gradient[:,2,2],gradient[:,1,2]+gradient[:,2,1],gradient[:,0,2]+gradient[:,2,0],gradient[:,0,1]+gradient[:,1,0]],axis=1)
        return dict(generated_charge_real_c=float(net.real),generated_charge_imag_c=float(net.imag),
            absolute_element_charge_sum_c=float(unsigned),cancellation_fraction=float(1-abs(net)/unsigned),
            charge_assembly_relative_difference=float(abs(net-g@u.ravel())/max(unsigned,1e-300)),
            displacement_rms_m=float(np.linalg.norm(u)/np.sqrt(u.size)),
            centroid_strain_rms=[float(v) for v in np.sqrt(np.mean(abs(strains)**2,axis=0))])
    # Actual production host surface interpolation, no global host allocation.
    interpolation=[];maps=[]
    for cells in (250,500,1000):
        # Coordinate metadata only, no host/GPU global vectors or operators.
        # This permits analytic interface sampling beyond full-run resource limits.
        counts=(cells,cells,1);lengths=config['plate_m'];nodes=line_rule(5)[0]
        axes=tuple(np.r_[np.concatenate([(j+(nodes[:-1]+1)/2)*L/count for j in range(count)]),L] for count,L in zip(counts,lengths))
        layout=TensorLayout(Plate(*lengths),counts,5,np.eye(3),axes,dict(scalar_dofs=int(np.prod([c*5+1 for c in counts]))))
        for center in (config['actuator_center_xy_m'],config['receiver_center_xy_m']):
            xy=mesh.points_m[bottom,:2]+center;T=mapping(layout,xy);ids=np.unique(T.indices);T=T[:,ids]
            xyz=coordinates(layout,ids);target=np.c_[xy,np.full(len(xy),.001)]
            rng=np.random.default_rng(19);x=rng.normal(size=len(ids));f=rng.normal(size=len(bottom))
            work=abs((T@x)@f-x@(T.T@f))/max(abs((T@x)@f),1e-300)
            row=dict(cells_per_axis=cells,element_width_mm=1000/cells,center=center,
                contact_nodes=len(bottom),host_surface_nodes=len(ids),work_relative_error=float(work),
                constant_error=float(np.max(abs(T@np.ones(len(ids))-1))),
                linear_error_m=float(np.max(abs(T@xyz-target))),responses=[])
            row['quadratic_error_m2']=float(np.max(abs(T@(xyz[:,0]**2+xyz[:,1]**2)-(target[:,0]**2+target[:,1]**2))))
            row['force_resultant_relative_error']=float(abs(np.sum(T.T@f)-np.sum(f))/max(np.linalg.norm(f),1e-300))
            row['force_first_moment_relative_error']=float(np.linalg.norm(xyz.T@(T.T@f)-target.T@f)/max(np.linalg.norm(target.T@f),1e-300))
            maps.append((row,T,xyz,target))
            for frequency in FREQ:
                for mode in ('S0','A0'):
                    true=profile(frequency,mode,target,.001,hostmat).reshape(3,-1)
                    interp=(T@profile(frequency,mode,xyz,.001,hostmat).reshape(3,-1).T).T
                    row['responses'].append(dict(frequency_hz=frequency,mode=mode,
                        surface_displacement_relative_percent=float(100*np.linalg.norm(interp-true)/np.linalg.norm(true))))
            interpolation.append(row)
    save('interpolation.json',interpolation);print('Production interpolation/work tests saved.',flush=True)
    if not (OUT/'receiver-interpolation.json').exists():
        # One-way receiver isolates the actual production interpolation from plate dynamics.
        fixed=np.r_[bottom,bottom+n,bottom+2*n];free=np.setdiff1d(np.arange(3*n),fixed)
        receiver=[]
        for frequency in FREQ:
            w=2*np.pi*frequency;A=a.stiffness-w*w*a.mass
            S=sparse.bmat([[A[free][:,free],BR[free]*scale],[BR[free].T*scale,-DR*scale**2]],format='csc')
            lu=splu(S,permc_spec='MMD_AT_PLUS_A')
            for mode in ('S0','A0'):
                def receive(d):
                    rhs=np.r_[-A[free][:,fixed]@d,-BR[fixed].T@d*scale]
                    sol=lu.solve(rhs.real)+1j*lu.solve(rhs.imag)
                    body=np.zeros(3*n,dtype=complex);body[fixed]=d;body[free]=sol[:len(free)]
                    return sol[-1]*scale,strain_charge(body)
                for row,T,xyz,target in maps:
                    exact,charge=receive(profile(frequency,mode,target,.001,hostmat))
                    d=(T@profile(frequency,mode,xyz,.001,hostmat).reshape(3,-1).T).T.ravel()
                    voltage,interpolated_charge=receive(d)
                    receiver.append(dict(cells=row['cells_per_axis'],center=row['center'],frequency_hz=frequency,mode=mode,
                        relative_voltage_difference_percent=float(100*abs(voltage-exact)/abs(exact)),absolute_voltage_difference_mv=float(1000*abs(voltage-exact)),
                        phase_difference_deg=float(np.degrees(np.angle(voltage/exact))),exact_charge=charge,interpolated_charge=interpolated_charge))
            del S,lu;save('receiver-interpolation.json',receiver)
    # Open-electrode free PZT: nearest four resonances to 150 kHz, not lowest modes.
    def resonances(K,M,B,name):
        sigma=(2*np.pi*150000)**2;S=sparse.bmat([[K-sigma*M,B*scale],[B.T*scale,-DR*scale**2]],format='csc')
        lu=splu(S,permc_spec='MMD_AT_PLUS_A');count=K.shape[0]
        di=splu(DR.tocsc())
        op=LinearOperator(K.shape,matvec=lambda x:K@x+B@di.solve(B.T@x),dtype=np.float64)
        inv=LinearOperator(K.shape,matvec=lambda x:lu.solve(np.r_[x,np.zeros(DR.shape[0])])[:count],dtype=np.float64)
        w,v=eigsh(op,M=M,k=4,sigma=sigma,OPinv=inv,tol=1e-8,maxiter=150,ncv=16,v0=np.random.default_rng(51).normal(size=count))
        residual=[float(np.linalg.norm(op@v[:,i]-w[i]*(M@v[:,i]))/np.linalg.norm(w[i]*(M@v[:,i]))) for i in range(4)]
        result=dict(model=name,nearest_150khz_resonance_hz=(np.sqrt(w)/(2*np.pi)).tolist(),residuals=residual)
        if name=='free-pzt':
            result['electrical_port_frf']=[]
            for f in FREQ:
                matrix=sparse.bmat([[K-(2*np.pi*f)**2*M,B*scale],[B.T*scale,-DR*scale**2]],format='csc')
                factor=splu(matrix,permc_spec='MMD_AT_PLUS_A');result['electrical_port_frf'].append(port_response(factor,count,f))
        if max(residual)>1e-6:raise ArithmeticError('Resonance eigenpair residual failed')
        save(name+'-resonances.json',result);return result
    free_path=OUT/'free-pzt-resonances.json'
    prior_free=json.loads(free_path.read_text()) if free_path.exists() else {}
    free_modes=prior_free if len(prior_free.get('electrical_port_frf',[]))==6 else resonances(a.stiffness,a.mass,BR,'free-pzt')
    # Small 12x12 mm bonded patch. Refine the patch under AND around the disk.
    # Q2 host surrogate is explicit; this does not silently replace production order 5.
    for cells in (4,6,8):
        filename='bonded-'+str(cells)+'.json'
        if (OUT/filename).exists() and len(json.loads((OUT/filename).read_text()).get('electrical_port_frf',[]))==6 and len(list(OUT.glob(f'strain-{cells}-*.npz')))==12:continue
        tick=time.perf_counter();layout=make_layout(Plate(.012,.012,.001),(cells,cells,1),2)
        xyz=layout.points();ns=len(xyz);T=mapping(layout,mesh.points_m[bottom,:2]+.006)
        freebody=np.setdiff1d(np.arange(n),bottom);nf=len(freebody)
        R=sparse.lil_matrix((n,ns+nf));R[bottom,:ns]=T;R[freebody,ns:]=sparse.eye(nf);R=R.tocsr()
        V=sparse.block_diag([R]*3,format='csr');total=V.shape[1]
        hk,_,hm=host_element(2,layout.jacobian,hostmat,'gll')
        rr=[];cc=[];vv=[];md=np.zeros(ns)
        for ids in layout.element_nodes(0,cells*cells):
            vec=np.r_[ids,ids+ns,ids+2*ns];rr.extend(np.repeat(vec,len(vec)));cc.extend(np.tile(vec,len(vec)));vv.extend(hk.ravel());md[ids]+=hm
        hostK=sparse.coo_matrix((vv,(rr,cc)),shape=(3*ns,3*ns)).tocsc()
        # Map component-major host DOFs into combined component-major coordinates.
        hi=np.r_[np.arange(ns),np.arange(ns)+ns+nf,np.arange(ns)+2*(ns+nf)]
        H=sparse.coo_matrix((np.ones(len(hi)),(np.arange(len(hi)),hi)),shape=(3*ns,total)).tocsr()
        K=(V.T@a.stiffness@V+H.T@hostK@H).tocsc()
        M=(V.T@a.mass@V+H.T@sparse.diags(np.tile(md,3))@H).tocsc();B=(V.T@BR).tocsr()
        edge=np.flatnonzero(np.any(np.isclose(xyz[:,:2],0)|np.isclose(xyz[:,:2],.012),axis=1))
        fixed=np.r_[edge,edge+ns+nf,edge+2*(ns+nf)];free=np.setdiff1d(np.arange(total),fixed)
        k=K[free][:,free];m=M[free][:,free];b=B[free]
        prior_modes=OUT/('bonded-'+str(cells)+'-resonances.json')
        modes=json.loads(prior_modes.read_text()) if prior_modes.exists() else resonances(k,m,b,'bonded-'+str(cells))
        result=dict(cells=cells,host_order=2,width_mm=12,host_dofs=3*ns,pzt_nodes=n,pzt_rings=8,
            boundary='Perimeter prescribed analytical Lamb displacement; finite patch with reflections',rows=[],
            modes=modes,electrical_port_frf=[])
        for frequency in FREQ:
            print('Bonded patch',cells,'cells;',frequency//1000,'kHz',flush=True)
            w=2*np.pi*frequency
            S=sparse.bmat([[k-w*w*m,b*scale],[b.T*scale,-DR*scale**2]],format='csc');lu=splu(S,permc_spec='MMD_AT_PLUS_A')
            result['electrical_port_frf'].append(port_response(lu,len(free),frequency))
            for mode in ('S0','A0'):
                d=profile(frequency,mode,xyz[edge],.001,hostmat)
                rhs=np.r_[-(K[free][:,fixed]-w*w*M[free][:,fixed])@d,-B[fixed].T@d*scale]
                sol=lu.solve(rhs.real)+1j*lu.solve(rhs.imag);u=np.zeros(total,dtype=complex);u[fixed]=d;u[free]=sol[:len(free)]
                residual=float(np.linalg.norm(S@sol-rhs)/np.linalg.norm(rhs))
                if residual>1e-8:raise ArithmeticError('Harmonic residual failed')
                voltage=sol[-1]*scale;body=V@u
                displacement=body.reshape(3,n)[:,mesh.hexahedra].transpose(1,0,2)
                gradient=np.einsum('eji,eai->eaj',np.array(grad_centers),displacement)
                strains=np.stack([gradient[:,0,0],gradient[:,1,1],gradient[:,2,2],gradient[:,1,2]+gradient[:,2,1],gradient[:,0,2]+gradient[:,2,0],gradient[:,0,1]+gradient[:,1,0]],axis=1)
                np.savez_compressed(OUT/f'strain-{cells}-{frequency//1000}-{mode}.npz',centroid_xyz_m=mesh.points_m[mesh.hexahedra].mean(axis=1),strain=strains,
                    generated_element_charge_c=np.sum(np.concatenate([displacement[:,i,:] for i in range(3)],axis=1)*element_g,axis=1))
                result['rows'].append(dict(frequency_hz=frequency,mode=mode,voltage_real_v=float(voltage.real),voltage_imag_v=float(voltage.imag),
                    residual=residual,**strain_charge(body)))
            del S,lu;save(filename,result)
        result.update(seconds=time.perf_counter()-tick,status='completed');save(filename,result)
        del K,M,B,k,m,b,R,V,hostK,H;gc.collect()
    save('status.json',dict(status='completed',seconds=time.perf_counter()-started,free_modes=free_modes,process_peak_rss_bytes=getattr(psutil.Process().memory_info(),'peak_wset',None)))


if __name__=='__main__':main()
