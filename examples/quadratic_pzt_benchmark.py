"""Isolated PZT frequency-domain experiments; never runs a plate transient."""
import os
for _name in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):
    os.environ[_name]='2'
import argparse,json,time,threading,hashlib
from pathlib import Path
import numpy as np
import psutil
from scipy import sparse
from scipy.sparse.linalg import splu,LinearOperator,eigsh
from traxis.piezoelectric import PiezoMaterial,read_profile
from traxis.pzt_receiver import DiskGeometry
from traxis.quadratic_piezoelectric import quadratic_disk,assemble,electrical_reduction

ROOT=Path(__file__).resolve().parents[1]
FREQUENCIES=[50000,100000,150000,200000,250000,300000]

def benchmark(rings,sectors,layers,output):
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    ident=f'r{rings}-a{sectors}-t{layers}';dest=out/(ident+'.json')
    fingerprint=hashlib.sha256((ROOT/'traxis/quadratic_piezoelectric.py').read_bytes()+Path(__file__).read_bytes()).hexdigest()
    if dest.exists():
        saved=json.loads(dest.read_text())
        if saved.get('input_fingerprint')==fingerprint:return saved
        raise FileExistsError('Preserving an older experiment; choose a fresh output folder')
    mat=PiezoMaterial.from_profile(read_profile(ROOT/'configs/pzt/nce51-noliac-ver1404.json'))
    mesh=quadratic_disk(DiskGeometry((.5,.5),.005,.0005,'top',1),rings=rings,sectors=sectors,layers=layers)
    n=len(mesh.points_m);available=psutil.virtual_memory().available
    # Conservative local component cap; no plate or GPU storage is involved.
    projected=(max(n,500)/3367)**2.3*1536*2**20
    if n>5000 or projected>min(2*2**30,.55*available):
        result=dict(id=ident,status='not_run_resource_forecast',nodes=n,tets=len(mesh.tetrahedra),
                    projected_working_bytes=projected,available_host_bytes=available)
        dest.write_text(json.dumps(result,indent=2));return result
    tick=time.perf_counter();stop=threading.Event();peak=[psutil.Process().memory_info().rss]
    def sample():
        while not stop.wait(.05):peak[0]=max(peak[0],psutil.Process().memory_info().rss)
    thread=threading.Thread(target=sample,daemon=True);thread.start()
    result=dict(id=ident,input_fingerprint=fingerprint,rings=rings,sectors=sectors,layers=layers,nodes=n,tets=len(mesh.tetrahedra),
                mechanical_dofs=3*n,domain='same straight 32-sided polygon when angular=32',
                boundary='isolated mechanically free PZT, ideal open electrode port',
                force='fixed radial body-force field [x/R,y/R,0] divided by ideal disk volume; no gain fitting',
                frequencies_hz=FREQUENCIES)
    try:
        print(ident,'assembling',n,'nodes',flush=True)
        system=assemble(mesh,mat,max_nodes=5000,max_workspace_mib=768)
        K,M=system.stiffness,system.mass;BR,DR,lower,upper=electrical_reduction(system,mesh.points_m)
        dl=splu(DR);ne=DR.shape[0];nm=3*n
        scale=np.sqrt(np.median(K.diagonal())/np.median(DR.diagonal()))
        xyz=mesh.points_m;radial=np.r_[xyz[:,0]/.005,xyz[:,1]/.005,np.zeros(n)]
        load=M@radial/(mat.density_kg_m3*np.pi*.005**2*.0005)
        right=np.r_[load,np.zeros(ne)]
        def matrix(omega2):return sparse.bmat([[K-omega2*M,BR*scale],[BR.T*scale,-DR*scale**2]],format='csc')
        def op(x):return K@x+BR@dl.solve(BR.T@x)
        sigma=(2*np.pi*230000.)**2
        lu=splu(matrix(sigma))
        inv=LinearOperator(K.shape,matvec=lambda x:lu.solve(np.r_[x,np.zeros(ne)])[:nm],dtype=float)
        vals,U=eigsh(LinearOperator(K.shape,matvec=op,dtype=float),k=8,M=M,sigma=sigma,OPinv=inv,
                    tol=1e-9,maxiter=300,v0=np.random.default_rng(72).normal(size=nm))
        norm=radial@(M@radial);part=(U.T@(M@radial))**2/norm;i=int(np.argmax(part))
        mode=U[:,i];res=float(np.linalg.norm(op(mode)-vals[i]*(M@mode))/np.linalg.norm(op(mode)))
        if part[i]<.5 or res>1e-6:raise ArithmeticError('Radial mode not identified reliably')
        result.update(radial_frequency_hz=float(np.sqrt(vals[i])/(2*np.pi)),radial_participation=float(part[i]),eigen_residual=res)
        del lu,inv,U
        responses=[]
        for f in FREQUENCIES:
            print(ident,'harmonic',f//1000,'kHz',flush=True)
            A=matrix((2*np.pi*f)**2);factor=splu(A);x=factor.solve(right)
            residual=float(np.linalg.norm(A@x-right)/np.linalg.norm(right))
            if residual>1e-6:raise ArithmeticError('Harmonic residual failed')
            voltage=float(x[-1]*scale)
            responses.append(dict(frequency_hz=f,voltage_v=voltage,residual=residual,
                                  displacement_norm=float(np.linalg.norm(x[:nm]))))
            del A,factor,x
        result.update(status='completed',responses=responses)
    except Exception as exc:
        result.update(status='failed',error=repr(exc))
        raise
    finally:
        stop.set();thread.join();peak[0]=max(peak[0],psutil.Process().memory_info().rss)
        result.update(seconds=time.perf_counter()-tick,sampled_peak_rss_mib=peak[0]/2**20,
                      note='RSS sampled at 50 ms; isolated tests do not validate plate waveform or LTS stability')
        dest.write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True);return result

if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--rings',type=int,required=True);p.add_argument('--sectors',type=int,default=32)
    p.add_argument('--layers',type=int,required=True);p.add_argument('--output',type=Path,default=ROOT/'outputs/quadratic-pzt-audit/benchmark')
    a=p.parse_args();benchmark(a.rings,a.sectors,a.layers,a.output)
