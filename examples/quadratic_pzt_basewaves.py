"""Small receiver tests with prescribed S0/A0 plate-surface displacement.

No plate is simulated: its analytical Lamb displacement is prescribed at
every bonded PZT node. Plate backreaction is absent in this diagnostic.
"""
from examples.quadratic_pzt_benchmark import (np,psutil,sparse,splu,time,threading,json,
    ROOT,FREQUENCIES,PiezoMaterial,read_profile,quadratic_disk,DiskGeometry,assemble,electrical_reduction)
import argparse
from pathlib import Path
from traxis.lamb_profiles import fundamental_profile
from traxis.config import Material

def benchmark(rings,sectors,layers,output):
    out=Path(output);out.mkdir(parents=True,exist_ok=True);ident=f'r{rings}-a{sectors}-t{layers}';dest=out/(ident+'.json')
    if dest.exists():raise FileExistsError('Preserving existing base-wave result; use a fresh output folder')
    geo=DiskGeometry((.5,.5),.005,.0005,'top',1);mesh=quadratic_disk(geo,rings=rings,sectors=sectors,layers=layers,max_nodes=20000)
    # Prescribing a whole mechanical face reduces fill relative to the free-body
    # modal test. Keep a separate conservative forecast for this problem.
    n=len(mesh.points_m);projected=len(mesh.tetrahedra)*2200*24*2+(max(n,500)/3367)**1.6*512*2**20
    if n>8192 or projected>min(2.5*2**30,.55*psutil.virtual_memory().available):
        result=dict(id=ident,status='not_run_resource_forecast',nodes=n,tets=len(mesh.tetrahedra),
                    projected_working_mib=projected/2**20,available_host_mib=psutil.virtual_memory().available/2**20,
                    note='Conservative experimental budget, not proof the component cannot fit; no matrices assembled')
        dest.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2));return result
    mat=PiezoMaterial.from_profile(read_profile(ROOT/'configs/pzt/nce51-noliac-ver1404.json'))
    tick=time.perf_counter();peak=[psutil.Process().memory_info().rss];stop=threading.Event()
    def sample():
        while not stop.wait(.05):peak[0]=max(peak[0],psutil.Process().memory_info().rss)
    t=threading.Thread(target=sample,daemon=True);t.start()
    result=dict(id=ident,nodes=n,tets=len(mesh.tetrahedra),status='running',boundary='prescribed analytical S0/A0 surface displacement on bottom PZT face, open port',
                incident_displacement_scale_m=1e-9,plate_thickness_m=.001,plate_material=dict(young_pa=71e9,poisson=.33,density_kg_m3=2700),rows=[])
    try:
        a=assemble(mesh,mat,max_nodes=8192);BR,DR,lower,upper=electrical_reduction(a,mesh.points_m)
        fixed=np.concatenate([lower+i*n for i in range(3)]);free=np.setdiff1d(np.arange(3*n),fixed)
        M=a.mass[free][:,free];K=a.stiffness[free][:,free];MF=a.mass[free][:,fixed];KF=a.stiffness[free][:,fixed]
        bf=BR[free];bc=BR[fixed];scale=np.sqrt(np.median(K.diagonal())/np.median(DR.diagonal()));host=Material(young_pa=71e9,poisson=.33,density_kg_m3=2700)
        for f in FREQUENCIES:
            print(ident,'prescribed S0/A0',f//1000,'kHz',flush=True)
            omega2=(2*np.pi*f)**2
            S=sparse.bmat([[K-omega2*M,bf*scale],[bf.T*scale,-DR*scale**2]],format='csc');lu=splu(S)
            for mode in ('S0','A0'):
                profile=fundamental_profile(f,.001,mode,host);u=profile.fields(np.array([.0005]))[0][:,0]
                displacement=(1e-9*u[:,None]*np.exp(1j*profile.wavenumber_rad_m*mesh.points_m[lower,0])).ravel()
                rhs=np.r_[-(KF-omega2*MF)@displacement,-bc.T@displacement*scale]
                sol=lu.solve(rhs.real)+1j*lu.solve(rhs.imag)
                residual=float(np.linalg.norm(S@sol-rhs)/np.linalg.norm(rhs))
                if residual>1e-7:raise ArithmeticError('Base-wave residual failed')
                voltage=sol[-1]*scale
                result['rows'].append(dict(frequency_hz=f,mode=mode,voltage_real_v=float(voltage.real),voltage_imag_v=float(voltage.imag),residual=residual,wavenumber_rad_m=profile.wavenumber_rad_m))
            del S,lu
        result['status']='completed'
    except Exception as exc:result.update(status='failed',error=repr(exc));raise
    finally:
        stop.set();t.join();result.update(seconds=time.perf_counter()-tick,sampled_peak_rss_mib=peak[0]/2**20)
        dest.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--rings',type=int,required=True);p.add_argument('--sectors',type=int,required=True);p.add_argument('--layers',type=int,required=True)
    p.add_argument('--output',type=Path,default=ROOT/'outputs/quadratic-pzt-audit/basewaves');a=p.parse_args();benchmark(a.rings,a.sectors,a.layers,a.output)
