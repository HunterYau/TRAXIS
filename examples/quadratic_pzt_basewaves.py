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

def benchmark(rings,sectors,layers,output,*,max_nodes=8192,max_working_mib=2560,memory_reference=None,ordering='COLAMD'):
    out=Path(output);out.mkdir(parents=True,exist_ok=True);ident=f'r{rings}-a{sectors}-t{layers}';dest=out/(ident+'.json')
    if dest.exists():raise FileExistsError('Preserving existing base-wave result; use a fresh output folder')
    geo=DiskGeometry((.5,.5),.005,.0005,'top',1);mesh=quadratic_disk(geo,rings=rings,sectors=sectors,layers=layers,max_nodes=20000)
    # Prescribing a whole mechanical face reduces fill relative to the free-body
    # modal test. Keep a separate conservative forecast for this problem.
    n=len(mesh.points_m);projected=len(mesh.tetrahedra)*2200*24*2+(max(n,500)/3367)**1.6*512*2**20
    forecast_note='Original conservative component forecast'
    if memory_reference is not None:
        measured=json.loads(Path(memory_reference).read_text())
        if measured.get('status')!='completed' or measured.get('nodes',0)<=0 or measured.get('sampled_peak_rss_mib',0)<=0:
            raise ValueError('A completed measured component reference is required')
        # Whole-process RSS extrapolation, with 50% margin and at least twice
        # the explicit sparse assembly forecast. This is not a rigorous bound.
        projected=max(2*len(mesh.tetrahedra)*2200*24*2,
            1.5*measured['sampled_peak_rss_mib']*2**20*(n/measured['nodes'])**2.5)
        forecast_note='Measured whole-process RSS scaled by nodes^2.5 with 50% margin; at least twice assembly forecast'
    if n>max_nodes or projected>min(max_working_mib*2**20,.55*psutil.virtual_memory().available):
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
                incident_displacement_scale_m=1e-9,plate_thickness_m=.001,plate_material=dict(young_pa=71e9,poisson=.33,density_kg_m3=2700),ordering=ordering,rows=[])
    try:
        result.update(projected_working_mib=projected/2**20,memory_forecast_note=forecast_note,max_nodes=max_nodes,max_working_mib=max_working_mib)
        a=assemble(mesh,mat,max_nodes=max_nodes);BR,DR,lower,upper=electrical_reduction(a,mesh.points_m)
        fixed=np.concatenate([lower+i*n for i in range(3)]);free=np.setdiff1d(np.arange(3*n),fixed)
        M=a.mass[free][:,free];K=a.stiffness[free][:,free];MF=a.mass[free][:,fixed];KF=a.stiffness[free][:,fixed]
        bf=BR[free];bc=BR[fixed];scale=np.sqrt(np.median(K.diagonal())/np.median(DR.diagonal()));host=Material(young_pa=71e9,poisson=.33,density_kg_m3=2700)
        for f in FREQUENCIES:
            print(ident,'prescribed S0/A0',f//1000,'kHz',flush=True)
            omega2=(2*np.pi*f)**2
            S=sparse.bmat([[K-omega2*M,bf*scale],[bf.T*scale,-DR*scale**2]],format='csc');lu=splu(S,permc_spec=ordering)
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
    p.add_argument('--output',type=Path,default=ROOT/'outputs/quadratic-pzt-audit/basewaves')
    p.add_argument('--max-nodes',type=int,default=8192);p.add_argument('--max-working-mib',type=float,default=2560)
    p.add_argument('--memory-reference',type=Path)
    p.add_argument('--ordering',choices=['COLAMD','MMD_AT_PLUS_A'],default='COLAMD')
    a=p.parse_args();benchmark(a.rings,a.sectors,a.layers,a.output,max_nodes=a.max_nodes,max_working_mib=a.max_working_mib,memory_reference=a.memory_reference,ordering=a.ordering)
