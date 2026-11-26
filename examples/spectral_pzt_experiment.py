"""Isolated same-polygon SEM/P2 receiver benchmark, no full plate or RunPod."""
import os
os.environ['OPENBLAS_NUM_THREADS']='2'
os.environ['OMP_NUM_THREADS']='2'
import argparse
import hashlib
import json
import threading
import time
from pathlib import Path
import numpy as np
import psutil
from scipy import sparse
from scipy.sparse.linalg import splu
from traxis.spectral_piezoelectric import disk_mesh,assemble
from traxis.quadratic_piezoelectric import electrical_reduction
from traxis.piezoelectric import PiezoMaterial,read_profile
from traxis.pzt_receiver import DiskGeometry
from traxis.lamb_profiles import fundamental_profile
from traxis.config import Material

ROOT=Path(__file__).resolve().parents[1]
FREQUENCIES=[50000,100000,150000,200000,250000,300000]


def run(order,side,radial,layers,output,*,quadrature_order=6,max_equations=20000,frequencies=None):
    ident=f'q{order}-s{side}-r{radial}-z{layers}-g{quadrature_order}'
    frequencies=FREQUENCIES if frequencies is None else list(frequencies)
    if not frequencies or any(f not in FREQUENCIES for f in frequencies):raise ValueError('Unsupported frequency selection')
    if frequencies!=FREQUENCIES:ident+='-f'+'-'.join(str(f//1000) for f in frequencies)
    path=output/(ident+'.json')
    if path.exists():
        prior=json.loads(path.read_text())
        if prior['status'] in ('completed','not_run_resource_budget'):return prior
        raise FileExistsError('Preserving an unfinished attempt; use another output folder')
    result=dict(id=ident,order=order,side_cells=side,radial_cells=radial,layers=layers,
        quadrature_order=quadrature_order,frequencies_hz=frequencies,rows=[],status='running',precision='float64',mass='full consistent')
    started=time.perf_counter();peak=[psutil.Process().memory_info().rss];stop=threading.Event()
    def sample():
        while not stop.wait(.025):peak[0]=max(peak[0],psutil.Process().memory_info().rss)
    thread=threading.Thread(target=sample,daemon=True);thread.start()
    try:
        geo=DiskGeometry((.5,.5),.005,.0005,'top',1)
        mesh=disk_mesh(geo,side_cells=side,radial_cells=radial,layers=layers,order=order)
        n=len(mesh.points_m);lower=np.flatnonzero(abs(mesh.points_m[:,2])<1e-14)
        equations=3*(n-len(lower))+n-2*len(lower)+1
        result.update(nodes=n,hexahedra=len(mesh.hexahedra),equations=equations,boundary_sides=mesh.boundary_sides)
        if equations>max_equations:raise MemoryError('Component equation budget exceeded; no matrices assembled')
        mat=PiezoMaterial.from_profile(read_profile(ROOT/'configs/pzt/nce51-noliac-ver1404.json'))
        tick=time.perf_counter();a=assemble(mesh,mat,quadrature_order=quadrature_order,max_workspace_mib=1536)
        result['assembly_seconds']=time.perf_counter()-tick
        result['volume_m3']=a.volume_m3
        result['sparse_matrix_bytes']=sum(m.data.nbytes+m.indices.nbytes+m.indptr.nbytes for m in [a.stiffness,a.mass,a.coupling,a.dielectric])
        BR,DR,lower,upper=electrical_reduction(a,mesh.points_m)
        fixed=np.concatenate([lower+i*n for i in range(3)]);free=np.setdiff1d(np.arange(3*n),fixed)
        K=a.stiffness[free][:,free];M=a.mass[free][:,free]
        KF=a.stiffness[free][:,fixed];MF=a.mass[free][:,fixed]
        bf=BR[free];bc=BR[fixed];scale=np.sqrt(np.median(K.diagonal())/np.median(DR.diagonal()))
        material=Material(young_pa=71e9,poisson=.33,density_kg_m3=2700)
        for f in frequencies:
            print(ident,f//1000,'kHz',flush=True);tick=time.perf_counter()
            S=sparse.bmat([[K-(2*np.pi*f)**2*M,bf*scale],[bf.T*scale,-DR*scale**2]],format='csc')
            lu=splu(S,permc_spec='MMD_AT_PLUS_A')
            factor_seconds=time.perf_counter()-tick
            for mode in ['S0','A0']:
                p=fundamental_profile(f,.001,mode,material);u=p.fields(np.array([.0005]))[0][:,0]
                displacement=(1e-9*u[:,None]*np.exp(1j*p.wavenumber_rad_m*mesh.points_m[lower,0])).ravel()
                rhs=np.r_[-(KF-(2*np.pi*f)**2*MF)@displacement,-bc.T@displacement*scale]
                sol=lu.solve(rhs.real)+1j*lu.solve(rhs.imag)
                residual=float(np.linalg.norm(S@sol-rhs)/np.linalg.norm(rhs))
                if residual>1e-9:raise ArithmeticError('Hex harmonic residual failed')
                voltage=sol[-1]*scale
                result['rows'].append(dict(frequency_hz=f,mode=mode,voltage_real_v=float(voltage.real),
                    voltage_imag_v=float(voltage.imag),residual=residual,factor_seconds=factor_seconds))
            del S,lu
            path.write_text(json.dumps(result,indent=2))
        result['status']='completed'
    except MemoryError as exc:result.update(status='not_run_resource_budget',error=str(exc))
    except Exception as exc:result.update(status='failed',error=repr(exc));raise
    finally:
        stop.set();thread.join();result.update(seconds=time.perf_counter()-started,sampled_peak_rss_mib=peak[0]/2**20)
        path.write_text(json.dumps(result,indent=2));print(ident,result['status'],result['seconds'],'s',flush=True)
    return result


def compare(reference,tested):
    def voltage(row,mode):return np.array([complex(v['voltage_real_v'],v['voltage_imag_v']) for v in row['rows'] if v['mode']==mode])
    result={}
    for mode in ['S0','A0']:
        selected={v['frequency_hz'] for v in tested['rows'] if v['mode']==mode}
        subset=dict(reference,rows=[v for v in reference['rows'] if v['frequency_hz'] in selected])
        a,b=voltage(subset,mode),voltage(tested,mode)
        result[mode]=dict(relative_l2_percent=float(100*np.linalg.norm(b-a)/np.linalg.norm(a)),
            per_frequency_percent=(100*abs(b-a)/abs(a)).tolist(),
            phase_change_deg=np.degrees(np.angle(b/a)).tolist(),
            absolute_difference_mv=(1000*abs(b-a)).tolist())
    return result


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--order',type=int,default=2);p.add_argument('--side',type=int,default=16)
    p.add_argument('--radial',type=int,default=1);p.add_argument('--layers',type=int,default=1)
    p.add_argument('--frequencies',nargs='+',type=int);p.add_argument('--max-equations',type=int,default=20000);p.add_argument('--quadrature',type=int,default=6);p.add_argument('--output',type=Path,default=ROOT/'outputs/spectral-pzt-experiment')
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    result=run(args.order,args.side,args.radial,args.layers,args.output,quadrature_order=args.quadrature,max_equations=args.max_equations,frequencies=args.frequencies)
    reference=json.loads((ROOT/'docs/evidence/quadratic-pzt-dimensions/comparisons.json').read_text())['reference']
    if result['status']=='completed':print(json.dumps(compare(reference,result),indent=2))


if __name__=='__main__':main()
