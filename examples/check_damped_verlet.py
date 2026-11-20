"""Short real CUDA oscillator checks, independent of plate/data fitting."""
import json
import math
import threading
from pathlib import Path
from types import SimpleNamespace

import cupy as cp
import numpy as np

from traxis.damping import damping_parameters
from traxis.gpu_enriched_verlet import gpu_explicit_simulate
from traxis.storage import sha256_file,write_json

PROJECT=Path(__file__).resolve().parents[1]


def oscillator(coupled, alpha, steps):
    omega=2*math.pi*100000;duration=20e-6
    mass=np.array([[2.,.3],[.3,1.]]) if coupled else np.diag([2.,1.])
    inv=np.linalg.inv(mass)
    def product(a,x):return cp.stack([a[0,0]*x[0]+a[0,1]*x[1],a[1,0]*x[0]+a[1,1]*x[1]])
    class Storage:
        device_id=0;lock=threading.Lock();applications=0
        def _launch(self,u,out,stiffness_scale,mass_scale):
            out[:]=stiffness_scale*omega**2*product(mass,u);self.applications+=1
    storage=Storage();storage.cp=cp
    stiffness=SimpleNamespace(storage=storage,applications=0)
    system=SimpleNamespace(stiffness=stiffness,mass_diagonal=np.diag(mass),stable_dt_s=2/omega,
        screen={'device_storage_screen_mib':0.,'max_device_mib':100.})
    if coupled:
        def inverse(values):values[:]=product(inv,values)
        system.inverse_mass_inplace=inverse
        system.mass_quadratic=lambda values,inverse=False:cp.sum(values*product(inv if inverse else mass,values))
    result=gpu_explicit_simulate(system,steps,duration/steps,np.eye(2),np.zeros(2),
        initial_displacement=np.array([1.,-.5]),energy_stride=1,mass_damping_per_s=alpha)
    wd=math.sqrt(omega**2-alpha**2/4);t=result.time_s
    exact=np.exp(-alpha*t/2)*(np.cos(wd*t)+alpha/(2*wd)*np.sin(wd*t))
    error=float(np.max(abs(result.displacement_m[:,0]-exact)))
    growth=float(max(0.,np.max(np.diff(result.modified_energy_j)))/result.modified_energy_j[0])
    if growth>1e-12:raise AssertionError('Unforced modified energy increased')
    return dict(coupled_mass=coupled,steps=steps,alpha_per_s=alpha,max_displacement_error=error,
                max_relative_energy_growth=growth,final_energy_fraction=float(result.physical_energy_j[-1]/result.physical_energy_j[0]))


def main():
    profile=json.loads((PROJECT/'configs/pzt/aluminum-damping-ono2022.json').read_text())
    damping=damping_parameters(profile,100000);alpha=damping['mass_damping_per_s']
    rows=[]
    for coupled in (False,True):
        zero=oscillator(coupled,0.,2000);rows.append(zero)
        if abs(zero['final_energy_fraction']-1)>1e-8:raise AssertionError('Lossless regression failed')
        refinements=[oscillator(coupled,alpha,s) for s in (500,1000,2000)]
        rows.extend(refinements)
        errors=[r['max_displacement_error'] for r in refinements]
        if not errors[0]/errors[1]>3.8 or not errors[1]/errors[2]>3.8 or errors[-1]>1e-4:
            raise AssertionError('Damped solution did not converge at second order')
        if not .98<refinements[-1]['final_energy_fraction']<1:
            raise AssertionError('Unexpected damping loss')
    report=dict(passed=True,scope='Real CUDA two-DOF damped oscillator vs analytic solution; diagonal and coupled consistent mass, time refinement and zero-damping regression. Not a full-plate damping check.',
        damping=damping,checks=rows,source_sha256={name:sha256_file(PROJECT/name) for name in
        ['traxis/gpu_enriched_verlet.py','traxis/damping.py','configs/pzt/aluminum-damping-ono2022.json','examples/check_damped_verlet.py']})
    write_json(PROJECT/'docs/evidence/damped-verlet-cuda-check.json',report)
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
