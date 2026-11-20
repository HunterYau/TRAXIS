"""Real CUDA coupled resistance check against an independent CPU ODE solve."""
import json
import threading
from pathlib import Path
from types import SimpleNamespace
import cupy as cp
import numpy as np
from scipy.integrate import solve_ivp
from traxis.gpu_enriched_verlet import gpu_explicit_simulate
from traxis.storage import write_json,sha256_file

PROJECT=Path(__file__).resolve().parents[1]


def check(steps,resistance):
    mass=np.array([[2.,.3],[.3,1.]])*1e-6;inverse=np.linalg.inv(mass)
    omega=2*np.pi*100000;cap=1.2e-9;g=np.array([.004,-.002])
    k=omega**2*mass+np.outer(g,g)/cap;load=np.array([1e-4,0.])
    duration=20e-6;alpha=700.
    def product(a,x):return cp.stack((a[0,0]*x[0]+a[0,1]*x[1],a[1,0]*x[0]+a[1,1]*x[1]))
    class Storage:
        device_id=0;lock=threading.Lock();applications=0
        def _launch(self,u,out,stiffness_scale,mass_scale):out[:]=product(k,u);self.applications+=1
    storage=Storage();storage.cp=cp
    def mass_inverse(values):values[:]=product(inverse,values)
    system=SimpleNamespace(stiffness=SimpleNamespace(storage=storage,applications=0),mass_diagonal=np.diag(mass),
        stable_dt_s=2/np.sqrt(np.linalg.eigvals(inverse@k).max()),screen={'device_storage_screen_mib':0.,'max_device_mib':100.},
        inverse_mass_inplace=mass_inverse,mass_quadratic=lambda v,inverse=False:cp.sum(v*product(np.linalg.inv(mass) if inverse else mass,v)))
    initial=np.array([1e-6,-.5e-6]);port=dict(indices=np.arange(2),coupling=g,capacitance_f=cap,resistance_ohm=resistance,probe_index=0)
    force=lambda t:np.sin(2*np.pi*120000*t) if t<5e-6 else 0.
    result=gpu_explicit_simulate(system,steps,duration/steps,np.array([g/cap]),load,force=force,
        initial_displacement=initial,mass_damping_per_s=alpha,electrical_ports=[port],energy_stride=1)
    def rhs(t,y):
        u=y[:2];v=y[2:4];q=y[4]
        return np.r_[v,inverse@(load*force(t)-k@u-g*q/cap)-alpha*v,
                     0. if resistance is None else -(q+g@u)/(resistance*cap)]
    reference=solve_ivp(rhs,[0,duration],np.r_[initial,0.,0.,0.],method='DOP853',
        rtol=1e-11,atol=1e-18,t_eval=result.time_s,max_step=duration/1000)
    exact=(g@reference.y[:2]+reference.y[4])/cap
    error=float(np.max(abs(result.displacement_m[:,0]-exact))/np.max(abs(exact)))
    tail=result.modified_energy_j[result.energy_time_s>6e-6]
    growth=float(max(0.,np.max(np.diff(tail)))/max(abs(tail[0]),1e-300))
    return dict(steps=steps,resistance_ohm=resistance,relative_voltage_error=error,post_drive_max_relative_energy_growth=growth)


def main():
    rows=[check(steps,r) for r in (None,1e6,1e3) for steps in (1000,2000)]
    for coarse,fine in zip(rows[::2],rows[1::2]):
        if not coarse['relative_voltage_error']/fine['relative_voltage_error']>3.5 or fine['relative_voltage_error']>1e-4:
            raise AssertionError('Coupled voltage did not converge at second order')
        if fine['post_drive_max_relative_energy_growth']>1e-8:raise AssertionError('Passive system gained energy')
    report=dict(passed=True,scope='Real CUDA transient voltage vs independent coupled mechanical/electrical SciPy ODE; forced pulse and free decay, open/1 Mohm/1 kohm loads, coupled mass, temporal refinement',
        checks=rows,source_sha256={name:sha256_file(PROJECT/name) for name in ['traxis/gpu_enriched_verlet.py','traxis/pzt_body.py','traxis/pzt_bond.py','examples/check_loaded_pzt.py']})
    write_json(PROJECT/'docs/evidence/loaded-pzt-cuda-check.json',report)
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
