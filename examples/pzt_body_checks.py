"""Small CPU/CUDA resolved-body check; never starts a full plate benchmark."""
import json
import os
from pathlib import Path

import numpy as np
from scipy.linalg import eigh, solve

from traxis.config import Plate, Material
from traxis.gpu_tensor import make_layout, build_gpu_tensor, streamed_action
from traxis.gpu_pzt_body import couple_gpu_bodies
from traxis.gpu_enriched_verlet import gpu_explicit_simulate, sparse_probe_set
from traxis.piezoelectric import PiezoMaterial, read_profile
from traxis.pzt_body import attach_disk
from traxis.pzt_receiver import DiskGeometry
from traxis.resources import ResourceGuard, ResourceLimits


def check(resolution=(12,1,2)):
    project=Path(__file__).resolve().parents[1]
    os.environ.setdefault('CUPY_CACHE_DIR',str(project/'outputs/pzt-development/pzt-body-check-cache'))
    guard=ResourceGuard(ResourceLimits(max_seconds=120,max_rss_mib=2048))
    layout=make_layout(Plate(.04,.04,.001),(2,2,1),2)
    material=PiezoMaterial.from_profile(read_profile(project/'configs/pzt/nce51-noliac-ver1404.json'))
    base=build_gpu_tensor(layout,Material(),guard=guard)
    bodies=[]; next_dof=layout.screen['vector_dofs']
    for surface in ('top','bottom'):
        body=attach_disk(layout,material,DiskGeometry((.02,.02),.005,.0005,surface,1),
            sectors=resolution[0],rings=resolution[1],layers=resolution[2],first_dof=next_dof,guard=guard,max_workspace_mib=2048)
        bodies.append(body); next_dof=body.next_dof
    coupled=couple_gpu_bodies(base,bodies,guard=guard,max_workspace_mib=1024)
    n=len(coupled.mass_diagonal); nh=len(base.mass_diagonal)
    # Independent CPU tensor contractions plus explicit joint body forms.
    k=np.zeros((n,n)); m=np.diag(np.r_[base.mass_diagonal,np.zeros(n-nh)])
    for j in range(nh):
        u=np.zeros(nh);u[j]=1
        k[:nh,j]=streamed_action(layout,Material(),u)
    for body in bodies:
        ix=np.ix_(body.indices,body.indices)
        k[ix]+=body.stiffness; m[ix]+=body.mass
    rng=np.random.default_rng(761); u=rng.normal(size=n)
    gpu_k=coupled.stiffness@u
    force_error=float(np.linalg.norm(gpu_k-k@u)/np.linalg.norm(k@u))
    storage=coupled.stiffness.storage; cp=storage.cp
    with storage.lock,cp.cuda.Device(storage.device_id):
        device_u=cp.asarray(u);storage.inverse_mass_inplace(device_u)
        inverse_error=float(np.linalg.norm(cp.asnumpy(device_u)-solve(m,u,assume_a='pos'))/
            np.linalg.norm(solve(m,u,assume_a='pos')))
    eigenvalues,modes=eigh((k+k.T)*.5,m)
    measured_bound=4/coupled.stable_dt_s**2
    if eigenvalues[-1] > measured_bound*(1+1e-10):
        raise AssertionError('stability estimate does not bound independent global eigenvalue')
    initial=modes[:,20]*1e-9; dt=.2*coupled.stable_dt_s; steps=512
    probe=sparse_probe_set([(np.array([int(np.argmax(abs(initial)))]),np.ones(1))],n)
    result=gpu_explicit_simulate(coupled,steps,dt,probe,np.zeros(n),
        initial_displacement=initial,output_stride=8,energy_stride=16,guard=guard)
    # Exact solution of a single eigenmode: independent of time-step algorithm.
    exact=initial*np.cos(np.sqrt(eigenvalues[20])*steps*dt)
    transient_error=float(np.linalg.norm(result.final_displacement_m-exact)/np.linalg.norm(initial))
    drift=float(np.ptp(result.modified_energy_j)/abs(result.modified_energy_j[0]))
    expected_energy=.5*(result.final_velocity_m_s@m@result.final_velocity_m_s+
        result.final_displacement_m@k@result.final_displacement_m)
    energy_error=float(abs(result.physical_energy_j[-1]/expected_energy-1))
    answer=dict(cpu_gpu_force_relative_error=force_error,mass_inverse_relative_error=inverse_error,
        eigenmode_transient_relative_error=transient_error,modified_energy_relative_drift=drift,
        cpu_gpu_energy_relative_error=energy_error,stable_dt_s=coupled.stable_dt_s,
        numerical_check_only=True,full_experiment_run=False,
        body_reports=[b.report for b in bodies],statistics=result.statistics)
    if force_error>1e-11 or inverse_error>1e-11 or transient_error>2e-4 or drift>1e-9 or energy_error>1e-10:
        raise AssertionError(json.dumps(answer,indent=2))
    return answer


if __name__=='__main__':
    print(json.dumps(check(),indent=2))
