"""Bounded PZT component verification; no experimental plate reruns or training."""
import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np
from scipy.linalg import eigh

from traxis.config import Material,Plate
from traxis.geometry import structured_mesh
from traxis.gpu_enriched_verlet import gpu_explicit_simulate,sparse_probe_set
from traxis.gpu_tensor import make_layout,build_gpu_tensor
from traxis.piezoelectric import (PiezoMaterial,Readout,assemble_piezoelectric,
    electrode_port,planar_from_profile,read_profile)
from traxis.pzt_receiver import DiskGeometry,disk_receiver
from traxis.resources import ResourceGuard,ResourceLimits
from traxis.storage import atomic_write,canonical_hash,sha256_file,write_json
from traxis.tensor_hex import make_grid,assemble_tensor

PROJECT=Path(__file__).resolve().parents[1]


def native_check(guard):
    from examples.piezoelectric_reference import native_operators,port_checks,circuit_check
    mat=PiezoMaterial.from_profile(read_profile(PROJECT/'configs/pzt/nce51-noliac-ver1404.json'))
    mesh=structured_mesh(Plate(.01,.008,.0005),(1,1,2))
    system=assemble_piezoelectric(mesh,mat,guard=guard)
    c,e,eps=mat.stress_charge
    native=native_operators(mesh,(c,e,eps,mat.density_kg_m3),guard)
    arrays=(system.stiffness,system.mass,system.coupling,system.dielectric)
    errors=[float(np.linalg.norm(a-b)/np.linalg.norm(b)) for a,b in zip(arrays,native[:4])]
    if max(errors)>1e-11:raise ArithmeticError('independent native coupled operators disagree')
    bottom=np.flatnonzero(mesh.points_m[:,2]==0);top=np.flatnonzero(mesh.points_m[:,2]==.0005)
    legacy,checks=port_checks((*arrays,system.volume_m3),native,bottom,top)
    new=electrode_port(system,bottom,top);free=legacy['free']
    port_error=float(np.linalg.norm(new.stiffness_sc[np.ix_(free,free)]-legacy['k'])/np.linalg.norm(legacy['k']))
    if port_error>1e-12:raise ArithmeticError('electrode condensation changed the coupled reference')
    circuit,proof=circuit_check(legacy,1e-4,0.,64)
    if circuit['total_energy_balance_relative_error']>1e-11:raise ArithmeticError('coupled circuit power balance failed')
    return dict(operator_relative_errors=errors,electrode_checks=checks,port_stiffness_relative_error=port_error,
                actual_material_circuit=circuit,fixture='small NCE51 rectangular FEM block, not a physical plate comparison')


def gpu_check(output,guard):
    os.environ.setdefault('CUPY_CACHE_DIR',str(output/'cupy-cache'))
    plate=Plate(.04,.03,.003175);cells=(2,2,1);order=3
    layout=make_layout(plate,cells,order);grid=make_grid(plate,cells,order)
    cpu=assemble_tensor(grid,Material(),guard=guard);system=build_gpu_tensor(layout,Material(),guard=guard)
    receivers=[]
    for filename,radius,thickness in [('steminc-smd07t02s412.json',.0035,.0002),('nce51-noliac-ver1404.json',.005,.0005)]:
        p=planar_from_profile(read_profile(PROJECT/'configs/pzt'/filename))
        receivers.append(disk_receiver(layout,p,DiskGeometry((.02,.015),radius,thickness,'top',1),guard=guard))
    ns=layout.screen['scalar_dofs'];u0=np.zeros((3,ns));u0[0]=1e-6*grid.points_m[:,0];u0[1]=-.5e-6*grid.points_m[:,1]
    mass=cpu.mass_kg;inv=1/np.sqrt(mass)
    eigen,vectors=eigh(cpu.stiffness.toarray()*inv[:,None]*inv[None,:])
    if eigen.min() < -1e-10*eigen.max():raise ArithmeticError('reference has a negative elastic mode')
    omega=np.sqrt(np.maximum(eigen,0.));coeff=vectors.T@(u0.ravel()/inv)
    dt=min(1e-10,system.stable_dt_s/100);steps=512
    probes=sparse_probe_set([(r.indices,r.charge_weights_c_m) for r in receivers],3*ns)
    result=gpu_explicit_simulate(system,steps,dt,probes,np.zeros(3*ns),
        initial_displacement=u0.ravel(),energy_stride=16,guard=guard)
    exact_modes=coeff[:,None]*np.cos(omega[:,None]*result.time_s)
    exact_u=inv[:,None]*(vectors@exact_modes)
    exact_q=np.column_stack([r.charge_weights_c_m@exact_u[r.indices] for r in receivers])
    relative=float(np.linalg.norm(result.displacement_m-exact_q)/np.linalg.norm(exact_q))
    if relative>1e-7:raise ArithmeticError('CUDA charge traces disagree with independent exact normal modes')
    final_q=np.array([r.strain_charge(result.final_displacement_m) for r in receivers])
    final_error=float(np.linalg.norm(final_q-result.displacement_m[-1])/np.linalg.norm(final_q))
    if final_error>1e-12:raise ArithmeticError('CUDA sparse charge output has wrong units/order')
    readouts=[r.record(result.time_s,result.displacement_m[:,j],Readout(None,20e-12)) for j,r in enumerate(receivers)]
    charge_balance=max(float(np.max(abs(x['charge_balance_c']))) for x in readouts)
    atomic_write(output/'gpu-receiver-traces.npz',lambda f:np.savez_compressed(f,
        time_s=result.time_s,strain_charge_c=result.displacement_m,exact_charge_c=exact_q,
        voltage_v=np.column_stack([r['voltage_v'] for r in readouts])))
    return dict(exact_modal_charge_relative_error=relative,final_cpu_charge_relative_error=final_error,
                maximum_free_charge_balance_c=charge_balance,steps=steps,dt_s=dt,
                vector_dofs=3*ns,receivers=[r.report for r in receivers],
                fixture='manufactured small elastic plate, two alternative one-way PZT observers; no ground-truth comparison',
                readout='ideal open input with explicit 20 pF parallel test capacitance, not a known DAQ')


def main():
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--native',action='store_true');parser.add_argument('--gpu',action='store_true')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    guard=ResourceGuard(ResourceLimits(max_seconds=180,max_rss_mib=1536))
    result=dict(completed=False,ready_for_training_data=False,experimental_trials=0)
    try:
        guard.check(force=True)
        if args.native:result['native']=native_check(guard)
        if args.gpu:result['gpu']=gpu_check(args.output,guard)
        result['source_sha256']={str(p.relative_to(PROJECT)).replace('\\','/'):sha256_file(p) for p in
            [*sorted((PROJECT/'traxis').glob('*.py')),Path(__file__),*sorted((PROJECT/'configs/pzt').glob('*.json'))]}
        result['runtime']=dict(python=sys.version,numpy=np.__version__)
        result['completed']=True
    finally:
        result['resources']=guard.report();result['fingerprint']=canonical_hash(result)
        write_json(args.output/'checks.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('resources','source_sha256')},indent=2))


if __name__=='__main__':main()
