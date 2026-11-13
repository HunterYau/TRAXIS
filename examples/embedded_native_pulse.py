"""Replay a complete small pulse with independently assembled native matrices."""
import argparse
import json
import math
from pathlib import Path
import time

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu

from examples import mapped_patch_reference, recess_geometry_reference
from examples.embedded_wave import mesh_controls, read_wave, validate_plan
from examples.tensor_wave import PROJECT, checked_receipt, read_json, sealed
from traxis.config import CircularRecess, Plate
from traxis.embedded_recess import make_embedded_recess_mesh
from traxis.resources import ResourceGuard, ResourceLimits
from traxis.storage import sha256_file, write_json


def central_difference(stiffness, mass, load, probes, steps, dt, amplitude, guard=None):
    """Displacement recurrence with zero initial conditions and final velocity."""
    n=len(mass); previous=np.zeros(n); displacement=np.zeros(n); trace=np.zeros((steps+1,len(probes)))
    for step in range(steps):
        acceleration=(amplitude(step*dt)*load-stiffness@displacement)/mass
        following=2*displacement-previous+dt*dt*acceleration
        previous,displacement=displacement,following
        trace[step+1]=probes@displacement
        if guard and step%32==0: guard.check()
    acceleration=(amplitude(steps*dt)*load-stiffness@displacement)/mass
    velocity=(displacement-previous)/dt+.5*dt*acceleration
    return trace,displacement,velocity


def verify(run, reference, output, guard):
    run,reference,output=map(lambda p:Path(p).resolve(),(run,reference,output))
    if not all(p.is_relative_to(PROJECT/'outputs') for p in (run,reference,output)):
        raise ValueError('native pulse paths must stay in project outputs')
    output.mkdir(parents=True,exist_ok=False); started=time.perf_counter()
    report=dict(completed=False,passed=False,ready_for_training_data=False,training_updates=0,
        trace_tolerance=1e-8,state_tolerance=1e-8,operator_mass_tolerance=1e-10)
    source_hash=sha256_file(__file__)
    try:
        level=read_wave(run,guard,states=True); plan=level['plan']; case=level['case']; validate_plan(plan,guard)
        if case['scope']!='smoke' or case['order']!=4: raise ValueError('native pulse replay is bounded to the small Q4 fixture')
        rp=checked_receipt(read_json(reference/'embedded-plan.json'))
        rr=checked_receipt(read_json(reference/'native/embedded-report.json'))
        rm=checked_receipt(read_json(reference/'monitor-result.json'))
        if (not rr['passed'] or not rm['passed'] or rr['plan_fingerprint']!=rp['fingerprint']
                or rm['report_sha256']!=sha256_file(reference/'native/embedded-report.json')
                or rp['runtime']!=plan['runtime'] or rp['order']!=case['order'] or rp['defective']!=case['defective']
                or any(rp[k]!=case[k] for k in ('plate','recess','material','geometry_order','angular_cells','radial_cells','layers','outer_cell_size_m'))):
            raise ValueError('native pulse reference is incomplete or represents another physical/discrete case')
        for name in ('embedded_recess.py','compact_recess.py','recess_hex.py','mapped_tensor.py','tensor_hex.py','config.py','mapped_patches.py'):
            if rp['source_sha256'][name]!=sha256_file(PROJECT/'traxis'/name):
                raise ValueError('native pulse numerical source changed')
        for name in ('examples/mapped_patch_reference.py','examples/recess_geometry_reference.py'):
            if rp['analysis_sources'][name]!=sha256_file(PROJECT/name):
                raise ValueError('native pulse geometry/port reference changed')
        if sha256_file(reference/'source-snapshot.zip')!=rp['source_snapshot_sha256']:
            raise ValueError('native pulse source archive changed')
        mesh=make_embedded_recess_mesh(Plate(**case['plate']),CircularRecess(**case['recess']),4,guard=guard,**mesh_controls(case))
        ns=len(mesh.points_m); ndof=3*ns; width=3*5**3; entries=len(mesh.element_nodes)*width**2
        if ndof>30000 or entries>10000000: raise MemoryError('native pulse sparse assembly exceeds its declared small-case bound')
        rows=np.empty(entries,dtype=np.int32); cols=np.empty(entries,dtype=np.int32); values=np.empty(entries)
        mass=np.zeros(ndof); first=0; files={}
        for item in rr['native']['element_files']:
            guard.check(); path=reference/'native'/item['path']; files[item['path']]=item['sha256']
            if sha256_file(path)!=item['sha256']: raise ValueError('native local operator archive changed')
            with np.load(path,allow_pickle=False) as a:
                ids=a['connectivity']; xyz=a['native_gll_points']
                if np.max(abs(mesh.points_m[ids]-xyz))>1e-12: raise ValueError('native element nodes differ from the wave mesh')
                dofs=np.concatenate([ids+c*ns for c in range(3)])
                k=a['native_stiffness_nodal']; inv=np.linalg.inv(a['basis'])
                native_mass=inv.T@a['native_mass_scalar_hierarchical']@inv
                diagonal=np.diag(native_mass)
                if np.linalg.norm(native_mass-np.diag(diagonal))/np.linalg.norm(diagonal)>1e-10 or np.any(diagonal<=0):
                    raise ArithmeticError('native GLL mass is not positive diagonal within roundoff')
                count=width**2; rows[first:first+count]=np.repeat(dofs,width); cols[first:first+count]=np.tile(dofs,width)
                values[first:first+count]=k.ravel(); first+=count
                np.add.at(mass,dofs,np.tile(diagonal,3))
        if first!=entries or np.any(mass<=0): raise ArithmeticError('native assembly is incomplete')
        stiffness=sparse.coo_matrix((values,(rows,cols)),shape=(ndof,ndof)).tocsr()
        del rows,cols,values
        asymmetry=float(sparse.linalg.norm(stiffness-stiffness.T)/sparse.linalg.norm(stiffness))
        global_path=reference/'native/global-actions.npz'
        if sha256_file(global_path)!=rr['native']['global_actions_sha256']: raise ValueError('native basis transform changed')
        with np.load(global_path,allow_pickle=False) as a:
            transform=sparse.csr_matrix((a['transform_data'],a['transform_indices'],a['transform_indptr']),shape=(ns,ns))
        factor=splu(transform.T.tocsc())
        geometry,_,model=recess_geometry_reference.native_geometry(mesh,guard,return_model=True)
        vectors=[]; port_checks=[]
        for i,p in enumerate(level['native']['patches']):
            from traxis.tensor_patches import TensorPatch
            saved=level['patches']; port=TensorPatch(saved[f'patch{i}_indices'],saved[f'patch{i}_weights'],saved[f'patch{i}_direction'],p)
            native,_,detail=mapped_patch_reference.native_covector(mesh,model,port,guard)
            if not detail['passed']: raise ArithmeticError('native pulse aperture comparison failed')
            if native.shape!=(ns,): raise ValueError('native scalar aperture dimension changed')
            nodal=np.concatenate([np.zeros(2*ns),factor.solve(native)])
            vector_error=float(np.linalg.norm(nodal-port.vector(ns))/np.linalg.norm(nodal))
            if vector_error>1e-10: raise ArithmeticError('native pulse force/probe conversion failed')
            vectors.append(nodal); port_checks.append(dict(port=i,conversion_relative_l2=vector_error,native=detail))
        probes=np.array(vectors[1:]); dt=case['duration_s']/case['steps']; duration=case['cycles']/case['frequency_hz']
        def amplitude(t):
            return 0. if t<0 or t>=duration else case['force_scale_n']*.5*(1-math.cos(2*math.pi*t/duration))*math.sin(2*math.pi*case['frequency_hz']*t)
        trace,u,v=central_difference(stiffness,mass,vectors[0],probes,case['steps'],dt,amplitude,guard)
        arrays=level['window']; relative=lambda a,b:float(np.linalg.norm(a-b)/np.linalg.norm(b))
        trace_error=[relative(trace[:,i],arrays['displacement_m'][:,i]) for i in range(3)]
        errors=dict(trace_relative_l2=trace_error,final_displacement_relative_l2=relative(u,arrays['final_displacement_m']),
            final_velocity_relative_l2=relative(v,arrays['final_velocity_m_s']),mass_relative_l2=relative(mass,arrays['mass_diagonal_kg']))
        np.savez_compressed(output/'native-pulse-proof.npz',time_s=arrays['time_s'],native_trace=trace,
            gpu_trace=arrays['displacement_m'],native_final_u=u,native_final_v=v,native_mass=mass)
        validate_plan(plan,guard)
        if sha256_file(__file__)!=source_hash or any(sha256_file(run/p)!=digest for p,digest in level['input_identities'].items()):
            raise ValueError('native pulse source/input changed during replay')
        passed=max(trace_error)<1e-8 and max(errors['final_displacement_relative_l2'],errors['final_velocity_relative_l2'])<1e-8 and errors['mass_relative_l2']<1e-10 and asymmetry<1e-10
        report.update(completed=True,passed=bool(passed),errors=errors,stiffness_symmetry_relative=asymmetry,
            sparse_nnz=stiffness.nnz,native_ports=port_checks,steps=case['steps'],duration_s=case['duration_s'],
            source_sha256=source_hash,plan_fingerprint=plan['fingerprint'],input_identities=level['input_identities'],
            native_reference_plan_fingerprint=rp['fingerprint'],native_element_files=files,
            proof_sha256=sha256_file(output/'native-pulse-proof.npz'),
            scope='Complete small mechanical trajectory replay with archived independent native operators and fresh native disk integrals; no continuum/full-size/PZT or physical validation.')
    except Exception as error:
        report['error']=f'{type(error).__name__}: {error}'; raise
    finally:
        report['wall_seconds']=time.perf_counter()-started; report['resources']=guard.report()
        write_json(output/'native-pulse-check.json',sealed(report))
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--reference',type=Path,required=True); parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(); guard=ResourceGuard(ResourceLimits(max_seconds=240,max_rss_mib=1024,max_system_memory_percent=85,min_available_mib=2048))
    result=verify(args.run,args.reference,args.output,guard)
    print(json.dumps({k:result[k] for k in ('completed','passed','errors','wall_seconds','fingerprint') if k in result}))
    return 0 if result['passed'] else 2


if __name__=='__main__': raise SystemExit(main())
