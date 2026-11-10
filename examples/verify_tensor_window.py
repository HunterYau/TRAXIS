"""Reassemble mass and elastic action independently for a saved tensor state."""
import argparse
import json
from pathlib import Path
import platform

import numpy as np
import scipy

from examples.tensor_wave import read_wave,validate_plan,sealed
from traxis.config import Material
from traxis.gpu_tensor import streamed_action
from traxis.resources import ResourceGuard,ResourceLimits
from traxis.storage import sha256_file,write_json
from traxis.tensor_hex import line_rule
from traxis.tensor_patches import circular_patch


def cpu_mass(layout,material,guard=None):
    """Element scatter; independent of the production axis-factorized mass."""
    w=line_rule(layout.order)[1]
    local=(w[:,None,None]*w[None,:,None]*w[None,None,:]).ravel()
    local*=material.density_kg_m3*np.linalg.det(layout.jacobian)
    mass=np.zeros(layout.screen['scalar_dofs'])
    for first in range(0,layout.screen['elements'],64):
        if guard:guard.check()
        ids=layout.element_nodes(first,min(first+64,layout.screen['elements']))
        np.add.at(mass,ids.ravel(),np.broadcast_to(local,ids.shape).ravel())
    if not np.isfinite(mass).all() or np.any(mass<=0):raise ArithmeticError('independent tensor mass is invalid')
    return mass


def cpu_state(layout,material,u,v,dt,guard=None):
    n=layout.screen['vector_dofs'];u,v=np.asarray(u),np.asarray(v)
    if (u.shape!=(n,) or v.shape!=(n,) or np.iscomplexobj(u) or np.iscomplexobj(v)
            or not np.isfinite(u).all() or not np.isfinite(v).all() or not np.isfinite(dt) or dt<=0):
        raise ValueError('CPU tensor state check needs finite matching vectors and positive time increment')
    scalar=cpu_mass(layout,material,guard);mass=np.tile(scalar,3)
    ku=streamed_action(layout,material,u,guard)
    physical=float(.5*(v@(mass*v)+u@ku))
    modified=float(physical-dt**2/8*(ku@(ku/mass)))
    return dict(scalar_mass_kg=scalar,physical_energy_j=physical,modified_energy_j=modified)


def verify(root,output,guard):
    root,output=Path(root).resolve(),Path(output).resolve()
    if output.exists() and any(output.iterdir()):raise FileExistsError('CPU tensor check needs a new directory')
    output.mkdir(parents=True,exist_ok=True)
    report=dict(completed=False,passed=False,ready_for_training_data=False,training_updates=0,
        checker_sha256=sha256_file(__file__),cpu_runtime=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__))
    try:
        level=read_wave(root,guard,states=True);plan=level['plan'];cfg=plan['case'];validate_plan(plan,guard)
        layout,window=level['layout'],level['window'];ns=layout.screen['scalar_dofs']
        reference=cpu_state(layout,Material(**cfg['material']),window['final_displacement_m'],
            window['final_velocity_m_s'],level['native']['dt_s'],guard)
        mass=window['mass_diagonal_kg']
        mass_error=float(np.linalg.norm(mass.reshape(3,ns)-reference['scalar_mass_kg'][None,:])/np.linalg.norm(mass))
        energies={}
        for key in ('physical_energy_j','modified_energy_j'):
            recorded=float(window[key][-1]);expected=reference[key]
            if min(recorded,expected)<=0:raise ArithmeticError('tensor final energies must be positive')
            energies[key]=dict(recorded=recorded,cpu=expected,relative_error=abs(expected/recorded-1))
        patch_rows=[];final=[]
        for i,center in enumerate([cfg['source_center_xy_m'],*cfg['receiver_centers_xy_m']]):
            p=circular_patch(layout,center,cfg['radius_m'],guard=guard);stored=level['patches']
            same=np.array_equal(p.indices,stored[f'patch{i}_indices']) and np.array_equal(p.direction,stored[f'patch{i}_direction'])
            if not same:raise ValueError('saved physical tensor footprint changed')
            error=float(np.linalg.norm(p.weights-stored[f'patch{i}_weights'])/np.linalg.norm(p.weights))
            patch_rows.append(dict(patch=i,weight_relative_l2=error,area_relative_error=p.report['relative_area_error'],
                moment_maximum_absolute_error=max(p.report['bounded_coordinate_moment_maximum_absolute_error'],
                    p.report['radius_scaled_quadratic_maximum_absolute_error'])))
            if i:final.append(float(p.weights@window['final_displacement_m'].reshape(3,ns)[2,p.indices]))
        final=np.asarray(final);delta=float(np.linalg.norm(final-window['displacement_m'][-1]))
        scale=max(float(np.linalg.norm(final)),float(np.linalg.norm(window['displacement_m'][-1])))
        probe_error=delta/scale if scale>0 else 0.
        if scale==0 and delta!=0:raise ArithmeticError('inconsistent zero final tensor probes')
        validate_plan(plan,guard)
        if any(sha256_file(root/name)!=digest for name,digest in level['input_identities'].items()):
            raise ValueError('tensor forward inputs changed during the CPU check')
        guard.check(force=True)
        report.update(completed=True,passed=bool(mass_error<1e-12 and probe_error<1e-10
            and max(e['relative_error'] for e in energies.values())<1e-9
            and max(p['weight_relative_l2'] for p in patch_rows)<1e-12),
            plan_fingerprint=plan['fingerprint'],forward_result_fingerprint=level['parent']['fingerprint'],
            input_identities=level['input_identities'],steps=cfg['steps'],duration_s=cfg['duration_s'],vector_dofs=3*ns,
            mass_relative_l2=mass_error,energies=energies,patches=patch_rows,final_probe_relative_l2=probe_error,
            final_probe_absolute_l2_m=delta,zero_final_probe_vector=scale==0,
            scope='Independent saved-state mass/stiffness/energy and physical-port checks; no CPU trajectory replay or mesh/PZT acceptance.')
    except Exception as error:report['error']=f'{type(error).__name__}: {error}';raise
    finally:
        report['resources']=guard.report();write_json(output/'cpu-window-check.json',sealed(report))
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--max-seconds',type=float,default=120)
    parser.add_argument('--max-rss-mib',type=float,default=2048);args=parser.parse_args()
    guard=ResourceGuard(ResourceLimits(max_seconds=args.max_seconds,max_rss_mib=args.max_rss_mib,
        max_system_memory_percent=85,min_available_mib=2048))
    report=verify(args.run,args.output,guard)
    print(json.dumps({k:v for k,v in report.items() if k not in ('resources','input_identities')}))
    return 0 if report['passed'] else 2


if __name__=='__main__':raise SystemExit(main())
