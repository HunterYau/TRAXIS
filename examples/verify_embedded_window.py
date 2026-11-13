"""Independent CPU readback of a completed embedded mechanical window."""
import argparse
import json
from pathlib import Path

import numpy as np

from examples.embedded_wave import mesh_controls, read_wave, validate_plan
from examples.tensor_wave import PROJECT, sealed
from traxis.config import CircularRecess, Material, Plate
from traxis.embedded_recess import make_embedded_recess_mesh
from traxis.mapped_tensor import action, build_system
from traxis.resources import ResourceGuard, ResourceLimits
from traxis.storage import sha256_file, write_json


def verify(root, output, guard):
    root,output=Path(root).resolve(),Path(output).resolve()
    if not root.is_relative_to(PROJECT/'outputs') or not output.is_relative_to(PROJECT/'outputs'):
        raise ValueError('embedded CPU evidence must stay in project outputs')
    output.mkdir(parents=True,exist_ok=False)
    report=dict(completed=False,passed=False,ready_for_training_data=False,training_updates=0)
    try:
        level=read_wave(root,guard,states=True); plan=level['plan']; case=level['case']; arrays=level['window']
        validate_plan(plan,guard)
        if sha256_file(root/'source-snapshot.zip')!=plan['source_snapshot_sha256']:
            raise ValueError('embedded source snapshot changed')
        guard.check(force=True)
        mesh=make_embedded_recess_mesh(Plate(**case['plate']),CircularRecess(**case['recess']),
            case['order'],guard=guard,**mesh_controls(case))
        cpu=build_system(mesh,Material(**case['material']),guard,max_bytes=case['max_operator_bytes'],max_batch_vectors=1)
        u,v,mass=(arrays[k] for k in ('final_displacement_m','final_velocity_m_s','mass_diagonal_kg'))
        if np.any(mass<=0): raise ValueError('embedded saved mass is nonpositive')
        force=action(cpu,u,guard)
        relative=lambda a,b:float(np.linalg.norm(a-b)/np.linalg.norm(b))
        force_error=relative(arrays['final_gpu_force_n'],force)
        mass_error=relative(mass,cpu.mass_kg)
        physical=float((v@(cpu.mass_kg*v)+u@force)/2)
        dt=case['duration_s']/case['steps']
        modified=physical-dt**2/8*float(force@(force/cpu.mass_kg))
        if physical<=0 or modified<=0: raise ArithmeticError('embedded final CPU energies are nonpositive')
        physical_error=abs(arrays['physical_energy_j'][-1]/physical-1)
        modified_error=abs(arrays['modified_energy_j'][-1]/modified-1)
        port_rows=[]; traces=[]; ns=len(mesh.points_m)
        for i,center in enumerate([case['source_center_xy_m'],*case['receiver_centers_xy_m']]):
            guard.check(); port=level['patches']; ids=port[f'patch{i}_indices']; weights=port[f'patch{i}_weights']; direction=port[f'patch{i}_direction']
            if (ids.dtype.kind!='i' or len(np.unique(ids))!=len(ids) or ids.min()<0 or ids.max()>=ns
                    or not np.array_equal(direction,[0.,0.,1.]) or np.max(abs(mesh.points_m[ids,2]-case['plate']['thickness_m']))>1e-14):
                raise ValueError('embedded disk topology, direction or top surface changed')
            force_moment_error=max(abs(float(weights.sum())-1),float(np.max(abs(weights@mesh.points_m[ids,:2]-center)))/case['radius_m'])
            if force_moment_error>1e-9: raise ArithmeticError('embedded disk force/first moments changed')
            port_rows.append(dict(port=i,force_and_first_moment_error=force_moment_error,negative_weights=int(np.count_nonzero(weights<0))))
            if i: traces.append(float(direction@u.reshape(3,ns)[:,ids]@weights))
        probe_error=relative(np.asarray(traces),arrays['displacement_m'][-1])
        after=arrays['modified_energy_j'][arrays['energy_time_s']>case['cycles']/case['frequency_hz']+2*dt]
        if len(after)<2 or after.mean()<=0: raise ArithmeticError('embedded postburst energy is missing/nonpositive')
        variation=float(np.ptp(after)/abs(after.mean()))
        if not all(np.linalg.norm(arrays['displacement_m'],axis=0)>0): raise ArithmeticError('embedded receiver traces are zero')
        if any(sha256_file(root/name)!=digest for name,digest in level['input_identities'].items()):
            raise ValueError('embedded inputs changed during CPU readback')
        validate_plan(plan,guard)
        passed=max(force_error,mass_error,physical_error,modified_error,probe_error)<case['operator_tolerance'] and variation<case['energy_tolerance']
        report.update(completed=True,passed=bool(passed),plan_fingerprint=plan['fingerprint'],
            forward_result_fingerprint=level['parent']['fingerprint'],input_identities=level['input_identities'],
            checker_sha256=sha256_file(__file__),source_sha256=plan['source_sha256'],
            final_force_relative_l2=force_error,mass_relative_l2=mass_error,
            final_physical_energy_relative_error=physical_error,final_modified_energy_relative_error=modified_error,
            final_probe_relative_l2=probe_error,postburst_modified_energy_relative_range=variation,ports=port_rows,
            steps=case['steps'],duration_s=case['duration_s'],scope=case['scope'],
            note='Independent CPU final force, mass, energies and trace contraction; does not certify full trajectory, convergence or physical PZT response.')
    except Exception as error:
        report['error']=f'{type(error).__name__}: {error}'; raise
    finally:
        report['resources']=guard.report(); write_json(output/'cpu-window-check.json',sealed(report))
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True); args=parser.parse_args()
    guard=ResourceGuard(ResourceLimits(max_seconds=240,max_rss_mib=2048,max_system_memory_percent=85,min_available_mib=2048))
    report=verify(args.run,args.output,guard)
    print(json.dumps({k:v for k,v in report.items() if k not in ('resources','source_sha256','input_identities','ports')}))
    return 0 if report['passed'] else 2


if __name__=='__main__': raise SystemExit(main())
