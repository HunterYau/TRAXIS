"""Independently check saved enriched GPU states using exact local CPU matrices."""

import argparse
import json
from pathlib import Path
import platform

import numpy as np
import scipy

from examples.enriched_time_refinement import read_level
from examples.matrixfree_step import _json,capacity_context,load_mesh
from traxis.elmer_plate_refinement import reference_source_hashes
from traxis.enriched_factor import streamed_action
from traxis.enriched_tet import WEIGHTS,build_topology
from traxis.resources import ResourceGuard,ResourceLimits
from traxis.storage import canonical_hash,sha256_file,write_json


def cpu_state_reference(mesh,material,settings,mapping,u,v,dt_s,guard=None):
    topology = build_topology(mesh,max_vertices=settings.max_vertices,max_tetrahedra=settings.max_tetrahedra,
        max_dofs=settings.max_dofs,max_workspace_mib=settings.max_topology_mib)
    n = len(topology.points_m)
    mapping,u,v = np.asarray(mapping),np.asarray(u),np.asarray(v)
    if (mapping.shape!=(n,) or mapping.dtype.kind not in 'iu'
            or not np.array_equal(np.sort(mapping),np.arange(n)) or u.shape!=v.shape or u.shape!=(3*n,)
            or np.iscomplexobj(u) or np.iscomplexobj(v) or not np.isfinite(u).all() or not np.isfinite(v).all()
            or not np.isfinite(dt_s) or dt_s<=0):
        raise ValueError('independent state check needs a bijective map and finite matching states')
    corners = mesh.points_m[mesh.tetrahedra]
    determinants = abs(np.linalg.det((corners[:,:3]-corners[:,3:4]).transpose(0,2,1)))
    mass = np.zeros(n)
    np.add.at(mass,topology.element_nodes.ravel(),(material.density_kg_m3*determinants[:,None]*WEIGHTS).ravel())
    if not np.isfinite(mass).all() or np.any(mass<=0):
        raise ArithmeticError('independent geometric positive mass failed')
    top_u = np.empty((3,n))
    top_u[:,mapping] = u.reshape(3,n)
    top_ku = streamed_action(topology,material,top_u.ravel(),guard)
    ku = top_ku.reshape(3,n)[:,mapping].ravel()
    scalar_mass = mass[mapping]
    vector_mass = np.tile(scalar_mass,3)
    physical = float((v@(vector_mass*v)+u@ku)/2)
    modified = float(physical-dt_s**2/8*(ku@(ku/vector_mass)))
    return dict(physical_energy_j=physical,modified_energy_j=modified,scalar_mass_kg=scalar_mass)


def verify(root,steps,output,guard):
    root,output = Path(root).resolve(),Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('independent window check needs a fresh output directory')
    output.mkdir(parents=True,exist_ok=True)
    report = dict(completed=False,passed=False,ready_for_training_data=False,training_updates=0,
        checker_sha256=sha256_file(Path(__file__)),cpu_runtime=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__))
    try:
        level = read_level(dict(directory=str(root),steps=steps),.0005,guard)
        plan,native = level['plan'],level['native']
        if plan['source_sha256']!=reference_source_hashes():
            raise ValueError('independent check source no longer matches the recorded formulation')
        context = capacity_context(Path(plan['capacity_directory']))
        _,_,_,material,settings = context
        mesh = load_mesh(Path(plan['capacity_directory']),context)
        with np.load(root/'native/mechanical-window.npz',allow_pickle=False) as arrays:
            u,v = arrays['final_displacement_m'],arrays['final_velocity_m_s']
            physical,modified = arrays['physical_energy_j'][-1],arrays['modified_energy_j'][-1]
        reference = cpu_state_reference(mesh,material,settings,level['coefficients']['native_to_topology'],u,v,native['dt_s'],guard)
        stored_mass = level['coefficients']['nodal_mass_kg']
        mass_error = float(np.linalg.norm(reference['scalar_mass_kg']-stored_mass)/np.linalg.norm(stored_mass))
        if min(physical,modified,reference['physical_energy_j'],reference['modified_energy_j'])<=0:
            raise ArithmeticError('final independent/reference energies must be positive')
        physical_error = abs(reference['physical_energy_j']/physical-1)
        modified_error = abs(reference['modified_energy_j']/modified-1)
        final_probe = level['coefficients']['probes']@u
        probe_norm = np.linalg.norm(final_probe)
        if probe_norm==0:
            raise ArithmeticError('independent final receiver vector is zero')
        probe_error = float(np.linalg.norm(final_probe-level['trace'][-1])/probe_norm)
        if plan['source_sha256']!=reference_source_hashes() or any(sha256_file(root/name)!=digest for name,digest in level['identities'].items()):
            raise ValueError('independent window source/inputs changed during assessment')
        guard.check(force=True)
        report.update(completed=True,passed=mass_error<1e-12 and max(physical_error,modified_error)<1e-9 and probe_error<1e-10,
            plan_fingerprint=plan['fingerprint'],forward_result_fingerprint=level['parent']['fingerprint'],
            duration_s=.0005,steps=steps,vector_dofs=len(u),input_identities=level['identities'],
            mass_relative_l2=mass_error,physical_energy_j=reference['physical_energy_j'],modified_energy_j=reference['modified_energy_j'],
            final_physical_energy_relative_error=physical_error,final_modified_energy_relative_error=modified_error,
            final_probe_relative_l2=probe_error,scope='Saved-state CPU mass/stiffness/energy/probe checks, not a full CPU replay or spatial/PZT acceptance.')
    except Exception as error:
        report['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        report['resources'] = guard.report()
        report['fingerprint'] = canonical_hash(report)
        write_json(output/'cpu-window-check.json',report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--steps',type=int,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    guard = ResourceGuard(ResourceLimits(max_seconds=90,max_rss_mib=1024,max_system_memory_percent=85,min_available_mib=2048))
    result = verify(args.run,args.steps,args.output,guard)
    print(json.dumps({k:v for k,v in result.items() if k not in ('resources','input_identities')}))
    return 0 if result['passed'] else 2


if __name__=='__main__':
    raise SystemExit(main())
