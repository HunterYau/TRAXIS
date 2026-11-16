"""Check an actual coupled in-plane/thickness proposal before any wave launch."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time
import zipfile

import numpy as np

from examples.tensor_wave import PROJECT, identities, read_json, sealed, snapshot_sources
from traxis.config import CircularRecess, Material, Plate
from traxis.embedded_recess import make_embedded_recess_mesh, storage_screen
from traxis.gpu_mapped_tensor import storage_screen as gpu_screen
from traxis.highorder_verification import distribution_identity
from traxis.mapped_tensor import build_system
from traxis.resources import ResourceGuard, ResourceLimits
from traxis.storage import canonical_hash, sha256_file, write_json


PROPOSAL_SHA256 = '61f8245813b26481ce40042f595ef6d6806486094fd6134f8922e5e91946b5c1'
PROPOSAL_FINGERPRINT = 'fb7fb254f7cb0f365dc1fa1eaedb71f00ea62ea8534eed029dba59096c2e2161'
INTEGRATOR_SHA256 = '72cc166792438c1cdee7cafcf42df2256d470d9667e035fa04c7df93b99550f5'


def load_proposal():
    path = PROJECT/'configs/embedded-xyz-proposal.json'
    proposal = read_json(path)
    if sha256_file(path)!=PROPOSAL_SHA256 or canonical_hash(proposal)!=PROPOSAL_FINGERPRINT:
        raise ValueError('coupled XYZ preflight needs the unchanged proposal')
    return proposal


def mesh_controls(proposal, level, defective):
    if type(level) is not int or not 0<=level<3 or type(defective) is not bool:
        raise ValueError('XYZ proposal needs an explicit level 0/1/2 and material state')
    if canonical_hash(proposal)!=PROPOSAL_FINGERPRINT:
        raise ValueError('XYZ proposal mesh settings changed')
    return dict(order=proposal['order'], geometry_order=proposal['geometry_order'],
        **proposal['levels'][level], defective=defective,
        max_vector_dofs=proposal['max_vector_dofs'], max_bytes=proposal['max_mesh_bytes'])


def preflight(level, defective, output, guard):
    output = Path(output).resolve()
    if not output.is_relative_to(PROJECT/'outputs'):
        raise ValueError('XYZ preflight output must remain in project outputs')
    proposal = load_proposal(); controls = mesh_controls(proposal,level,defective)
    plate,recess = Plate(**proposal['plate']),CircularRecess(**proposal['recess'])
    mesh_screen = storage_screen(plate,recess,**controls)
    integrator = PROJECT/'traxis/gpu_enriched_verlet.py'
    if sha256_file(integrator)!=INTEGRATOR_SHA256:
        raise ValueError('recheck explicit API limits against the changed integrator source')
    # These exact limits were inspected in the bound integrator. This is an
    # interface screen, not a CUDA launch or proof of free device memory.
    samples = proposal['steps']+1
    energies = (proposal['steps']+255)//256+1
    sample_bytes = samples*3*8+energies*24
    vector_sample_mib = (16*mesh_screen['vector_dofs']*8+sample_bytes)/1024**2
    api_supported = bool(proposal['steps']<=500000 and sample_bytes<=64*1024**2
        and vector_sample_mib<=proposal['max_workspace_mib']<=2048)
    output.mkdir(parents=True,exist_ok=False)
    analysis = {name:sha256_file(PROJECT/name) for name in
        ('examples/embedded_xyz_preflight.py','configs/embedded-xyz-proposal.json')}
    plan = dict(kind='embedded_xyz_cpu_capacity_preflight', proposal=proposal,
        proposal_fingerprint=PROPOSAL_FINGERPRINT, level=level, defective=defective, mesh_screen=mesh_screen,
        **identities(), analysis_sources=analysis, runtime=distribution_identity(guard), limits=asdict(guard.limits))
    snapshot_sources(output/'source-snapshot.zip',plan)
    with zipfile.ZipFile(output/'source-snapshot.zip','a',compression=zipfile.ZIP_DEFLATED) as archive:
        for name in analysis:
            archive.writestr(name,(PROJECT/name).read_bytes())
    plan['source_snapshot_sha256']=sha256_file(output/'source-snapshot.zip')
    write_json(output/'preflight-plan.json',sealed(plan))
    report = dict(kind='embedded_xyz_cpu_capacity_preflight', completed=False, passed=False,
        ready_for_training_data=False,training_updates=0,plan_fingerprint=plan['fingerprint'],
        gpu_operator_allocated=False,wave_steps=0,wave_launch_ready=False,
        integrator_api_supported=api_supported,vector_sample_workspace_lower_mib=vector_sample_mib,
        inspected_integrator_workspace_ceiling_mib=2048,proposed_workspace_ceiling_mib=proposal['max_workspace_mib'],
        launch_blocker='The proposed 3072 MiB workspace ceiling requires an explicit API extension and archived/current CUDA equivalence checks before any wave run.',
        scope='Actual CPU mesh/operator, conservative stability and GPU allocation forecasts for coupled XYZ refinement; no GPU wave or convergence acceptance.')
    started = time.perf_counter()
    try:
        mesh = make_embedded_recess_mesh(plate,recess,guard=guard,**controls)
        print(json.dumps(dict(phase='mesh_built',vector_dofs=3*len(mesh.points_m),
            elapsed_seconds=time.perf_counter()-started)),flush=True)
        cpu = build_system(mesh,Material(),guard,max_bytes=proposal['max_operator_bytes'],max_batch_vectors=1)
        allocation = gpu_screen(cpu,max_device_mib=proposal['max_device_mib'],
            max_host_mib=proposal['max_host_mib'],max_vector_dofs=proposal['max_vector_dofs'])
        dt = proposal['duration_s']/proposal['steps']
        if dt>proposal['max_dt_fraction']*cpu.stable_dt_s:
            raise ValueError('proposed time step exceeds the unchanged stability margin')
        arrays = {name:getattr(mesh,name) for name in ('points_m','element_nodes')}
        arrays.update({name:getattr(cpu,name) for name in ('mass_kg','inverse_jacobians','volume_weights_m3')})
        hashes = {}
        for name,array in arrays.items():
            guard.check(force=True)
            if not array.flags.c_contiguous:
                raise ValueError('XYZ preflight array layout changed')
            hashes[name]=hashlib.sha256(memoryview(array).cast('B')).hexdigest()
        if (identities()!= {k:plan[k] for k in ('source_sha256','driver_sha256')}
                or distribution_identity(guard)!=plan['runtime'] or load_proposal()!=proposal
                or any(sha256_file(PROJECT/name)!=digest for name,digest in analysis.items())
                or sha256_file(output/'source-snapshot.zip')!=plan['source_snapshot_sha256']):
            raise ValueError('XYZ preflight source, proposal or runtime changed during assembly')
        volume = float(cpu.volume_weights_m3.sum())
        analytic = plate.width_m*plate.height_m*plate.thickness_m
        if defective:
            analytic -= np.pi*recess.radius_m**2*recess.depth_m
        report.update(completed=True,passed=True,vector_dofs=3*len(mesh.points_m),cpu_operator=cpu.screen,
            gpu_allocation_forecast=allocation,dt_s=dt,stable_dt_s=cpu.stable_dt_s,
            stability_margin_used=dt/cpu.stable_dt_s,volume_relative_difference=abs(volume/analytic-1),
            shared_coordinate_error_m=mesh.shared_coordinate_error_m,array_sha256=hashes)
    except Exception as error:
        report['error']=f'{type(error).__name__}: {error}'
        raise
    finally:
        report['wall_seconds']=time.perf_counter()-started;report['resources']=guard.report()
        write_json(output/'preflight-report.json',sealed(report))
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--level',type=int,required=True)
    parser.add_argument('--recess',action='store_true')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    guard=ResourceGuard(ResourceLimits(max_seconds=1200,max_rss_mib=4096,
        max_system_memory_percent=89.9,min_available_mib=768))
    report=preflight(args.level,args.recess,args.output,guard)
    print(json.dumps({k:report[k] for k in ('completed','passed','vector_dofs',
        'dt_s','stable_dt_s','integrator_api_supported','wave_launch_ready','fingerprint')}))
    return 0 if report['passed'] else 2


if __name__=='__main__':
    raise SystemExit(main())
