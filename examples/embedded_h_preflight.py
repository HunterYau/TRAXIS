"""Build an actual finer-grid CPU operator and screen GPU storage without CUDA allocation."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time
import zipfile

import numpy as np

from examples.embedded_wave import case_config,gpu_controls,mesh_controls,screen
from examples.tensor_wave import PROJECT,identities,sealed,snapshot_sources
from traxis.config import CircularRecess,Material,Plate
from traxis.embedded_recess import make_embedded_recess_mesh
from traxis.gpu_mapped_tensor import storage_screen as gpu_screen
from traxis.highorder_verification import distribution_identity
from traxis.mapped_tensor import build_system
from traxis.resources import ResourceGuard,ResourceLimits
from traxis.storage import sha256_file,write_json


def preflight(level,defective,output,guard):
    output=Path(output).resolve()
    if not output.is_relative_to(PROJECT/'outputs'):raise ValueError('h preflight output must remain in project outputs')
    case=case_config('h_refinement',h_level=level,defective=defective)
    output.mkdir(parents=True,exist_ok=False)
    analysis={name:sha256_file(PROJECT/name) for name in ('examples/embedded_h_preflight.py','examples/embedded_wave.py')}
    plan=dict(kind='embedded_h_cpu_capacity_preflight',case=case,mesh_screen=screen(case),
        **identities(),analysis_sources=analysis,runtime=distribution_identity(guard),limits=asdict(guard.limits))
    snapshot_sources(output/'source-snapshot.zip',plan)
    with zipfile.ZipFile(output/'source-snapshot.zip','a',compression=zipfile.ZIP_DEFLATED) as z:
        for name in analysis:z.writestr(name,(PROJECT/name).read_bytes())
    plan['source_snapshot_sha256']=sha256_file(output/'source-snapshot.zip')
    write_json(output/'preflight-plan.json',sealed(plan))
    report=dict(completed=False,passed=False,ready_for_training_data=False,training_updates=0,
        plan_fingerprint=plan['fingerprint'],gpu_operator_allocated=False,wave_steps=0,
        scope='Actual CPU geometry/operator and analytic GPU storage/stability screens only; no GPU action or wave accuracy acceptance.')
    started=time.perf_counter()
    try:
        mesh=make_embedded_recess_mesh(Plate(**case['plate']),CircularRecess(**case['recess']),case['order'],
            guard=guard,**mesh_controls(case))
        print(json.dumps(dict(phase='mesh_built',vector_dofs=3*len(mesh.points_m),elapsed_seconds=time.perf_counter()-started)),flush=True)
        cpu=build_system(mesh,Material(**case['material']),guard,max_bytes=case['max_operator_bytes'],max_batch_vectors=1)
        allocation=gpu_screen(cpu,**gpu_controls(case));dt=case['duration_s']/case['steps']
        if dt>.8*cpu.stable_dt_s:raise ValueError('declared h time step exceeds the unchanged stability margin')
        arrays={'points_m':mesh.points_m,'element_nodes':mesh.element_nodes,'mass_kg':cpu.mass_kg,
            'inverse_jacobians':cpu.inverse_jacobians,'volume_weights_m3':cpu.volume_weights_m3}
        hashes={}
        for name,array in arrays.items():
            guard.check(force=True)
            if not array.flags.c_contiguous:raise ValueError('preflight array layout changed')
            hashes[name]=hashlib.sha256(memoryview(array).cast('B')).hexdigest()
        if (any(plan[k]!=v for k,v in identities().items()) or plan['runtime']!=distribution_identity(guard)
                or any(sha256_file(PROJECT/name)!=h for name,h in analysis.items())
                or sha256_file(output/'source-snapshot.zip')!=plan['source_snapshot_sha256']):
            raise ValueError('preflight source or runtime changed during assembly')
        volume=float(cpu.volume_weights_m3.sum())
        analytic=case['plate']['width_m']*case['plate']['height_m']*case['plate']['thickness_m']
        if defective:analytic-=np.pi*case['recess']['radius_m']**2*case['recess']['depth_m']
        report.update(completed=True,passed=True,vector_dofs=3*len(mesh.points_m),cpu_operator=cpu.screen,
            gpu_allocation_forecast=allocation,dt_s=dt,stable_dt_s=cpu.stable_dt_s,
            stability_margin_used=dt/cpu.stable_dt_s,total_quadrature_volume_m3=volume,
            analytic_volume_m3=analytic,volume_relative_difference=abs(volume/analytic-1),
            shared_coordinate_error_m=mesh.shared_coordinate_error_m,array_sha256=hashes)
    except Exception as error:
        report['error']=f'{type(error).__name__}: {error}';raise
    finally:
        report['wall_seconds']=time.perf_counter()-started;report['resources']=guard.report()
        write_json(output/'preflight-report.json',sealed(report))
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--h-level',type=int,required=True)
    parser.add_argument('--recess',action='store_true');parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();guard=ResourceGuard(ResourceLimits(max_seconds=900,max_rss_mib=4096,
        max_system_memory_percent=89.9,min_available_mib=768))
    report=preflight(args.h_level,args.recess,args.output,guard)
    print(json.dumps({k:report[k] for k in ('completed','passed','vector_dofs','dt_s','stable_dt_s','fingerprint')}))
    return 0 if report['passed'] else 2


if __name__=='__main__':raise SystemExit(main())
