"""Check actual coupled-layer Q4 GPU actions and planned launch interfaces."""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import time
import zipfile

import numpy as np

from examples.embedded_wave import case_config,gpu_controls
from examples.embedded_wave_evidence import check_snapshot
from examples.tensor_wave import PROJECT,identities,sealed,snapshot_sources
from traxis.config import CircularRecess,Material,Plate
from traxis.embedded_recess import make_embedded_recess_mesh
from traxis.fem import tone_burst
from traxis.gpu_enriched_verlet import gpu_explicit_simulate,tensor_patch_probe_set
from traxis.gpu_mapped_tensor import build_gpu_mapped
from traxis.gpu_verification import gpu_package_identity
from traxis.highorder_verification import distribution_identity
from traxis.mapped_tensor import action,build_system
from traxis.resources import ResourceGuard,ResourceLimits
from traxis.storage import sha256_file,write_json


class FirstBatch(RuntimeError):
    pass


def verify(output,guard):
    output=Path(output).resolve()
    if not output.is_relative_to(PROJECT/'outputs'):
        raise ValueError('XYZ GPU debug output must remain in project outputs')
    output.mkdir(parents=True,exist_ok=False)
    temporary=output/'temporary';temporary.mkdir()
    os.environ.update(CUPY_CACHE_DIR=str(output/'kernel-cache'),CUPY_CACHE_SAVE_CUDA_SOURCE='1',
        **{key:str(temporary) for key in ('TMP','TEMP','TMPDIR')})
    base=dict(plate=asdict(Plate(.08,.06,.003175)),recess=asdict(CircularRecess(.041,.029,.008,.0015)),
        angular_cells=2,radial_cells=[1,1],outer_cell_size_m=.04,order=4,geometry_order=3)
    cases=[case_config('xyz_refinement',xyz_level=i,defective=defective)
        for defective in (False,True) for i in range(3)]
    names=('examples/verify_xyz_gpu_interfaces.py','examples/embedded_wave.py','examples/embedded_wave_budget.py',
        'examples/embedded_wave_evidence.py',
        'examples/embedded_xyz_preflight.py','configs/embedded-xyz-proposal.json')
    plan=dict(kind='embedded_xyz_small_gpu_interface_debug',case=dict(fixture=base,planned_controls=cases),
        **identities(),analysis_sources={name:sha256_file(PROJECT/name) for name in names},
        runtime=distribution_identity(guard),gpu_packages=gpu_package_identity(guard),limits=asdict(guard.limits))
    snapshot_sources(output/'source-snapshot.zip',plan)
    with zipfile.ZipFile(output/'source-snapshot.zip','a',compression=zipfile.ZIP_DEFLATED) as archive:
        for name in names:
            archive.writestr(name,(PROJECT/name).read_bytes())
    plan['source_snapshot_sha256']=sha256_file(output/'source-snapshot.zip')
    write_json(output/'interface-plan.json',sealed(plan))
    report=dict(kind='embedded_xyz_small_gpu_interface_debug',completed=False,passed=False,
        ready_for_training_data=False,training_updates=0,plan_fingerprint=plan['fingerprint'],
        full_size_gpu_operator_allocated=False,complete_wave_window=False,
        scope='Actual small coupled-layer Q4 GPU/CPU force parity and planned API settings, deliberately stopped at batch 16. No full-size GPU allocation, waveform convergence or physical-data acceptance.')
    started=time.perf_counter()
    try:
        rng=np.random.default_rng(20261116);rows=[]
        for case in cases:
            guard.check(force=True)
            mesh=make_embedded_recess_mesh(Plate(**base['plate']),CircularRecess(**base['recess']),base['order'],
                geometry_order=base['geometry_order'],angular_cells=base['angular_cells'],
                radial_cells=tuple(base['radial_cells']),outer_cell_size_m=base['outer_cell_size_m'],
                layers=tuple(case['layers']),defective=case['defective'],guard=guard)
            cpu=build_system(mesh,Material(),guard,max_bytes=512*1024**2,max_batch_vectors=1)
            gpu=build_gpu_mapped(cpu,guard=guard,**gpu_controls(case))
            state=rng.normal(size=len(cpu.mass_kg));expected=action(cpu,state,guard);actual=gpu.stiffness@state
            error=float(np.linalg.norm(actual-expected)/np.linalg.norm(expected))
            if not error<1e-11 or not np.array_equal(gpu.mass_diagonal,cpu.mass_kg):
                raise ArithmeticError('coupled-layer GPU force or mass differs from the independent CPU contraction')
            # A sparse covector samples all three component blocks. These are
            # algebraic probes/loads for this debug check, not mechanical PZTs.
            from types import SimpleNamespace
            patches=[SimpleNamespace(indices=np.array([0,len(mesh.points_m)//2,len(mesh.points_m)-1]),
                weights=np.array([.2,.3,.5]),direction=np.array(direction))
                for direction in ([1.,0.,0.],[0.,1.,0.],[.2,-.3,.9])]
            probes=tensor_patch_probe_set(patches,len(mesh.points_m));load=rng.normal(size=len(cpu.mass_kg))
            events=[]
            def stop(event):
                events.append(dict(event))
                gpu.stiffness.storage.cp.get_default_pinned_memory_pool().free_all_blocks()
                raise FirstBatch()
            try:
                gpu_explicit_simulate(gpu,case['steps'],case['duration_s']/case['steps'],probes,load,
                    force=lambda t:case['force_scale_n']*tone_burst(t,case['frequency_hz'],case['cycles']),
                    output_stride=case['output_stride'],energy_stride=case['energy_stride'],guard=guard,
                    max_workspace_mib=case['max_workspace_mib'],progress=stop)
            except FirstBatch:
                if len(events)!=1 or events[0]['steps_completed']!=16:
                    raise ArithmeticError('XYZ interface did not reach the expected synchronized batch')
            else:
                raise ArithmeticError('XYZ interface debug unexpectedly completed the full requested window')
            rows.append(dict(level=case['xyz_level'],defective=case['defective'],layers=case['layers'],passed=True,
                actual_fixture_vector_dofs=len(cpu.mass_kg),actual_fixture_elements=len(mesh.element_nodes),
                force_relative_l2=error,mass_exact=True,planned_steps=case['steps'],steps_completed=16,
                actual_probe_count=3,
                actual_dt_s=case['duration_s']/case['steps'],actual_fixture_stable_dt_s=cpu.stable_dt_s,
                requested_workspace_mib=case['max_workspace_mib'],device=gpu.device))
            print(json.dumps({k:rows[-1][k] for k in ('level','defective','layers','actual_fixture_vector_dofs','force_relative_l2','steps_completed')}),flush=True)
        if (any(plan[k]!=v for k,v in identities().items()) or plan['runtime']!=distribution_identity(guard)
                or plan['gpu_packages']!=gpu_package_identity(guard)
                or any(sha256_file(PROJECT/name)!=digest for name,digest in plan['analysis_sources'].items())):
            raise ValueError('XYZ GPU interface source or runtime changed during checking')
        check_snapshot(plan,output/'source-snapshot.zip');guard.check(force=True)
        report.update(completed=True,passed=True,cases=rows)
    except Exception as error:
        report['error']=f'{type(error).__name__}: {error}'
        raise
    finally:
        report['wall_seconds']=time.perf_counter()-started;report['resources']=guard.report()
        write_json(output/'interface-report.json',sealed(report))
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    guard=ResourceGuard(ResourceLimits(max_seconds=180,max_rss_mib=1536,
        max_system_memory_percent=89.9,min_available_mib=768))
    report=verify(args.output,guard)
    print(json.dumps({k:report[k] for k in ('completed','passed','fingerprint')}))
    return 0 if report['passed'] else 2


if __name__=='__main__':
    raise SystemExit(main())
