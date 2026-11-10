"""Guarded native CUDA tensor checks and actual full-plate operator cost."""
import argparse
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import sys
import time
import zipfile
import numpy as np
from scipy.linalg import eigh
from examples.tensor_reference import native_operators
from traxis.config import Material,Plate
from traxis.tensor_hex import make_grid
from traxis.gpu_tensor import make_layout,build_gpu_tensor,streamed_action
from traxis.gpu_enriched_verlet import gpu_explicit_simulate
from traxis.elmer_plate_refinement import reference_source_hashes
from traxis.elmer_reference import run_monitored
from traxis.gpu_verification import gpu_package_identity
from traxis.highorder_verification import distribution_identity
from traxis.resources import ResourceGuard,ResourceLimits
from traxis.storage import canonical_hash,sha256_file,write_json


def identities():
    return dict(source_sha256=reference_source_hashes(),driver_sha256={name:sha256_file(Path(__file__).parent/name)
        for name in ('gpu_tensor_reference.py','tensor_reference.py')})


def small_reference(output,guard):
    material=Material();cases=[];arrays={};mode_result=None
    random=np.random.default_rng(293)
    for name,transform in (('rectangular',np.eye(3)),('affine-skewed',np.array([[1,.13,.07],[.04,1,.11],[.02,.03,1.]]))):
        for order in range(1,5):
            guard.check(force=True)
            plate=Plate(width_m=.04,height_m=.03,thickness_m=.003)
            grid=make_grid(plate,(2,1,1),order,transform);layout=make_layout(plate,(2,1,1),order,transform)
            system=build_gpu_tensor(layout,material,guard=guard)
            native,_,overlap=native_operators(grid,material,guard)
            k,m=native[2],native[3];n=len(k)
            basis=np.eye(n);gpu=np.column_stack([system.stiffness@basis[:,i] for i in range(n)])
            matrix_error=float(np.linalg.norm(gpu-k)/np.linalg.norm(k))
            mass_error=float(np.linalg.norm(system.mass_diagonal-np.diag(m))/np.linalg.norm(np.diag(m)))
            state=random.normal(size=n);reference=streamed_action(layout,material,state,guard)
            field_error=float(np.linalg.norm(system.stiffness@state-reference)/np.linalg.norm(reference))
            scaled=(-.5*system.stiffness)@state
            scale_error=float(np.linalg.norm(scaled+.5*(k@state))/np.linalg.norm(.5*(k@state)))
            cases.append(dict(geometry=name,order=order,vector_dofs=n,all_column_native_relative_l2=matrix_error,
                native_mass_relative_l2=mass_error,streamed_cpu_field_relative_l2=field_error,negative_scale_relative_l2=scale_error,
                shared_transform_error=overlap,stable_dt_s=system.stable_dt_s,
                passed=bool(max(matrix_error,mass_error,field_error,scale_error)<1e-10)))
            key=f'{name}_p{order}_';arrays[key+'native_stiffness']=k;arrays[key+'gpu_stiffness']=gpu
            arrays[key+'mass_diagonal']=system.mass_diagonal
            if name=='rectangular' and order==3:
                eigenvalues,eigenvectors=eigh(k,m);omega=math.sqrt(eigenvalues[-1])
                initial=eigenvectors[:,-1];initial=initial*(1e-9/np.max(abs(initial)))
                duration=2/omega;probes=np.zeros((1,n));probes[0,np.argmax(abs(initial))]=1.
                levels=[]
                for steps in (32,64,128):
                    result=gpu_explicit_simulate(system,steps,duration/steps,probes,np.zeros(n),
                        initial_displacement=initial,guard=guard)
                    exact_u=initial*np.cos(omega*duration);exact_v=-omega*initial*np.sin(omega*duration)
                    error=float(np.sqrt(np.sum(system.mass_diagonal*(result.final_displacement_m-exact_u)**2))/
                        np.sqrt(np.sum(system.mass_diagonal*initial**2)))
                    velocity_error=float(np.sqrt(np.sum(system.mass_diagonal*(result.final_velocity_m_s-exact_v)**2))/
                        (omega*np.sqrt(np.sum(system.mass_diagonal*initial**2))))
                    energy_range=float(np.ptp(result.modified_energy_j)/abs(result.modified_energy_j.mean()))
                    levels.append(dict(steps=steps,dt_s=duration/steps,displacement_error=error,velocity_error=velocity_error,
                        modified_energy_relative_range=energy_range,statistics=result.statistics))
                    for label,value in dict(time_s=result.time_s,trace=result.displacement_m,final_u=result.final_displacement_m,
                        final_v=result.final_velocity_m_s,exact_u=exact_u,exact_v=exact_v,modified_energy=result.modified_energy_j).items():
                        arrays[f'mode_{steps}_{label}']=value
                ratios=[levels[i]['displacement_error']/levels[i+1]['displacement_error'] for i in (0,1)]
                mode_result=dict(duration_s=duration,omega_rad_s=omega,levels=levels,displacement_error_ratios=ratios,
                    passed=bool(all(3.8<r<4.2 for r in ratios) and max(r['modified_energy_relative_range'] for r in levels)<1e-8),
                    scope='Small native eigenmode verification, not a full-plate waveform or physical source/probe.')
    np.savez_compressed(output/'native-gpu-comparisons.npz',**arrays)
    return dict(cases=cases,mode_check=mode_result,archive='native-gpu-comparisons.npz',
        passed=bool(all(c['passed'] for c in cases) and mode_result['passed']))


def full_reference(output,guard):
    plate,material=Plate(),Material()
    layout=make_layout(plate,(150,150,1),4,max_device_mib=1024,max_host_mib=1024)
    system=build_gpu_tensor(layout,material,guard=guard)
    points=layout.points();x,y,z=(points[:,i]/length for i,length in enumerate((.5,.5,.003)))
    state=np.array([np.sin(2*np.pi*x)+.17*np.cos(3*np.pi*z),.5*np.cos(np.pi*y)*np.sin(np.pi*z),
                    .25*np.sin(np.pi*(x+y))*np.cos(np.pi*z)]).ravel()
    started=time.perf_counter();gpu=system.stiffness@state;host_seconds=time.perf_counter()-started
    started=time.perf_counter();reference=streamed_action(layout,material,state,guard,chunk_elements=64)
    cpu_seconds=time.perf_counter()-started
    error=float(np.linalg.norm(gpu-reference)/np.linalg.norm(reference))
    storage=system.stiffness.storage;cp=storage.cp;event_seconds=[];wall_seconds=[]
    with storage.lock,cp.cuda.Device(storage.device_id):
        for _ in range(5):
            guard.check()
            start,stop=cp.cuda.Event(),cp.cuda.Event();wall=time.perf_counter()
            start.record();storage._launch(storage.input,storage.output,1.,0.);stop.record();stop.synchronize()
            wall_seconds.append(time.perf_counter()-wall);event_seconds.append(cp.cuda.get_elapsed_time(start,stop)/1000)
    ns=layout.screen['scalar_dofs'];mass=system.mass_diagonal[:ns]
    expected=2700*.5*.5*.003
    mass_error=abs(mass.sum()/expected-1)
    moments=[float(mass@(points[:,i]/length)/mass.sum()) for i,length in enumerate((.5,.5,.003))]
    second=[float(mass@(points[:,i]/length)**2/mass.sum()) for i,length in enumerate((.5,.5,.003))]
    moment_error=max(max(abs(v-.5) for v in moments),max(abs(v-1/3) for v in second))
    np.savez_compressed(output/'full-operator-comparison.npz',input=state,gpu_action=gpu,cpu_action=reference,
        points_m=points,nodal_mass_kg=mass)
    return dict(passed=bool(error<1e-10 and mass_error<1e-12 and moment_error<1e-12),
        archive='full-operator-comparison.npz',screen=system.screen,device=system.device,vector_dofs=len(state),
        positive_mass=bool(np.all(mass>0)),total_mass_kg=float(mass.sum()),mass_relative_error=mass_error,
        maximum_box_moment_error=moment_error,operator_relative_l2=error,host_action_seconds=host_seconds,
        streamed_cpu_reference_seconds=cpu_seconds,device_action_event_seconds=event_seconds,
        device_action_wall_seconds=wall_seconds,stable_dt_s=system.stable_dt_s,
        time_steps_completed=0,full_window_completed=False,
        scope='Actual 500 mm intact Q4 operator/capacity measurement; no time propagation, convergence, recess or PZT acceptance.')


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--mode',choices=('small','full'),required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--worker',action='store_true');parser.add_argument('--plan',type=Path)
    args=parser.parse_args();output=args.output.resolve()
    guard=ResourceGuard(ResourceLimits(max_seconds=240,max_rss_mib=1536,max_system_memory_percent=85,min_available_mib=2048))
    if args.worker:
        plan=json.loads(args.plan.read_text())
        if (canonical_hash({k:v for k,v in plan.items() if k!='fingerprint'})!=plan['fingerprint']
                or any(plan[k]!=v for k,v in identities().items()) or plan['mode']!=args.mode
                or plan['runtime']!=distribution_identity(guard) or plan['gpu_packages']!=gpu_package_identity(guard)):
            raise ValueError('tensor GPU source/runtime/plan changed')
        output.mkdir();report=dict(completed=False,passed=False,ready_for_training_data=False,training_updates=0,
            mode=args.mode,plan_fingerprint=plan['fingerprint'])
        try:
            report.update(small_reference(output,guard) if args.mode=='small' else full_reference(output,guard))
            if (any(plan[k]!=v for k,v in identities().items()) or plan['runtime']!=distribution_identity(guard)
                    or plan['gpu_packages']!=gpu_package_identity(guard)):
                raise ValueError('tensor GPU source/runtime changed during measurement')
            report.update(completed=True,archive_sha256=sha256_file(output/report['archive']))
        except Exception as error:
            report['error']=f'{type(error).__name__}: {error}';raise
        finally:
            report['resources']=guard.report();report['fingerprint']=canonical_hash(report)
            write_json(output/'gpu-tensor-report.json',report)
        return 0 if report['passed'] else 2
    if output.exists() and any(output.iterdir()):raise FileExistsError('tensor GPU verification needs a fresh output directory')
    output.mkdir(parents=True,exist_ok=True)
    plan={**identities(),'mode':args.mode,'material':asdict(Material()),'runtime':distribution_identity(guard),
        'gpu_packages':gpu_package_identity(guard),'limits':guard.report()['limits']}
    with zipfile.ZipFile(output/'source-snapshot.zip','w',compression=zipfile.ZIP_DEFLATED) as archive:
        for name in plan['source_sha256']:archive.write(Path(__file__).resolve().parents[1]/'traxis'/name,'traxis/'+name)
        for name in plan['driver_sha256']:archive.write(Path(__file__).parent/name,'examples/'+name)
    plan['source_snapshot_sha256']=sha256_file(output/'source-snapshot.zip');plan['fingerprint']=canonical_hash(plan)
    path=output/'gpu-tensor-plan.json';write_json(path,plan)
    env=os.environ.copy();env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',CUPY_CACHE_DIR=str(output/'kernel-cache'),CUPY_CACHE_SAVE_CUDA_SOURCE='1')
    temporary=output/'temporary-files';temporary.mkdir();env.update({name:str(temporary) for name in ('TEMP','TMP','TMPDIR')})
    command=[sys.executable,'-m','examples.gpu_tensor_reference','--worker','--plan',str(path),'--mode',args.mode,'--output',str(output/'native')]
    report=dict(completed=False,passed=False,ready_for_training_data=False,training_updates=0,plan_fingerprint=plan['fingerprint'])
    try:
        worker=run_monitored(command,Path(__file__).resolve().parents[1],env,guard,output/'worker.log');report['worker_run']=worker
        if worker['exit_code'] not in (0,2) or not (output/'native/gpu-tensor-report.json').is_file():
            raise RuntimeError(f"tensor GPU worker exited {worker['exit_code']}; see worker.log")
        native=json.loads((output/'native/gpu-tensor-report.json').read_text())
        if (native['plan_fingerprint']!=plan['fingerprint'] or canonical_hash({k:v for k,v in native.items() if k!='fingerprint'})!=native['fingerprint']
                or native['archive_sha256']!=sha256_file(output/'native'/native['archive'])
                or any(plan[k]!=v for k,v in identities().items())):
            raise ValueError('tensor GPU receipt or source changed')
        report.update(completed=True,passed=worker['exit_code']==0 and native['passed'],native_report_sha256=sha256_file(output/'native/gpu-tensor-report.json'))
    except Exception as error:report['error']=f'{type(error).__name__}: {error}'
    finally:
        report['resources']=guard.report();report['fingerprint']=canonical_hash(report);write_json(output/'monitor-result.json',report)
    print(json.dumps({k:v for k,v in report.items() if k!='resources'}))
    return 0 if report['passed'] else 2


if __name__=='__main__':raise SystemExit(main())
