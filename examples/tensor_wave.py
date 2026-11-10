"""Guarded complete tensor wave windows with physical circular mechanical ports."""
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

from traxis.config import Material,Plate
from traxis.elmer_plate_refinement import reference_source_hashes
from traxis.elmer_reference import run_monitored
from traxis.fem import tone_burst
from traxis.gpu_enriched_verlet import gpu_explicit_simulate
from traxis.gpu_tensor import make_layout,build_gpu_tensor,streamed_action
from traxis.gpu_verification import gpu_package_identity
from traxis.highorder_verification import distribution_identity
from traxis.resources import ResourceGuard,ResourceLimits
from traxis.storage import atomic_write,canonical_hash,sha256_file,write_json
from traxis.tensor_patches import circular_patch


PROJECT=Path(__file__).resolve().parents[1]
DRIVERS=('tensor_wave.py','tensor_wave_study.py','verify_tensor_window.py','pace_gpu_worker.py')
PHYSICS=dict(duration_s=.0005,frequency_hz=100000.,cycles=2,force_scale_n=1.,radius_m=.003,
    max_device_mib=1536,max_host_mib=1024,operator_tolerance=1e-10,energy_tolerance=1e-8)


def read_json(path):
    path=Path(path)
    if path.stat().st_size>4*1024**2:raise MemoryError('tensor JSON input exceeds 4 MiB')
    return json.loads(path.read_text(encoding='utf-8'))


def sealed(value):
    value['fingerprint']=canonical_hash(value)
    return value


def checked_receipt(value):
    if value.get('fingerprint')!=canonical_hash({k:v for k,v in value.items() if k!='fingerprint'}):
        raise ValueError('tensor receipt fingerprint changed')
    return value


def identities():
    return dict(source_sha256=reference_source_hashes(),
        driver_sha256={name:sha256_file(PROJECT/'examples'/name) for name in DRIVERS})


def checked_case(cfg):
    expected=set(PHYSICS)|{'kind','scope','plate','material','cells','order','steps',
        'source_center_xy_m','receiver_centers_xy_m'}
    if set(cfg)!=expected or cfg['kind']!='tensor_mechanical_window_case':
        raise ValueError('tensor wave case has missing or unknown settings')
    if any(cfg[k]!=v for k,v in PHYSICS.items()) or cfg['material']!=asdict(Material()):
        raise ValueError('tensor case changed the declared material, pulse, window or limits')
    if type(cfg['order']) is not int or type(cfg['steps']) is not int:
        raise ValueError('tensor order and steps must be integers')
    if cfg['scope']=='research_intact':
        geometry=(asdict(Plate()),[150,150,1],[.02,.02],[[.25,.25],[.48,.48],[.35,.20]])
        valid=cfg['order'] in (3,4,5) and cfg['steps'] in (32768,65536,131072)
    elif cfg['scope']=='verification_smoke':
        geometry=(dict(width_m=.04,height_m=.03,thickness_m=.003),[2,2,1],[.008,.008],
            [[.02,.015],[.032,.022],[.028,.009]])
        valid=cfg['order']==4 and cfg['steps']==32768
    else:raise ValueError('unknown tensor wave scope')
    if not valid or tuple(cfg[k] for k in ('plate','cells','source_center_xy_m','receiver_centers_xy_m'))!=geometry:
        raise ValueError('tensor wave case changed the fixed geometry or refinement levels')
    make_layout(Plate(**cfg['plate']),cfg['cells'],cfg['order'],
        max_device_mib=cfg['max_device_mib'],max_host_mib=cfg['max_host_mib'])
    return cfg


def smoke_case():
    return checked_case(dict(kind='tensor_mechanical_window_case',scope='verification_smoke',
        plate=dict(width_m=.04,height_m=.03,thickness_m=.003),material=asdict(Material()),cells=[2,2,1],
        order=4,steps=32768,source_center_xy_m=[.008,.008],
        receiver_centers_xy_m=[[.02,.015],[.032,.022],[.028,.009]],**PHYSICS))


def layout_for(cfg):
    return make_layout(Plate(**cfg['plate']),cfg['cells'],cfg['order'],
        max_device_mib=cfg['max_device_mib'],max_host_mib=cfg['max_host_mib'])


def validate_plan(plan,guard=None):
    checked_receipt(plan);checked_case(plan['case'])
    if plan['kind']!='tensor_mechanical_window' or any(plan[k]!=v for k,v in identities().items()):
        raise ValueError('tensor wave implementation changed')
    if plan['runtime']!=distribution_identity(guard) or plan['gpu_packages']!=gpu_package_identity(guard):
        raise ValueError('tensor wave native/GPU runtime changed')
    if plan['study'] is not None:
        binding=plan['study'];path=Path(binding['path'])
        study=checked_receipt(read_json(path))
        if (sha256_file(path)!=binding['sha256'] or study['fingerprint']!=binding['fingerprint']
                or study['cases'][binding['case_name']]['case']!=plan['case']
                or any(study[k]!=plan[k] for k in ('source_sha256','driver_sha256','runtime','gpu_packages'))):
            raise ValueError('tensor wave frozen study changed')
    elif plan['case']['scope']!='verification_smoke':
        raise ValueError('research windows require a frozen study')
    return plan


def snapshot_sources(path,plan):
    def save(stream):
        with zipfile.ZipFile(stream,'w',compression=zipfile.ZIP_DEFLATED) as archive:
            for key,folder in (('source_sha256','traxis'),('driver_sha256','examples')):
                for name in plan[key]:archive.writestr(folder+'/'+name,(PROJECT/folder/name).read_bytes())
            archive.writestr('case.json',json.dumps(plan.get('case',plan),indent=2))
    atomic_write(path,save)


def physical_field(layout):
    x,y,z=(axis/length for axis,length in zip(layout.axes_m,
        (layout.plate.width_m,layout.plate.height_m,layout.plate.thickness_m)))
    x,y,z=x[:,None,None],y[None,:,None],z[None,None,:]
    values=np.empty((3,len(layout.axes_m[0]),len(layout.axes_m[1]),len(layout.axes_m[2])))
    values[0]=np.sin(2*np.pi*x)+.17*np.cos(3*np.pi*z)
    values[1]=.5*np.cos(np.pi*y)*np.sin(np.pi*z)
    values[2]=.25*np.sin(np.pi*(x+y))*np.cos(np.pi*z)
    return values.ravel()


def box_moments(layout,mass,material):
    n=layout.screen['scalar_dofs'];m=mass[:n].reshape(tuple(len(a) for a in layout.axes_m))
    axes=[a/length for a,length in zip(layout.axes_m,
        (layout.plate.width_m,layout.plate.height_m,layout.plate.thickness_m))]
    total=material.density_kg_m3*layout.plate.width_m*layout.plate.height_m*layout.plate.thickness_m
    rows=[]
    for degree in range(3):
        for a in range(degree+1):
            for b in range(degree-a+1):
                c=degree-a-b;expected=total/((a+1)*(b+1)*(c+1))
                measured=float(np.sum(m*axes[0][:,None,None]**a*axes[1][None,:,None]**b*axes[2][None,None,:]**c))
                rows.append(dict(powers=[a,b,c],measured=measured,expected=expected,relative_error=abs(measured/expected-1)))
    if max(r['relative_error'] for r in rows)>1e-10:raise ArithmeticError('tensor box mass moments failed')
    return rows


def operator_preflight(system,material,output,guard):
    field=physical_field(system.layout);started=time.perf_counter()
    action=system.stiffness@field;gpu_seconds=time.perf_counter()-started
    started=time.perf_counter();reference=streamed_action(system.layout,material,field,guard)
    error=float(np.linalg.norm(action-reference)/np.linalg.norm(reference));cpu_seconds=time.perf_counter()-started
    atomic_write(output/'operator-comparison.npz',lambda s:np.savez_compressed(s,
        input_field=field,gpu_action=action,reference_action=reference))
    if not math.isfinite(error) or error>1e-10:raise ArithmeticError('full tensor GPU/CPU action disagreement')
    return dict(relative_l2=error,host_gpu_seconds=gpu_seconds,cpu_seconds=cpu_seconds)


def worker(plan_path,output,guard):
    plan_path,output=Path(plan_path).resolve(),Path(output).resolve()
    plan=validate_plan(read_json(plan_path),guard);cfg=plan['case']
    if output!=plan_path.parent/'native' or (output.exists() and any(output.iterdir())):
        raise ValueError('tensor worker needs its new declared native output directory')
    output.mkdir(parents=True,exist_ok=True);started=time.perf_counter()
    report=dict(completed=False,passed=False,ready_for_training_data=False,training_updates=0,
        plan_fingerprint=plan['fingerprint'],steps_completed=0,full_window_completed=False,stages=[])
    def progress(stage,**fields):
        report['stages'].append(dict(stage=stage,elapsed_seconds=time.perf_counter()-started,**fields))
        write_json(output/'progress.json',report,overwrite=True)
    try:
        layout=layout_for(cfg);material=Material(**cfg['material'])
        system=build_gpu_tensor(layout,material,guard=guard)
        steps=cfg['steps'];dt=cfg['duration_s']/steps
        if dt>.95*system.stable_dt_s:raise ValueError('declared tensor time step exceeds its explicit bound')
        report.update(screen=system.screen,device=system.device,vector_dofs=layout.screen['vector_dofs'],
            stable_dt_s=system.stable_dt_s,planned_steps=steps,dt_s=dt,
            mass_moments=box_moments(layout,system.mass_diagonal,material))
        progress('operator_ready')
        report['operator']=operator_preflight(system,material,output,guard)
        report['operator_comparison_sha256']=sha256_file(output/'operator-comparison.npz')
        progress('operator_verified',**report['operator'])
        patches=[circular_patch(layout,c,cfg['radius_m'],guard=guard) for c in
            [cfg['source_center_xy_m'],*cfg['receiver_centers_xy_m']]]
        report['patches']=[p.report for p in patches];arrays={}
        for i,p in enumerate(patches):
            for key in ('indices','weights','direction'):arrays[f'patch{i}_{key}']=getattr(p,key)
        atomic_write(output/'mechanical-patches.npz',lambda s:np.savez_compressed(s,**arrays))
        report['mechanical_patches_sha256']=sha256_file(output/'mechanical-patches.npz')
        n=layout.screen['scalar_dofs'];load=patches[0].vector(n);probes=np.zeros((3,3*n))
        for row,p in zip(probes,patches[1:]):row.reshape(3,n)[:,p.indices]=p.direction[:,None]*p.weights
        progress('mechanical_patches_ready')
        last=-math.inf
        def stepped(event):
            nonlocal last
            report['steps_completed']=event['steps_completed'];elapsed=time.perf_counter()-started
            if elapsed-last>=1 or event['steps_completed']==steps:
                progress('propagating',**event);last=elapsed
        before=time.perf_counter()
        result=gpu_explicit_simulate(system,steps,dt,probes,load,guard=guard,progress=stepped,
            force=lambda t:cfg['force_scale_n']*tone_burst(t,cfg['frequency_hz'],cfg['cycles']),
            max_workspace_mib=cfg['max_device_mib'])
        report['propagation_seconds']=time.perf_counter()-before
        atomic_write(output/'mechanical-window.npz',lambda s:np.savez_compressed(s,
            time_s=result.time_s,displacement_m=result.displacement_m,energy_time_s=result.energy_time_s,
            physical_energy_j=result.physical_energy_j,modified_energy_j=result.modified_energy_j,
            final_displacement_m=result.final_displacement_m,final_velocity_m_s=result.final_velocity_m_s,
            mass_diagonal_kg=system.mass_diagonal))
        report['mechanical_window_sha256']=sha256_file(output/'mechanical-window.npz')
        after=result.modified_energy_j[result.energy_time_s>cfg['cycles']/cfg['frequency_hz']+2*dt]
        if len(after)<2 or after.mean()<=0:raise ArithmeticError('tensor postburst energy is missing/nonpositive')
        variation=float(np.ptp(after)/abs(after.mean()))
        if variation>cfg['energy_tolerance'] or np.any(np.linalg.norm(result.displacement_m,axis=0)==0):
            raise ArithmeticError('tensor energy or receiver identifiability failed')
        if result.statistics['steps_completed']!=steps or result.time_s[-1]!=cfg['duration_s']:
            raise ArithmeticError('tensor propagation did not complete the declared window')
        validate_plan(plan,guard)
        report.update(completed=True,passed=True,full_window_completed=True,
            postburst_modified_energy_relative_range=variation,propagation_statistics=result.statistics,
            peak_receiver_displacement_m=np.max(abs(result.displacement_m),axis=0).tolist(),
            scope='Complete mechanical tensor window and operator/energy checks only; no convergence, defect or PZT/physical acceptance.')
    except Exception as error:
        report['error']=f'{type(error).__name__}: {error}';raise
    finally:
        report['wall_seconds']=time.perf_counter()-started;report['resources']=guard.report()
        write_json(output/'wave-report.json',sealed(report))
    return report


def archive_schema(path,expected_hash,spec):
    path=Path(path)
    bound=sum(math.prod(shape)*8+4096 for shape in spec.values())
    if path.stat().st_size>bound+65536 or sha256_file(path)!=expected_hash:
        raise ValueError('tensor raw archive changed or exceeds its compressed bound')
    with zipfile.ZipFile(path) as z:
        entries=z.infolist()
        if len(entries)!=len(spec) or {e.filename for e in entries}!={k+'.npy' for k in spec}:
            raise ValueError('tensor raw archive has missing, duplicate or unknown arrays')
        if any(e.file_size>math.prod(spec[e.filename[:-4]])*8+4096 for e in entries):
            raise MemoryError('tensor raw array exceeds its shape-derived expansion bound')
        for entry in entries:
            with z.open(entry) as stream:
                version=np.lib.format.read_magic(stream)
                if version not in ((1,0),(2,0)):raise ValueError('unsupported tensor NPY header version')
                reader=np.lib.format.read_array_header_1_0 if version==(1,0) else np.lib.format.read_array_header_2_0
                shape,fortran,dtype=reader(stream,max_header_size=4096)
                if (shape!=spec[entry.filename[:-4]] or fortran or dtype.kind not in 'fi' or dtype.itemsize!=8
                        or entry.file_size-stream.tell()!=math.prod(shape)*8):
                    raise ValueError('tensor NPY header or payload does not match its bounded schema')


def load_arrays(path,spec,keys=None):
    selected=set(spec) if keys is None else set(keys);answer={}
    with np.load(path,allow_pickle=False) as arrays:
        for key in selected:
            value=arrays[key]
            if value.shape!=spec[key] or value.dtype.kind not in 'fi' or value.dtype.itemsize!=8 or not np.isfinite(value).all():
                raise ValueError('tensor raw array shape, numeric type or finiteness changed')
            answer[key]=value
    return answer


def read_wave(root,guard=None,*,states=False):
    root=Path(root).resolve()
    plan,parent,native=(checked_receipt(read_json(root/p)) for p in
        ('tensor-wave-plan.json','monitor-result.json','native/wave-report.json'))
    cfg=checked_case(plan['case']);layout=layout_for(cfg);n=layout.screen['vector_dofs'];steps=cfg['steps']
    if (parent['completed'] is not True or parent['passed'] is not True or parent['worker_run']['exit_code']!=0
            or native['completed'] is not True or native['passed'] is not True or native['full_window_completed'] is not True
            or parent['plan_fingerprint']!=plan['fingerprint'] or native['plan_fingerprint']!=plan['fingerprint']
            or parent['native_report_sha256']!=sha256_file(root/'native/wave-report.json')
            or any(r['ready_for_training_data'] is not False or r['training_updates']!=0 for r in (parent,native))
            or native['steps_completed']!=steps or native['planned_steps']!=steps or native['vector_dofs']!=n
            or native['dt_s']!=cfg['duration_s']/steps or native['operator']['relative_l2']>cfg['operator_tolerance']):
        raise ValueError('tensor window receipt is incomplete or incompatible')
    if len(native['patches'])!=4:raise ValueError('tensor window needs one source and three receivers')
    patch_spec={}
    for i,p in enumerate(native['patches']):
        support=p['scalar_support']
        if type(support) is not int or not 1<=support<=256*(cfg['order']+1)**2:
            raise ValueError('tensor patch support exceeds its bounded surface domain')
        patch_spec.update({f'patch{i}_indices':(support,),f'patch{i}_weights':(support,),f'patch{i}_direction':(3,)})
    energy=steps//16+1
    window_spec=dict(time_s=(steps+1,),displacement_m=(steps+1,3),energy_time_s=(energy,),
        physical_energy_j=(energy,),modified_energy_j=(energy,),final_displacement_m=(n,),
        final_velocity_m_s=(n,),mass_diagonal_kg=(n,))
    specs={'operator-comparison.npz':dict(input_field=(n,),gpu_action=(n,),reference_action=(n,)),
        'mechanical-patches.npz':patch_spec,'mechanical-window.npz':window_spec}
    hashes={'operator-comparison.npz':'operator_comparison_sha256','mechanical-patches.npz':'mechanical_patches_sha256',
        'mechanical-window.npz':'mechanical_window_sha256'}
    for name,spec in specs.items():
        if guard:guard.check()
        archive_schema(root/'native'/name,native[hashes[name]],spec)
    keys=set(window_spec) if states else set(window_spec)-{'final_displacement_m','final_velocity_m_s','mass_diagonal_kg'}
    window=load_arrays(root/'native/mechanical-window.npz',window_spec,keys)
    patches=load_arrays(root/'native/mechanical-patches.npz',patch_spec)
    for i in range(4):
        ids,w,d=(patches[f'patch{i}_{key}'] for key in ('indices','weights','direction'))
        if (ids.dtype.kind!='i' or np.any(ids<0) or np.any(ids>=n//3) or not np.all(np.diff(ids)>0)
                or not np.all(ids%(cfg['cells'][2]*cfg['order']+1)==cfg['cells'][2]*cfg['order'])
                or abs(w.sum()-1)>1e-10 or not np.array_equal(d,[0.,0.,1.])):
            raise ValueError('tensor sparse mechanical covector is invalid')
    dt=native['dt_s'];expected=np.arange(steps+1)*dt
    if not np.array_equal(window['time_s'],expected) or expected[-1]!=cfg['duration_s']:
        raise ValueError('tensor waveform is missing exact full-window samples')
    if not np.array_equal(window['energy_time_s'],np.arange(0,steps+1,16)*dt):
        raise ValueError('tensor energy samples changed')
    after=window['modified_energy_j'][window['energy_time_s']>cfg['cycles']/cfg['frequency_hz']+2*dt]
    if len(after)<2 or after.mean()<=0 or np.ptp(after)/after.mean()>cfg['energy_tolerance']:
        raise ValueError('tensor raw postburst energy fails the declared check')
    paths=['tensor-wave-plan.json','monitor-result.json','source-snapshot.zip','native/wave-report.json',
        *['native/'+name for name in specs]]
    if sha256_file(root/'source-snapshot.zip')!=parent['source_snapshot_sha256']:
        raise ValueError('tensor source snapshot changed')
    return dict(root=root,plan=plan,parent=parent,native=native,layout=layout,window=window,patches=patches,
        input_identities={p:sha256_file(root/p) for p in paths})


def run_case(cfg,output,study,guard):
    cfg=checked_case(cfg);output=Path(output).resolve()
    if not output.is_relative_to(PROJECT/'outputs') or (output.exists() and any(output.iterdir())):
        raise ValueError('tensor wave run needs a fresh project output directory')
    output.mkdir(parents=True,exist_ok=True)
    result=dict(completed=False,passed=False,ready_for_training_data=False,training_updates=0)
    try:
        plan=sealed(dict(kind='tensor_mechanical_window',case=cfg,study=study,**identities(),
            runtime=distribution_identity(guard),gpu_packages=gpu_package_identity(guard)))
        validate_plan(plan,guard);write_json(output/'tensor-wave-plan.json',plan)
        snapshot_sources(output/'source-snapshot.zip',plan)
        result.update(plan_fingerprint=plan['fingerprint'],source_snapshot_sha256=sha256_file(output/'source-snapshot.zip'))
        temporary,cache=output/'temporary',output/'kernel-cache';temporary.mkdir();cache.mkdir()
        command=[sys.executable,'-m','examples.tensor_wave','--worker','--plan',str(output/'tensor-wave-plan.json'),
            '--output',str(output/'native'),'--max-seconds',str(max(1.,guard.limits.max_seconds-(time.monotonic()-guard.started))),
            '--max-rss-mib',str(guard.limits.max_rss_mib)]
        env={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','CUPY_CACHE_DIR':str(cache),
            'CUPY_CACHE_SAVE_CUDA_SOURCE':'1','TEMP':str(temporary),'TMP':str(temporary),'TMPDIR':str(temporary)}
        result['worker_run']=run_monitored(command,PROJECT,env,guard,output/'worker.log')
        if result['worker_run']['exit_code']!=0:raise RuntimeError('tensor worker stopped; inspect worker log/report')
        native=checked_receipt(read_json(output/'native/wave-report.json'))
        if native['passed'] is not True or native['full_window_completed'] is not True or native['plan_fingerprint']!=plan['fingerprint']:
            raise ValueError('tensor worker returned an incomplete window')
        validate_plan(plan,guard)
        result.update(completed=True,passed=True,steps_completed=native['steps_completed'],vector_dofs=native['vector_dofs'],
            full_window_completed=True,native_report_sha256=sha256_file(output/'native/wave-report.json'))
    except Exception as error:result['error']=f'{type(error).__name__}: {error}'
    finally:
        result['resources']=guard.report();write_json(output/'monitor-result.json',sealed(result))
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study',type=Path);parser.add_argument('--case');parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--output',type=Path);parser.add_argument('--plan',type=Path)
    parser.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    parser.add_argument('--max-seconds',type=float,default=1800);parser.add_argument('--max-rss-mib',type=float,default=2048)
    args=parser.parse_args();guard=ResourceGuard(ResourceLimits(max_seconds=args.max_seconds,max_rss_mib=args.max_rss_mib,
        max_system_memory_percent=85,min_available_mib=2048))
    if args.worker:return 0 if worker(args.plan,args.output,guard)['passed'] else 2
    if args.smoke:
        if args.study or args.case or not args.output:parser.error('smoke needs only a new output directory')
        cfg,output,binding=smoke_case(),args.output,None
    else:
        from examples.tensor_wave_study import read_study
        if not args.study or not args.case:parser.error('research run needs --study and --case')
        study=read_study(args.study,guard);entry=study['cases'][args.case]
        cfg,output=entry['case'],Path(entry['directory'])
        if args.output and args.output.resolve()!=output:parser.error('output differs from the frozen study')
        binding=dict(path=str(args.study.resolve()),sha256=sha256_file(args.study),fingerprint=study['fingerprint'],case_name=args.case)
    result=run_case(cfg,output,binding,guard)
    print(json.dumps({k:v for k,v in result.items() if k not in ('resources','worker_run')}))
    return 0 if result['passed'] else 1


if __name__=='__main__':raise SystemExit(main())
