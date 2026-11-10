"""Freeze and assess complete tensor p-refinement and time-refinement windows."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from examples.tensor_wave import (PROJECT,PHYSICS,checked_case,checked_receipt,identities,read_json,
    read_wave,sealed,snapshot_sources,validate_plan)
from traxis.config import Plate,Material
from traxis.gpu_verification import gpu_package_identity
from traxis.highorder_verification import distribution_identity
from traxis.resources import ResourceGuard,ResourceLimits
from traxis.storage import atomic_write,sha256_file,write_json


def checked_protocol(cfg):
    expected=dict(kind='tensor_fullplate_wave_study',plate=asdict(Plate()),material=asdict(Material()),cells=[150,150,1],
        source_center_xy_m=[.02,.02],receiver_centers_xy_m=[[.25,.25],[.48,.48],[.35,.20]],
        spatial_orders=[3,4,5],spatial_steps=65536,time_order=4,time_steps=[32768,65536,131072],
        spatial_finest_change_limit=.05,require_decreasing_spatial_differences=True,time_ratio_bounds=[3.8,4.2],
        time_finest_change_limit=.01,relative_identifiability_floor=1e-11,**PHYSICS)
    if cfg!=expected:raise ValueError('tensor study must retain the declared controls, levels and raw comparison criteria')
    return cfg


def case_for(protocol,order,steps):
    fields=set(PHYSICS)|{'plate','material','cells','source_center_xy_m','receiver_centers_xy_m'}
    return checked_case(dict(kind='tensor_mechanical_window_case',scope='research_intact',order=order,steps=steps,
        **{k:protocol[k] for k in fields}))


def case_levels(protocol):
    return sorted(set([(p,protocol['spatial_steps']) for p in protocol['spatial_orders']]
        +[(protocol['time_order'],n) for n in protocol['time_steps']]))


def read_study(path,guard=None):
    path=Path(path).resolve();study=checked_receipt(read_json(path));cfg=checked_protocol(study['protocol'])
    expected={f'Q{p}-N{n}':case_for(cfg,p,n) for p,n in case_levels(cfg)}
    if (study['kind']!='frozen_tensor_wave_study' or set(study['cases'])!=set(expected)
            or any(study['cases'][name]['case']!=case for name,case in expected.items())
            or any(study[k]!=v for k,v in identities().items()) or study['runtime']!=distribution_identity(guard)
            or study['gpu_packages']!=gpu_package_identity(guard)):
        raise ValueError('tensor study cases, source or runtime changed')
    for name,entry in study['cases'].items():
        if Path(entry['directory'])!=path.parent/'runs'/name or Path(entry['cpu_directory'])!=path.parent/'cpu'/name:
            raise ValueError('tensor study output path changed')
    if sha256_file(path.parent/'source-snapshot.zip')!=study['source_snapshot_sha256']:
        raise ValueError('tensor study source snapshot changed')
    return study


def prepare(protocol_path,output,guard):
    output=Path(output).resolve();cfg=checked_protocol(read_json(protocol_path))
    if not output.is_relative_to(PROJECT/'outputs') or (output.exists() and any(output.iterdir())):
        raise ValueError('tensor study needs a fresh project output directory')
    output.mkdir(parents=True,exist_ok=True)
    study=dict(kind='frozen_tensor_wave_study',protocol=cfg,protocol_sha256=sha256_file(protocol_path),
        **identities(),runtime=distribution_identity(guard),gpu_packages=gpu_package_identity(guard),
        ready_for_training_data=False,training_updates=0,cases={})
    for p,n in case_levels(cfg):
        name=f'Q{p}-N{n}';study['cases'][name]=dict(case=case_for(cfg,p,n),
            directory=str(output/'runs'/name),cpu_directory=str(output/'cpu'/name))
    snapshot_sources(output/'source-snapshot.zip',study)
    study['source_snapshot_sha256']=sha256_file(output/'source-snapshot.zip')
    write_json(output/'tensor-study-plan.json',sealed(study))
    return study


def compare_raw(times,traces,protocol,mode):
    cfg=checked_protocol(protocol)
    counts=[cfg['spatial_steps']]*3 if mode=='spatial' else cfg['time_steps'] if mode=='time' else None
    if counts is None or len(times)!=3 or len(traces)!=3:raise ValueError('tensor comparison needs three declared levels')
    aligned=[]
    for n,t,y in zip(counts,times,traces):
        t,y=np.asarray(t),np.asarray(y)
        if (t.shape!=(n+1,) or y.shape!=(n+1,3) or np.iscomplexobj(t) or np.iscomplexobj(y)
                or not np.isfinite(t).all() or not np.isfinite(y).all()
                or not np.array_equal(t,np.arange(n+1)*(cfg['duration_s']/n))):
            raise ValueError('tensor comparison needs unchanged finite complete time grids and three receivers')
        factor=n//counts[0]
        if not np.array_equal(t[::factor],times[0]):raise ValueError('tensor levels do not share exact time samples')
        aligned.append(y[::factor])
    coarse,fine=(np.linalg.norm(aligned[i]-aligned[i+1],axis=0) for i in (0,1))
    norms=np.linalg.norm(aligned[-1],axis=0);rows=[]
    for i in range(3):
        if not np.isfinite([coarse[i],fine[i],norms[i]]).all():raise ValueError('tensor comparison norms overflowed')
        identified=bool(norms[i]>0 and min(coarse[i],fine[i])>cfg['relative_identifiability_floor']*norms[i])
        change=float(fine[i]/norms[i]) if norms[i]>0 else None
        ratio=float(coarse[i]/fine[i]) if identified else None
        if mode=='spatial':passed=identified and fine[i]<coarse[i] and change<cfg['spatial_finest_change_limit']
        else:passed=identified and cfg['time_ratio_bounds'][0]<=ratio<=cfg['time_ratio_bounds'][1] and change<cfg['time_finest_change_limit']
        rows.append(dict(receiver=i,coarse_difference_l2_m=float(coarse[i]),fine_difference_l2_m=float(fine[i]),
            finest_waveform_l2_m=float(norms[i]),rate_identifiable=identified,difference_ratio=ratio,
            finest_relative_change=change,observed_time_order=float(np.log2(ratio)) if mode=='time' and identified else None,
            passed=bool(passed)))
    return rows,np.asarray(aligned)


def assess(path,mode,output,guard):
    path=Path(path).resolve();output=Path(output).resolve();study=read_study(path,guard);cfg=study['protocol']
    if output.exists() and any(output.iterdir()):raise FileExistsError('tensor assessment needs a fresh directory')
    output.mkdir(parents=True,exist_ok=True)
    report=dict(completed=False,passed=False,ready_for_training_data=False,training_updates=0,
        study_fingerprint=study['fingerprint'],study_sha256=sha256_file(path),mode=mode)
    try:
        pairs=[(p,cfg['spatial_steps']) for p in cfg['spatial_orders']] if mode=='spatial' else [(cfg['time_order'],n) for n in cfg['time_steps']]
        levels=[];checks=[]
        for p,n in pairs:
            name=f'Q{p}-N{n}';entry=study['cases'][name];level=read_wave(entry['directory'],guard)
            validate_plan(level['plan'],guard)
            binding=level['plan']['study']
            if binding is None or binding['case_name']!=name or binding['sha256']!=sha256_file(path):
                raise ValueError('tensor waveform does not belong to this frozen study case')
            cpu_path=Path(entry['cpu_directory'])/'cpu-window-check.json';cpu=checked_receipt(read_json(cpu_path))
            if (cpu['completed'] is not True or cpu['passed'] is not True or cpu['plan_fingerprint']!=level['plan']['fingerprint']
                    or cpu['forward_result_fingerprint']!=level['parent']['fingerprint']
                    or cpu['input_identities']!=level['input_identities'] or cpu['checker_sha256']!=sha256_file(PROJECT/'examples/verify_tensor_window.py')):
                raise ValueError('independent CPU tensor state check is missing or incompatible')
            levels.append(level);checks.append(dict(path=str(cpu_path),sha256=sha256_file(cpu_path)))
        first=levels[0]
        for level in levels[1:]:
            if level['native']['device']!=first['native']['device']:
                # Free VRAM is observational; compare physical device/precision fields only.
                keys=('name','compute_capability','total_bytes','cupy_version','precision')
                if any(level['native']['device'][k]!=first['native']['device'][k] for k in keys):
                    raise ValueError('tensor study changed the CUDA device or precision')
            common=lambda c:{k:v for k,v in c.items() if k!=('order' if mode=='spatial' else 'steps')}
            if common(level['plan']['case'])!=common(first['plan']['case']):raise ValueError('tensor study changed the physical problem')
            if mode=='time' and any(not np.array_equal(v,level['patches'][k]) for k,v in first['patches'].items()):
                raise ValueError('tensor time study changed discrete mechanical footprints')
        rows,aligned=compare_raw([l['window']['time_s'] for l in levels],[l['window']['displacement_m'] for l in levels],cfg,mode)
        atomic_write(output/'comparison.npz',lambda s:np.savez_compressed(s,time_s=first['window']['time_s'],traces_m=aligned))
        for level in levels:
            if any(sha256_file(level['root']/n)!=h for n,h in level['input_identities'].items()):raise ValueError('tensor inputs changed during assessment')
        read_study(path,guard)
        report.update(completed=True,passed=all(r['passed'] for r in rows),receivers=rows,cpu_checks=checks,
            levels=[dict(directory=str(l['root']),input_identities=l['input_identities']) for l in levels],
            comparison_sha256=sha256_file(output/'comparison.npz'),
            scope='Raw complete-window p-refinement or time refinement for the fixed intact mechanical experiment; no defect/PZT/physical acceptance.')
    except Exception as error:report['error']=f'{type(error).__name__}: {error}';raise
    finally:
        report['resources']=guard.report();write_json(output/f'{mode}-refinement-report.json',sealed(report))
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('mode',choices=('prepare','spatial','time'))
    parser.add_argument('--protocol',type=Path,default=PROJECT/'configs/tensor-wave-protocol.json')
    parser.add_argument('--plan',type=Path);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    guard=ResourceGuard(ResourceLimits(max_seconds=180,max_rss_mib=1536,max_system_memory_percent=85,min_available_mib=2048))
    result=prepare(args.protocol,args.output,guard) if args.mode=='prepare' else assess(args.plan,args.mode,args.output,guard)
    print(json.dumps({k:result[k] for k in ('kind','fingerprint') if k in result} if args.mode=='prepare'
        else {k:result[k] for k in ('completed','passed','receivers')}))
    return 0 if args.mode=='prepare' or result['passed'] else 2


if __name__=='__main__':raise SystemExit(main())
