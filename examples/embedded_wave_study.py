"""Compare predeclared full-window spatial and time refinements."""
import argparse
import json
from pathlib import Path

import numpy as np

from examples.embedded_wave import read_wave, validate_plan
from examples.tensor_wave import PROJECT, checked_receipt, read_json, sealed
from traxis.resources import ResourceGuard, ResourceLimits
from traxis.storage import sha256_file, write_json

PROTOCOL = dict(kind='embedded_complete_window_refinement', duration_s=.0005,
    spatial_orders=[3,4,5],spatial_steps=196608,time_order=4,time_steps=[131072,196608,294912],
    comparison_intervals=32768,spatial_finest_change_limit=.01,time_finest_change_limit=.001,
    time_order_bounds=[1.7,2.3],relative_identifiability_floor=1e-11)


def compare(times, traces, mode, protocol):
    if protocol != PROTOCOL or mode not in ('spatial','time') or len(times)!=3 or len(traces)!=3:
        raise ValueError('embedded comparison needs the fixed protocol and three levels')
    counts=[protocol['spatial_steps']]*3 if mode=='spatial' else protocol['time_steps']
    common=np.arange(protocol['comparison_intervals']+1)*(protocol['duration_s']/protocol['comparison_intervals'])
    aligned=[]
    for t,y,n in zip(times,traces,counts):
        t,y=np.asarray(t),np.asarray(y)
        if (t.shape!=(n+1,) or y.shape!=(n+1,3) or np.iscomplexobj(t) or np.iscomplexobj(y)
                or not np.isfinite(t).all() or not np.isfinite(y).all()
                or not np.array_equal(t,np.arange(n+1)*(protocol['duration_s']/n))):
            raise ValueError('embedded comparison requires unchanged complete raw grids')
        stride=n//protocol['comparison_intervals']
        # Integer strides select the same physical times. Allow arithmetic
        # roundoff only; no interpolation, amplitude fit or phase alignment.
        if n%protocol['comparison_intervals'] or np.max(abs(t[::stride]-common))>8*np.finfo(float).eps*protocol['duration_s']:
            raise ValueError('embedded levels do not share the declared physical sample indices')
        aligned.append(y[::stride])
    coarse,fine=(np.linalg.norm(aligned[i]-aligned[i+1],axis=0) for i in (0,1))
    norms=np.linalg.norm(aligned[-1],axis=0); rows=[]
    for i in range(3):
        if not np.isfinite([coarse[i],fine[i],norms[i]]).all(): raise ValueError('embedded norms overflowed')
        identified=bool(norms[i]>0 and min(coarse[i],fine[i])>protocol['relative_identifiability_floor']*norms[i])
        change=float(fine[i]/norms[i]) if norms[i]>0 else None
        ratio=float(coarse[i]/fine[i]) if identified else None
        order=float(np.log(ratio)/np.log(1.5)) if mode=='time' and identified else None
        passed=identified and fine[i]<coarse[i] and change<protocol[mode+'_finest_change_limit']
        if mode=='time': passed=passed and protocol['time_order_bounds'][0]<=order<=protocol['time_order_bounds'][1]
        rows.append(dict(receiver=i,coarse_difference_l2_m=float(coarse[i]),fine_difference_l2_m=float(fine[i]),
            finest_waveform_l2_m=float(norms[i]),rate_identifiable=identified,finest_relative_change=change,
            difference_ratio=ratio,observed_time_order=order,passed=bool(passed)))
    return common,np.asarray(aligned),rows


def assess(roots, cpu_roots, mode, protocol_path, output, guard):
    output=Path(output).resolve(); protocol_path=Path(protocol_path).resolve()
    if not output.is_relative_to(PROJECT/'outputs') or len(roots)!=3 or len(cpu_roots)!=3:
        raise ValueError('embedded assessment requires three runs/CPU checks and fresh project output')
    output.mkdir(parents=True,exist_ok=False); protocol=read_json(protocol_path)
    if protocol!=PROTOCOL: raise ValueError('embedded refinement protocol changed')
    analysis_hash=sha256_file(__file__); protocol_hash=sha256_file(protocol_path)
    report=dict(completed=False,passed=False,ready_for_training_data=False,training_updates=0,mode=mode,
        protocol=protocol,protocol_sha256=protocol_hash,analysis_sha256=analysis_hash)
    try:
        levels=[]; checks=[]
        for root,cpu_root in zip(roots,cpu_roots):
            guard.check(); level=read_wave(root,guard); validate_plan(level['plan'],guard)
            if sha256_file(level['root']/'source-snapshot.zip')!=level['plan']['source_snapshot_sha256']:
                raise ValueError('embedded waveform source snapshot changed')
            path=Path(cpu_root)/'cpu-window-check.json'; cpu=checked_receipt(read_json(path))
            if (not cpu['completed'] or not cpu['passed'] or cpu['plan_fingerprint']!=level['plan']['fingerprint']
                    or cpu['ready_for_training_data'] is not False or cpu['training_updates']!=0
                    or cpu['forward_result_fingerprint']!=level['parent']['fingerprint']
                    or cpu['input_identities']!=level['input_identities']
                    or cpu['source_sha256']!=level['plan']['source_sha256']
                    or cpu['checker_sha256']!=sha256_file(PROJECT/'examples/verify_embedded_window.py')):
                raise ValueError('embedded independent CPU receipt is missing or incompatible')
            levels.append(level); checks.append(dict(path=str(path),sha256=sha256_file(path)))
        cases=[level['case'] for level in levels]
        pairs=list(zip(protocol['spatial_orders'],[protocol['spatial_steps']]*3)) if mode=='spatial' else [(protocol['time_order'],n) for n in protocol['time_steps']]
        if [(c['order'],c['steps']) for c in cases]!=pairs or any(c['scope']!='full' for c in cases):
            raise ValueError('embedded comparison changed the declared degree/time levels')
        varying={'order'} if mode=='spatial' else {'steps'}
        fixed=lambda c:{k:v for k,v in c.items() if k not in varying}
        if any(fixed(c)!=fixed(cases[0]) for c in cases[1:]): raise ValueError('embedded comparison changed the physical case')
        keys=('name','compute_capability','total_bytes','cupy_version','precision')
        if any(any(level['native']['device'][k]!=levels[0]['native']['device'][k] for k in keys) for level in levels[1:]):
            raise ValueError('embedded comparison changed device or precision')
        if mode=='time' and any(any(not np.array_equal(value,level['patches'][key]) for key,value in levels[0]['patches'].items()) for level in levels[1:]):
            raise ValueError('embedded time comparison changed discrete load/probe covectors')
        t,y,rows=compare([l['window']['time_s'] for l in levels],[l['window']['displacement_m'] for l in levels],mode,protocol)
        np.savez_compressed(output/'comparison.npz',time_s=t,traces_m=y)
        for level in levels:
            if any(sha256_file(level['root']/name)!=digest for name,digest in level['input_identities'].items()):
                raise ValueError('embedded raw evidence changed during assessment')
        if sha256_file(__file__)!=analysis_hash or sha256_file(protocol_path)!=protocol_hash:
            raise ValueError('embedded assessment source/protocol changed')
        report.update(completed=True,passed=all(row['passed'] for row in rows),receivers=rows,cpu_checks=checks,
            levels=[dict(directory=str(l['root']),plan_fingerprint=l['plan']['fingerprint'],input_identities=l['input_identities']) for l in levels],
            comparison_sha256=sha256_file(output/'comparison.npz'),
            scope='Complete embedded mechanical-window refinement; no calibrated PZT or physical-data acceptance.')
    except Exception as error:
        report['error']=f'{type(error).__name__}: {error}'; raise
    finally:
        report['resources']=guard.report(); write_json(output/(mode+'-refinement-report.json'),sealed(report))
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('mode',choices=('spatial','time'))
    parser.add_argument('--runs',nargs=3,type=Path,required=True);parser.add_argument('--cpu-checks',nargs=3,type=Path,required=True)
    parser.add_argument('--protocol',type=Path,default=PROJECT/'configs/embedded-wave-protocol.json')
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    guard=ResourceGuard(ResourceLimits(max_seconds=180,max_rss_mib=1024,max_system_memory_percent=85,min_available_mib=2048))
    result=assess(args.runs,args.cpu_checks,args.mode,args.protocol,args.output,guard)
    print(json.dumps({k:result[k] for k in ('completed','passed','receivers','fingerprint')}))
    return 0 if result['passed'] else 2


if __name__=='__main__': raise SystemExit(main())
