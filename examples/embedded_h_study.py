"""Assess the predeclared finer-cell controls and their defect-induced traces."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from examples.embedded_wave import case_config, checked_case
from examples.embedded_wave_study import checked_resources
from examples.embedded_wave_evidence import check_integrator_transition, validate_saved_plan
from examples.tensor_wave import PROJECT, checked_receipt, read_json, sealed
from examples.verify_green_synthesis import reference_window as checked_cpu_window
from examples.verify_embedded_h_window import cpu_limits
from traxis.resources import ResourceGuard, ResourceLimits
from traxis.storage import atomic_write, canonical_hash, sha256_file, write_json


PROTOCOL_SHA256='59d853cd1ad1fc32691a154b4f277084fbf177c125087d2ccc0607dd2ba17adc'
PROTOCOL_FINGERPRINT='60aed16f43005c5929d68c2fd3425dce98e3bd2d187071e81a06379a7a75d36a'


def load_protocol(path):
    protocol=read_json(path)
    if sha256_file(path)!=PROTOCOL_SHA256 or canonical_hash(protocol)!=PROTOCOL_FINGERPRINT:
        raise ValueError('the predeclared h-refinement protocol changed')
    return protocol


def checked_level(case,index,defective):
    actual=checked_case(case)
    expected=case_config('h_refinement',4,294912,defective,h_level=index)
    if actual==expected:return expected
    # The exact original coarse control has the same mesh/physics. Its older,
    # stricter allocation budgets remain in its original plan and receipt.
    if index==0 and actual==case_config('full',4,294912,defective):return expected
    raise ValueError('h comparison changed a declared level or its material state')


def checked_launcher(case,cpu_root):
    if case['scope']!='h_refinement':return None
    path=Path(cpu_root)/'cpu-launcher.json';receipt=checked_receipt(read_json(path))
    proof=Path(cpu_root)/'cpu-window-check.json'
    cpu=checked_receipt(read_json(proof))
    if (receipt.get('kind')!='embedded_h_cpu_launcher' or not receipt['completed'] or not receipt['passed']
            or receipt['ready_for_training_data'] or receipt['training_updates']!=0
            or receipt['cpu_proof_sha256']!=sha256_file(proof)
            or receipt['launcher_sha256']!=sha256_file(PROJECT/'examples/verify_embedded_h_window.py')
            or receipt['limits']!=asdict(cpu_limits()) or cpu['resources']['limits']!=receipt['limits']):
        raise ValueError('h CPU launcher is missing its passing bound proof, source or resource envelope')
    return dict(path=str(path.resolve()),sha256=sha256_file(path),fingerprint=receipt['fingerprint'])


def compare_h(times,traces,protocol):
    if canonical_hash(protocol)!=PROTOCOL_FINGERPRINT or len(times)!=3 or len(traces)!=3:
        raise ValueError('h comparison needs its unchanged protocol and three raw levels')
    n=protocol['steps'];intervals=protocol['comparison_intervals'];stride=n//intervals
    common=np.arange(intervals+1)*(protocol['duration_s']/intervals)
    raw_grid=np.arange(n+1)*(protocol['duration_s']/n);aligned=[]
    if n%intervals:raise ValueError('h comparison requires integer sample selection')
    for time_s,values in zip(times,traces):
        t,y=np.asarray(time_s),np.asarray(values)
        if (t.shape!=(n+1,) or y.shape!=(n+1,3) or t.dtype!=np.float64 or y.dtype!=np.float64
                or not np.isfinite(t).all() or not np.isfinite(y).all() or not np.array_equal(t,raw_grid)
                or np.max(abs(t[::stride]-common))>8*np.finfo(float).eps*protocol['duration_s']):
            raise ValueError('h comparison requires complete float64 physical sample grids')
        aligned.append(y[::stride])
    coarse,fine=(np.linalg.norm(aligned[i]-aligned[i+1],axis=0) for i in (0,1))
    norms=np.linalg.norm(aligned[-1],axis=0);rows=[]
    if not np.isfinite([coarse,fine,norms]).all():raise ValueError('h comparison norms overflowed')
    for i in range(3):
        identifiable=bool(norms[i]>0 and min(coarse[i],fine[i])>protocol['relative_identifiability_floor']*norms[i])
        change=float(fine[i]/norms[i]) if norms[i]>0 else None
        passed=identifiable and fine[i]<coarse[i] and change<protocol['finest_relative_change_limit']
        rows.append(dict(receiver=i,coarse_difference_l2_m=float(coarse[i]),fine_difference_l2_m=float(fine[i]),
            finest_waveform_l2_m=float(norms[i]),rate_identifiable=identifiable,finest_relative_change=change,
            difference_ratio=float(coarse[i]/fine[i]) if identifiable else None,passed=bool(passed)))
    return common,np.asarray(aligned),rows


def compare_pairs(intact,recessed,protocol):
    if len(intact)!=3 or len(recessed)!=3:raise ValueError('h scattering needs three paired levels')
    for i,(a,b) in enumerate(zip(intact,recessed)):
        checked_level(a['case'],i,False);checked_level(b['case'],i,True)
        if not np.array_equal(a['window']['time_s'],b['window']['time_s']):
            raise ValueError('h scattering pair has different physical time grids')
    times=[a['window']['time_s'] for a in intact]
    fields={name:[level['window']['displacement_m'] for level in levels]
            for name,levels in [('intact',intact),('recessed',recessed)]}
    fields['scattering']=[b-a for a,b in zip(fields['intact'],fields['recessed'])]
    aligned={};results={}
    for name,values in fields.items():
        time_s,aligned[name],results[name]=compare_h(times,values,protocol)
    return time_s,aligned,results


def assess(intact_roots,intact_cpu,recessed_roots,recessed_cpu,protocol_path,output,guard,source_check=None):
    output=Path(output).resolve();protocol_path=Path(protocol_path).resolve()
    roots=[*intact_roots,*recessed_roots];checks=[*intact_cpu,*recessed_cpu]
    if (not output.is_relative_to(PROJECT/'outputs') or len(roots)!=6 or len(checks)!=6
            or any(len(group)!=3 for group in (intact_roots,intact_cpu,recessed_roots,recessed_cpu))
            or len({Path(p).resolve() for p in roots})!=6):
        raise ValueError('h study requires six distinct windows, six CPU checks and a fresh project output')
    protocol=load_protocol(protocol_path)
    sources={name:sha256_file(PROJECT/name) for name in ('examples/embedded_h_study.py',
        'examples/embedded_wave_study.py','examples/embedded_wave_evidence.py','examples/verify_green_synthesis.py',
        'examples/verify_embedded_h_window.py')}
    output.mkdir(parents=True,exist_ok=False)
    report=dict(kind='embedded_h_refinement',completed=False,passed=False,ready_for_training_data=False,
        training_updates=0,protocol=protocol,protocol_sha256=PROTOCOL_SHA256,analysis_sources=sources,
        scope='Fixed-Q4 raw intact/recessed/scattering spatial screen; no PZT or physical-data acceptance.')
    try:
        levels=[];bindings=[]
        for i,(root,cpu) in enumerate(zip(roots,checks)):
            guard.check();level,binding=checked_cpu_window(root,cpu,guard)
            checked_level(level['case'],i%3,i>=3);checked_resources(level)
            binding['cpu_launcher']=checked_launcher(level['case'],cpu)
            levels.append(level);bindings.append(binding)
        transition=check_integrator_transition([b['source_check'] for b in bindings],source_check)
        keys=('name','compute_capability','total_bytes','cupy_version','precision')
        if any(any(level['native']['device'][k]!=levels[0]['native']['device'][k] for k in keys) for level in levels[1:]):
            raise ValueError('h comparison changed device or precision')
        time_s,traces,rows=compare_pairs(levels[:3],levels[3:],protocol)
        atomic_write(output/'comparison.npz',lambda f:np.savez_compressed(f,time_s=time_s,
            **{name+'_m':values for name,values in traces.items()}))
        for level,binding in zip(levels,bindings):
            if (any(sha256_file(level['root']/name)!=h for name,h in binding['input_identities'].items())
                    or sha256_file(binding['cpu_path'])!=binding['cpu_sha256']
                    or (binding['cpu_launcher'] is not None and
                        sha256_file(binding['cpu_launcher']['path'])!=binding['cpu_launcher']['sha256'])
                    or validate_saved_plan(level['plan'],level['root'],guard)!=binding['source_check']):
                raise ValueError('h study forward, CPU or source evidence changed')
        if (load_protocol(protocol_path)!=protocol or any(sha256_file(PROJECT/name)!=h for name,h in sources.items())
                or (transition is not None and sha256_file(transition['path'])!=transition['sha256'])):
            raise ValueError('h study protocol, analysis or source transition changed')
        report.update(completed=True,passed=all(row['passed'] for group in rows.values() for row in group),
            receivers=rows,levels=bindings,integrator_transition=transition,
            comparison_sha256=sha256_file(output/'comparison.npz'))
    except Exception as error:
        report['error']=f'{type(error).__name__}: {error}';raise
    finally:
        report['resources']=guard.report();write_json(output/'h-refinement-report.json',sealed(report))
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('intact-runs','intact-cpu-checks','recessed-runs','recessed-cpu-checks'):
        parser.add_argument('--'+name,nargs=3,type=Path,required=True)
    parser.add_argument('--source-check',type=Path)
    parser.add_argument('--protocol',type=Path,default=PROJECT/'configs/embedded-h-protocol.json')
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    guard=ResourceGuard(ResourceLimits(max_seconds=300,max_rss_mib=1536,max_system_memory_percent=89.9,min_available_mib=768))
    report=assess(args.intact_runs,args.intact_cpu_checks,args.recessed_runs,args.recessed_cpu_checks,
        args.protocol,args.output,guard,args.source_check)
    print(json.dumps({k:report[k] for k in ('completed','passed','receivers','fingerprint')}))
    return 0 if report['passed'] else 2


if __name__=='__main__':raise SystemExit(main())
