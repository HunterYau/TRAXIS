"""Check raw recessed-minus-intact signals on the declared refinement grids."""
import argparse
import json
from pathlib import Path

import numpy as np

from examples.embedded_wave import checked_case, read_wave
from examples.embedded_wave_study import PROTOCOL, assess, compare
from examples.tensor_wave import PROJECT, read_json, sealed
from traxis.resources import ResourceGuard, ResourceLimits
from traxis.storage import atomic_write, sha256_file, write_json


SCATTER_PROTOCOL = dict(kind='embedded_complete_window_scattering_refinement',
    signal='recessed_minus_intact_at_each_degree_and_time_grid',
    parent_protocol=PROTOCOL,
    require_both_total_waveform_screens=True,
    amplitude_fit=False, phase_alignment=False, interpolation=False)


def paired_difference(intact, recessed, mode):
    """Subtract matched physical controls before applying the fixed screen."""
    if mode not in ('spatial','time') or len(intact)!=3 or len(recessed)!=3:
        raise ValueError('scattering refinement needs three intact/recessed pairs')
    expected=(list(zip(PROTOCOL['spatial_orders'],[PROTOCOL['spatial_steps']]*3)) if mode=='spatial'
              else [(PROTOCOL['time_order'],n) for n in PROTOCOL['time_steps']])
    if [(level['case']['order'],level['case']['steps']) for level in intact]!=expected:
        raise ValueError('scattering refinement changed the declared degree/time levels')
    times=[];traces=[]
    for plain,damaged in zip(intact,recessed):
        a,b=checked_case(plain['case']),checked_case(damaged['case'])
        if (a['scope']!='full' or a['defective'] or not b['defective']
                or {k:v for k,v in a.items() if k!='defective'}!={k:v for k,v in b.items() if k!='defective'}):
            raise ValueError('scattering pair changed a physical or numerical setting beyond material loss')
        ta,tb=plain['window']['time_s'],damaged['window']['time_s']
        ya,yb=plain['window']['displacement_m'],damaged['window']['displacement_m']
        if not np.array_equal(ta,tb) or ya.shape!=yb.shape:
            raise ValueError('scattering pair requires identical complete time grids and receiver shapes')
        times.append(ta);traces.append(yb-ya)
    return compare(times,traces,mode,PROTOCOL)


def assess_scattering(intact_roots, intact_cpu, recessed_roots, recessed_cpu,
                      mode, protocol_path, output, guard, source_check=None):
    output=Path(output).resolve();protocol_path=Path(protocol_path).resolve()
    if not output.is_relative_to(PROJECT/'outputs'):
        raise ValueError('scattering proof must remain in project outputs')
    protocol=read_json(protocol_path)
    if protocol!=SCATTER_PROTOCOL:
        raise ValueError('scattering refinement protocol changed')
    sources={name:sha256_file(PROJECT/name) for name in
        ('examples/embedded_scattering_study.py','examples/embedded_wave_study.py')}
    protocol_hash=sha256_file(protocol_path)
    output.mkdir(parents=True,exist_ok=False)
    report=dict(kind='embedded_scattering_refinement',completed=False,passed=False,
        ready_for_training_data=False,training_updates=0,mode=mode,
        protocol=protocol,protocol_sha256=protocol_hash,analysis_sources=sources,
        scope='Raw mechanical scattering convergence; no PZT or physical-data acceptance.')
    try:
        parents=[];parent_reports=[]
        for name,roots,cpu in [('intact',intact_roots,intact_cpu),('recessed',recessed_roots,recessed_cpu)]:
            parent=assess(roots,cpu,mode,PROJECT/'configs/embedded-wave-protocol.json',
                          output/name,guard,source_check)
            parent_reports.append(parent)
            path=output/name/(mode+'-refinement-report.json')
            parents.append(dict(state=name,path=str(path),sha256=sha256_file(path),
                fingerprint=parent['fingerprint'],passed=parent['passed']))
        levels=[[read_wave(root,guard) for root in roots] for roots in (intact_roots,recessed_roots)]
        for group,parent in zip(levels,parent_reports):
            for level,binding in zip(group,parent['levels']):
                if (str(level['root'])!=binding['directory'] or level['plan']['fingerprint']!=binding['plan_fingerprint']
                        or level['input_identities']!=binding['input_identities']):
                    raise ValueError('scattering inputs differ from the CPU-checked total-waveform comparison')
        time_s,traces,rows=paired_difference(*levels,mode)
        atomic_write(output/'comparison.npz',lambda f:np.savez_compressed(f,time_s=time_s,traces_m=traces))
        for group in levels:
            for level in group:
                if any(sha256_file(level['root']/name)!=digest for name,digest in level['input_identities'].items()):
                    raise ValueError('scattering raw inputs changed during comparison')
        if (sha256_file(protocol_path)!=protocol_hash
                or any(sha256_file(PROJECT/name)!=digest for name,digest in sources.items())
                or any(sha256_file(parent['path'])!=parent['sha256'] for parent in parents)
                or any(sha256_file(check['path'])!=check['sha256'] for parent in parent_reports for check in parent['cpu_checks'])):
            raise ValueError('scattering protocol, analysis or total-waveform proof changed')
        report.update(completed=True,passed=all(p['passed'] for p in parents) and all(r['passed'] for r in rows),
            receivers=rows,total_waveform_checks=parents,
            comparison_sha256=sha256_file(output/'comparison.npz'),
            paired_levels=[dict(intact_directory=str(a['root']),recessed_directory=str(b['root']),
                intact_plan_fingerprint=a['plan']['fingerprint'],recessed_plan_fingerprint=b['plan']['fingerprint'],
                intact_inputs=a['input_identities'],recessed_inputs=b['input_identities']) for a,b in zip(*levels)])
    except Exception as error:
        report['error']=f'{type(error).__name__}: {error}';raise
    finally:
        report['resources']=guard.report()
        write_json(output/(mode+'-scattering-report.json'),sealed(report))
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('spatial','time'))
    parser.add_argument('--intact-runs',nargs=3,type=Path,required=True)
    parser.add_argument('--intact-cpu-checks',nargs=3,type=Path,required=True)
    parser.add_argument('--recessed-runs',nargs=3,type=Path,required=True)
    parser.add_argument('--recessed-cpu-checks',nargs=3,type=Path,required=True)
    parser.add_argument('--protocol',type=Path,default=PROJECT/'configs/embedded-scattering-protocol.json')
    parser.add_argument('--source-check',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    guard=ResourceGuard(ResourceLimits(max_seconds=300,max_rss_mib=1536,
        max_system_memory_percent=89.9,min_available_mib=768))
    report=assess_scattering(args.intact_runs,args.intact_cpu_checks,args.recessed_runs,args.recessed_cpu_checks,
        args.mode,args.protocol,args.output,guard,args.source_check)
    print(json.dumps({k:report[k] for k in ('completed','passed','receivers','fingerprint')}))
    return 0 if report['passed'] else 2


if __name__=='__main__':raise SystemExit(main())
