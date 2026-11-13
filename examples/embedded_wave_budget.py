"""Run the frozen embedded worker with an explicit longer wall-time profile."""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
import zipfile

from examples.embedded_wave import FULL_LEVELS,case_config,checked_case,screen,validate_plan,worker
from examples.tensor_wave import PROJECT,checked_receipt,identities,read_json,sealed,snapshot_sources
from traxis.elmer_reference import run_monitored
from traxis.gpu_verification import gpu_package_identity
from traxis.highorder_verification import distribution_identity
from traxis.resources import ResourceGuard,ResourceLimits
from traxis.storage import sha256_file,write_json


def execution_profile(case):
    checked_case(case)
    if case['scope']!='full': raise ValueError('extended execution is restricted to declared full-size cases')
    limits=ResourceLimits(max_seconds=3600,max_rss_mib=2048,max_system_memory_percent=85,min_available_mib=2048)
    return dict(kind='embedded_extended_wall_budget',limits=asdict(limits),
        reason='Measured coarse-level timing projects a fine-level runtime beyond 2400 seconds; numerical controls and memory/thermal limits are unchanged.')


class ExtendedResourceGuard(ResourceGuard):
    """Retain late observations and extrema before the base history fills."""
    def __init__(self,limits):
        super().__init__(limits);self.compactions=0;self.total_samples=0

    def compact_history(self):
        history=self.history
        if len(history)<8000:return
        selected=set(range(0,len(history),2))|{0,len(history)-1}
        for key,maximum in [('process_rss_mib',True),('system_memory_percent',True),('available_mib',False)]:
            choose=max if maximum else min
            selected.add(choose(range(len(history)),key=lambda i:history[i][key]))
        gpus=[(i,g) for i,s in enumerate(history) for g in s['gpus']]
        if gpus:
            for metric in (lambda g:g['used_mib'],lambda g:g['temperature_c'],lambda g:g['used_mib']/g['total_mib']):
                selected.add(max(gpus,key=lambda item:metric(item[1]))[0])
        cpu=[(i,max(s['cpu_temperatures_c'])) for i,s in enumerate(history) if s['cpu_temperatures_c']]
        if cpu:selected.add(max(cpu,key=lambda item:item[1])[0])
        self.history=[history[i] for i in sorted(selected)];self.compactions+=1

    def check(self,force=False):
        self.compact_history();previous=self.last_check
        try:super().check(force)
        finally:
            if self.last_check!=previous:self.total_samples+=1

    def report(self):
        return {**super().report(),'history_compactions':self.compactions,'total_samples_observed':self.total_samples,
            'sample_retention':'Before 8000 entries, retain every second older sample plus first/latest and every resource extremum; checks and enforcement continue unchanged.'}


def check_execution(plan,guard):
    expected=execution_profile(plan['case'])
    if (plan.get('execution')!=expected or asdict(guard.limits)!=expected['limits']
            or plan['analysis_sources'].get('examples/embedded_wave_budget.py')!=sha256_file(__file__)):
        raise ValueError('embedded execution budget, actual guard or launcher source changed')


def worker_entry(plan_path,output,guard):
    plan=checked_receipt(read_json(plan_path));check_execution(plan,guard);validate_plan(plan,guard)
    report=worker(plan_path,output,guard)
    check_execution(plan,guard)
    if report['resources']['limits']!=plan['execution']['limits']:
        raise ValueError('embedded worker did not use its declared resource limits')
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--order',type=int,default=4);parser.add_argument('--steps',type=int,default=294912)
    parser.add_argument('--recess',action='store_true');parser.add_argument('--worker',action='store_true')
    parser.add_argument('--plan',type=Path);args=parser.parse_args();output=args.output.resolve()
    case=checked_case(read_json(args.plan)['case']) if args.worker else case_config('full',args.order,args.steps,args.recess)
    profile=execution_profile(case);guard=ExtendedResourceGuard(ResourceLimits(**profile['limits']))
    if args.worker: return 0 if worker_entry(args.plan,output,guard)['passed'] else 2
    if not output.is_relative_to(PROJECT/'outputs'): raise ValueError('embedded evidence must remain in project outputs')
    output.mkdir(parents=True,exist_ok=False)
    analysis={name:sha256_file(PROJECT/name) for name in ('examples/embedded_wave.py',
        'examples/verify_embedded_window.py','examples/embedded_wave_budget.py')}
    plan=dict(kind='embedded_mechanical_window',case=case,execution=profile,declared_full_levels=[list(p) for p in FULL_LEVELS],
        mesh_screen=screen(case),**identities(),analysis_sources=analysis,runtime=distribution_identity(guard),
        gpu_packages=gpu_package_identity(guard))
    snapshot_sources(output/'source-snapshot.zip',plan)
    with zipfile.ZipFile(output/'source-snapshot.zip','a',compression=zipfile.ZIP_DEFLATED) as archive:
        for name in analysis: archive.writestr(name,(PROJECT/name).read_bytes())
    plan['source_snapshot_sha256']=sha256_file(output/'source-snapshot.zip');write_json(output/'embedded-wave-plan.json',sealed(plan))
    temporary=output/'temporary';temporary.mkdir()
    env={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','CUPY_CACHE_DIR':str(output/'kernel-cache'),
        'CUPY_CACHE_SAVE_CUDA_SOURCE':'1',**{name:str(temporary) for name in ('TEMP','TMP','TMPDIR')}}
    result=dict(completed=False,passed=False,ready_for_training_data=False,training_updates=0,
        plan_fingerprint=plan['fingerprint'],execution=profile)
    try:
        check_execution(plan,guard);validate_plan(plan,guard)
        command=[sys.executable,'-m','examples.embedded_wave_budget','--worker','--plan',str(output/'embedded-wave-plan.json'),'--output',str(output/'native')]
        result['worker_run']=run_monitored(command,PROJECT,env,guard,output/'worker.log')
        native=checked_receipt(read_json(output/'native/wave-report.json'))
        if (result['worker_run']['exit_code']!=0 or not native['completed'] or not native['passed']
                or native['plan_fingerprint']!=plan['fingerprint'] or native['resources']['limits']!=profile['limits']):
            raise ValueError('extended embedded worker failed or changed its declared limits')
        for name,key in [('mechanical-window.npz','mechanical_window_sha256'),('mechanical-patches.npz','mechanical_patches_sha256')]:
            if sha256_file(output/'native'/name)!=native[key]: raise ValueError('extended embedded proof changed')
        check_execution(plan,guard);validate_plan(plan,guard)
        if sha256_file(output/'source-snapshot.zip')!=plan['source_snapshot_sha256']: raise ValueError('extended embedded snapshot changed')
        result.update(completed=True,passed=True,report_sha256=sha256_file(output/'native/wave-report.json'))
    except Exception as error: result['error']=f'{type(error).__name__}: {error}'
    finally:
        result['resources']=guard.report();write_json(output/'monitor-result.json',sealed(result))
    print(json.dumps({k:v for k,v in result.items() if k!='resources'}));return 0 if result['passed'] else 2


if __name__=='__main__':raise SystemExit(main())
