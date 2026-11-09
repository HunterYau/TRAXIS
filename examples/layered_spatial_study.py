"""Freeze a common time grid and assess complete layered thickness waveforms."""

import argparse
import json
import math
from pathlib import Path
import zipfile

import numpy as np

from examples.enriched_gpu_wave import identity as wave_identity,mesh_context
from examples.enriched_time_refinement import read_level
from examples.layered_mesh import bundle_context
from examples.matrixfree_step import _json
from traxis.elmer_plate_refinement import reference_source_hashes
from traxis.gpu_verification import gpu_package_identity
from traxis.highorder_verification import distribution_identity
from traxis.resources import ResourceGuard,ResourceLimits
from traxis.storage import atomic_write,canonical_hash,sha256_file,write_json


def identity():
    names = ('layered_spatial_study.py','verify_enriched_window.py','enriched_time_refinement.py','pace_gpu_worker.py')
    return {**wave_identity(),**{name:sha256_file(Path(__file__).parent/name) for name in names}}


def common_grid(duration,limits,minimum=65536,fraction=.4):
    if (not math.isfinite(duration) or duration<=0 or type(minimum) is not int or not 1<=minimum<=500000
            or not 0<fraction<=.4 or len(limits)!=3 or any(not math.isfinite(v) or v<=0 for v in limits)):
        raise ValueError('invalid common-grid duration, stability limits or step bound')
    requested = max(minimum,math.ceil(duration/(fraction*min(limits))))
    steps = 1<<(requested-1).bit_length()
    if steps>500000:
        raise MemoryError('complete common window exceeds the explicit step-count limit')
    dt = duration/steps
    fractions = [float(np.nextafter(dt/limit,np.inf)) for limit in limits]
    if any(math.ceil(duration/(c*limit))!=steps or c>fraction+1e-14 for c,limit in zip(fractions,limits)):
        raise ArithmeticError('common physical sample grid could not be represented safely')
    return steps,dt,fractions


def checked_configuration(value):
    fields = {'kind','wave_settings','minimum_common_steps','maximum_cfl_fraction',
        'maximum_finest_relative_change','require_decreasing_differences','levels'}
    if (set(value)!=fields or value['kind']!='enriched_layered_thickness_study'
            or value['minimum_common_steps']!=65536 or value['maximum_cfl_fraction']!=.4
            or value['maximum_finest_relative_change']!=.05 or value['require_decreasing_differences'] is not True
            or len(value['levels'])!=3 or [r['layers'] for r in value['levels']]!=[1,2,4]
            or any(set(r)!={'layers','mesh','operator','wave','cpu','max_seconds','max_rss_mib'} for r in value['levels'])
            or len({r['wave'] for r in value['levels']})!=3
            or any(not 0<r['max_seconds']<=7200 or not 0<r['max_rss_mib']<=2048 for r in value['levels'])):
        raise ValueError('layered thickness study needs the fixed three-level protocol and criteria')
    return value


def prepare(configuration,output,guard):
    configuration,output = Path(configuration).resolve(),Path(output).resolve()
    cfg = checked_configuration(json.loads(configuration.read_text()))
    wave = json.loads(Path(cfg['wave_settings']).read_text())
    if wave['duration_s']!=.0005:
        raise ValueError('layered study must retain the complete 500-microsecond window')
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('layered study planning needs a fresh output directory')
    operators,contexts,operator_plans = [],[],[]
    for row in cfg['levels']:
        guard.check()
        root = Path(row['operator']).resolve()
        plan,parent,native = (_json(root/p) for p in ('enriched-gpu-plan.json','monitor-result.json','native/wave-report.json'))
        if (parent['completed'] is not True or parent['passed'] is not True or parent['worker_run']['exit_code']!=0
                or parent['native_report_sha256']!=sha256_file(root/'native/wave-report.json')
                or native['completed'] is not True or native['passed'] is not True or native['operator_check_passed'] is not True
                or native['steps_completed']!=0 or native['full_window_completed'] is not False
                or native['plan_fingerprint']!=plan['fingerprint'] or plan['mode']!='operator'
                or plan['settings']!=wave or plan['source_sha256']!=reference_source_hashes()
                or plan['driver_sha256']!=wave_identity() or Path(plan['mesh_bundle_directory']).resolve()!=Path(row['mesh']).resolve()):
            raise ValueError('layered operator preflight is incomplete, changed or belongs to another case')
        _,context,layered = mesh_context(plan)
        spec = context[0]['specification']
        if not layered or spec['cells']!=[100,100,row['layers']]:
            raise ValueError('thickness study changed an in-plane cell count')
        if spec['plate']!={'width_m':.5,'height_m':.5,'thickness_m':.003}:
            raise ValueError('layered research control must retain the full 500 mm plate')
        if contexts and {k:v for k,v in spec.items() if k!='cells'}!={k:v for k,v in contexts[0][0]['specification'].items() if k!='cells'}:
            raise ValueError('thickness study changed plate, material or capacity settings')
        path = root/'native/operator-comparison.npz'
        if sha256_file(path)!=native['operator_comparison_sha256']:
            raise ValueError('layered operator comparison archive changed')
        with np.load(path,allow_pickle=False) as arrays:
            error = float(np.linalg.norm(arrays['gpu_action']-arrays['reference_action'])/np.linalg.norm(arrays['reference_action']))
        if error>1e-10 or not np.isfinite(error):
            raise ArithmeticError('layered GPU operator fails independent CPU agreement')
        operators.append(dict(directory=str(root),stable_dt_s=native['stable_dt_s'],operator_relative_l2=error,
            identities={name:sha256_file(root/name) for name in ('enriched-gpu-plan.json','monitor-result.json','native/wave-report.json','native/operator-comparison.npz')}))
        contexts.append(context)
        operator_plans.append(plan)
    if (any(any(p[k]!=operator_plans[0][k] for k in ('runtime','gpu_packages')) for p in operator_plans[1:])
            or operator_plans[0]['runtime']!=distribution_identity(guard)
            or operator_plans[0]['gpu_packages']!=gpu_package_identity(guard)):
        raise ValueError('layered operator runtime changed before the common grid was frozen')
    steps,dt,fractions = common_grid(wave['duration_s'],[r['stable_dt_s'] for r in operators],cfg['minimum_common_steps'],cfg['maximum_cfl_fraction'])
    output.mkdir(parents=True,exist_ok=True)
    plan = dict(kind='enriched_layered_thickness_plan',configuration=cfg,configuration_sha256=sha256_file(configuration),
        source_sha256=reference_source_hashes(),driver_sha256=identity(),steps=steps,dt_s=dt,operators=operators,levels=[],
        runtime=operator_plans[0]['runtime'],gpu_packages=operator_plans[0]['gpu_packages'])
    for row,cfl in zip(cfg['levels'],fractions):
        settings = {**wave,'cfl_fraction':cfl}
        file = output/f'settings-z{row["layers"]}.json'
        write_json(file,settings)
        plan['levels'].append({**row,'settings':settings,'settings_file':str(file),'settings_sha256':sha256_file(file)})
    def snapshot(stream):
        with zipfile.ZipFile(stream,'w',compression=zipfile.ZIP_DEFLATED) as archive:
            for name in plan['source_sha256']:
                archive.writestr('traxis/'+name,(Path(__file__).resolve().parents[1]/'traxis'/name).read_bytes())
            for name in plan['driver_sha256']:
                archive.writestr('examples/'+name,(Path(__file__).parent/name).read_bytes())
    atomic_write(output/'source-snapshot.zip',snapshot)
    plan['source_snapshot_sha256'] = sha256_file(output/'source-snapshot.zip')
    plan['fingerprint'] = canonical_hash(plan)
    write_json(output/'study-plan.json',plan)
    return plan


def spatial_differences(time,traces,steps,duration=.0005):
    time = np.asarray(time)
    traces = [np.asarray(a) for a in traces]
    if np.iscomplexobj(time) or time.shape!=(steps+1,) or time[0]!=0. or time[-1]!=duration or len(traces)!=3:
        raise ValueError('spatial comparison needs three complete common-time traces')
    if not np.allclose(time,np.arange(steps+1)*(duration/steps),rtol=0,atol=16*np.finfo(float).eps*duration):
        raise ValueError('spatial comparison time grid changed')
    if any(a.shape!=traces[0].shape or a.ndim!=2 or a.shape[0]!=steps+1 or not 1<=a.shape[1]<=32
           or np.iscomplexobj(a) or not np.isfinite(a).all() for a in traces):
        raise ValueError('spatial receiver traces changed dimensions or contain nonfinite data')
    first,second = (np.linalg.norm(traces[i]-traces[i+1],axis=0) for i in (0,1))
    norm = np.linalg.norm(traces[-1],axis=0)
    if not all(np.isfinite(a).all() for a in (first,second,norm)):
        raise ValueError('spatial waveform norms overflowed')
    rows = []
    for i in range(len(norm)):
        resolved = bool(norm[i]>0 and np.isfinite(norm[i]) and min(first[i],second[i])>1e-11*norm[i])
        relative = float(second[i]/norm[i]) if norm[i]>0 else None
        decreasing = bool(resolved and second[i]<first[i])
        rows.append(dict(receiver=i,coarse_difference_l2_m=float(first[i]),fine_difference_l2_m=float(second[i]),
            finest_relative_change=relative,decreasing=decreasing,
            passed=bool(decreasing and relative<.05)))
    return rows


def assess(plan_path,output,guard):
    plan_path,output = Path(plan_path).resolve(),Path(output).resolve()
    plan = _json(plan_path)
    if plan['kind']!='enriched_layered_thickness_plan' or plan['source_sha256']!=reference_source_hashes() or plan['driver_sha256']!=identity():
        raise ValueError('layered study source/plan changed')
    checked_configuration(plan['configuration'])
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('layered assessment needs a fresh output directory')
    output.mkdir(parents=True,exist_ok=True)
    report = dict(completed=False,passed=False,ready_for_training_data=False,training_updates=0,plan_fingerprint=plan['fingerprint'])
    try:
        levels = []
        for row in plan['levels']:
            guard.check()
            if sha256_file(row['settings_file'])!=row['settings_sha256']:
                raise ValueError('frozen layered excitation settings changed')
            level = read_level(dict(directory=row['wave'],steps=plan['steps']),.0005,guard)
            if level['plan']['settings']!=row['settings'] or level['plan']['driver_sha256']!=wave_identity() or level['plan']['source_sha256']!=plan['source_sha256']:
                raise ValueError('layered waveform changed its frozen code or physical settings')
            if levels and any(level['plan'][k]!=levels[0]['plan'][k] for k in ('runtime','gpu_packages')):
                raise ValueError('layered waveform runtime changed')
            if any(level['plan'][k]!=plan[k] for k in ('runtime','gpu_packages')):
                raise ValueError('layered waveform runtime differs from its frozen operator preflight')
            root,context,layered = mesh_context(level['plan'])
            if not layered or root!=Path(row['mesh']).resolve() or context[0]['specification']['cells']!=[100,100,row['layers']]:
                raise ValueError('layered waveform geometry changed')
            if level['native']['dt_s']!=plan['dt_s']:
                raise ValueError('layered waveform increment changed')
            if levels and not np.array_equal(level['time'],levels[0]['time']):
                raise ValueError('spatial waveforms do not share exact sample times')
            cpu = _json(Path(row['cpu'])/'cpu-window-check.json')
            if (cpu['completed'] is not True or cpu['passed'] is not True or cpu['forward_result_fingerprint']!=level['parent']['fingerprint']
                    or cpu['checker_sha256']!=plan['driver_sha256']['verify_enriched_window.py']
                    or cpu['input_identities']!=level['identities']):
                raise ValueError('layered waveform lacks a matching independent CPU check')
            levels.append(level)
        rows = spatial_differences(levels[0]['time'],[r['trace'] for r in levels],plan['steps'])
        atomic_write(output/'comparison.npz',lambda stream:np.savez_compressed(stream,time_s=levels[0]['time'],
            z1_m=levels[0]['trace'],z2_m=levels[1]['trace'],z4_m=levels[2]['trace']))
        for level in levels:
            if any(sha256_file(level['root']/n)!=h for n,h in level['identities'].items()):
                raise ValueError('layered waveform inputs changed during assessment')
        report.update(completed=True,passed=all(r['passed'] for r in rows),receivers=rows,
            comparison_sha256=sha256_file(output/'comparison.npz'),levels=[dict(directory=str(r['root']),identities=r['identities']) for r in levels],
            scope='Thickness discrepancy screen on fixed 5 mm lateral cells and common time grid; not a continuum, in-plane, defect or PZT acceptance.')
    except Exception as error:
        report['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        report['resources'] = guard.report()
        report['fingerprint'] = canonical_hash(report)
        write_json(output/'thickness-report.json',report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('prepare','assess'))
    parser.add_argument('--configuration',type=Path,default=Path('configs/layered-thickness-study.json'))
    parser.add_argument('--plan',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    if args.action=='assess' and not args.plan:
        parser.error('assessment requires the frozen study plan')
    guard = ResourceGuard(ResourceLimits(max_seconds=120,max_rss_mib=1536,max_system_memory_percent=85,min_available_mib=2048))
    result = prepare(args.configuration,args.output,guard) if args.action=='prepare' else assess(args.plan,args.output,guard)
    print(json.dumps({k:result[k] for k in ('kind','steps','dt_s','levels')} if args.action=='prepare'
        else {k:result[k] for k in ('completed','passed','ready_for_training_data','receivers')}))
    return 0 if args.action=='prepare' or result['passed'] else 2


if __name__=='__main__':
    raise SystemExit(main())
