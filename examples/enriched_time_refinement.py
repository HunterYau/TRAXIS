"""Compare source-bound complete waveforms at exactly matched time samples."""

import argparse
import json
import math
from pathlib import Path
import zipfile

import numpy as np

from examples.enriched_gpu_wave import identity as wave_driver_identity
from examples.matrixfree_step import _json
from traxis.elmer_plate_refinement import reference_source_hashes
from traxis.gpu_verification import gpu_package_identity
from traxis.highorder_verification import distribution_identity
from traxis.resources import ResourceGuard,ResourceLimits
from traxis.storage import atomic_write,canonical_hash,sha256_file,write_json


def checked_plan(values):
    fields = {'kind','duration_s','reference_result_fingerprint','levels','ratio_bounds',
        'maximum_finest_relative_change','relative_identifiability_floor'}
    if (set(values)!=fields or values['kind']!='enriched_full_window_time_refinement'
            or values['duration_s']!=.0005 or values['ratio_bounds']!=[3.8,4.2]
            or values['maximum_finest_relative_change']!=.01 or values['relative_identifiability_floor']!=1e-11
            or len(values['levels'])!=3 or [r['factor'] for r in values['levels']]!=[1,2,4]
            or any(set(r)!={'factor','steps','directory'} for r in values['levels'])
            or any(type(r['steps']) is not int or not 1<=r['steps']<=500000 for r in values['levels'])
            or any(r['steps']!=values['levels'][0]['steps']*r['factor'] for r in values['levels'])
            or len({r['directory'] for r in values['levels']})!=3):
        raise ValueError('time refinement needs the fixed full-window criteria and three distinct doubled grids')
    return values


def compare_traces(times,traces,criteria):
    """Use raw per-receiver L2 differences on the complete coarse time grid."""
    checked_plan(criteria)
    if len(times)!=3 or len(traces)!=3:
        raise ValueError('time comparison needs exactly three levels')
    duration = criteria['duration_s']
    tolerance = 16*np.finfo(float).eps*duration
    aligned = []
    receivers = None
    for i,(time,values) in enumerate(zip(times,traces)):
        time,values = np.asarray(time),np.asarray(values)
        steps = criteria['levels'][i]['steps']
        if (np.iscomplexobj(time) or time.shape!=(steps+1,) or not np.isfinite(time).all()
                or time[0]!=0. or time[-1]!=duration
                or not np.allclose(time,np.arange(steps+1)*(duration/steps),rtol=0,atol=tolerance)
                or np.iscomplexobj(values) or values.ndim!=2 or values.shape[0]!=len(time)
                or not 1<=values.shape[1]<=32 or not np.isfinite(values).all()):
            raise ValueError('time comparison needs complete finite uniform physical-time records')
        if receivers is None:
            receivers = values.shape[1]
        if values.shape[1]!=receivers:
            raise ValueError('receiver count changed between refinements')
        factor = criteria['levels'][i]['factor']
        if not np.allclose(time[::factor],times[0],rtol=0,atol=tolerance):
            raise ValueError('time levels do not share exact coarse samples')
        aligned.append(values[::factor])
    first,second = (np.linalg.norm(aligned[i]-aligned[i+1],axis=0) for i in (0,1))
    norms = np.linalg.norm(aligned[2],axis=0)
    if not all(np.isfinite(a).all() for a in (first,second,norms)):
        raise ValueError('time-study waveform norms overflowed')
    rows = []
    for i in range(receivers):
        identifiable = bool(norms[i]>0 and min(first[i],second[i])>criteria['relative_identifiability_floor']*norms[i])
        ratio = float(first[i]/second[i]) if identifiable else None
        change = float(second[i]/norms[i]) if norms[i]>0 else None
        passed = bool(identifiable and criteria['ratio_bounds'][0]<=ratio<=criteria['ratio_bounds'][1]
            and change<criteria['maximum_finest_relative_change'])
        rows.append(dict(receiver=i,coarse_difference_l2_m=float(first[i]),fine_difference_l2_m=float(second[i]),
            finest_waveform_l2_m=float(norms[i]),difference_ratio=ratio,observed_order=math.log2(ratio) if ratio else None,
            finest_relative_change=change,rate_identifiable=identifiable,passed=passed,
            second_order_richardson_relative_estimate=change/3 if passed else None))
    return rows,np.asarray(aligned)


def same_problem(plans):
    """Only CFL may change; mesh, physics, code, runtime and device settings stay fixed."""
    base = plans[0]
    for plan in plans[1:]:
        if any(plan[k]!=base[k] for k in ('kind','mode','input_sha256','source_sha256','driver_sha256','runtime','gpu_packages')):
            raise ValueError('time study changed geometry, numerical implementation or runtime')
        common = lambda p:{k:v for k,v in p['settings'].items() if k!='cfl_fraction'}
        if common(plan)!=common(base):
            raise ValueError('time study changed source, receivers or other physical settings')


def checked_archive(path,expected_hash):
    if path.stat().st_size>256*1024**2 or sha256_file(path)!=expected_hash:
        raise ValueError('time-study archive changed or exceeds its storage bound')
    with zipfile.ZipFile(path) as archive:
        if sum(r.file_size for r in archive.infolist())>256*1024**2:
            raise MemoryError('time-study archive exceeds its expansion bound')


def read_level(level,duration,guard):
    root = Path(level['directory']).resolve()
    plan,parent,native = (_json(root/p) for p in ('enriched-gpu-plan.json','monitor-result.json','native/wave-report.json'))
    if (parent['completed'] is not True or parent['passed'] is not True or parent['worker_run']['exit_code']!=0
            or parent['native_report_sha256']!=sha256_file(root/'native/wave-report.json')
            or native['plan_fingerprint']!=plan['fingerprint'] or parent['plan_fingerprint']!=plan['fingerprint']
            or native['completed'] is not True or native['passed'] is not True or native['full_window_completed'] is not True
            or native['steps_completed']!=level['steps'] or native['planned_steps']!=level['steps']
            or native['requested_duration_s']!=duration or plan['settings']['duration_s']!=duration
            or plan['mode']!='wave' or parent['training_updates']!=0 or native['training_updates']!=0
            or native['ready_for_training_data'] is not False):
        raise ValueError('time-study level is incomplete, incompatible or changed')
    artifacts = {'operator-comparison.npz':'operator_comparison_sha256',
        'mechanical-patches.npz':'mechanical_patches_sha256','mechanical-window.npz':'mechanical_window_sha256'}
    for name,key in artifacts.items():
        guard.check()
        checked_archive(root/'native'/name,native[key])
    with np.load(root/'native/mechanical-window.npz',allow_pickle=False) as arrays:
        time,trace = arrays['time_s'],arrays['displacement_m']
        et,modified = arrays['energy_time_s'],arrays['modified_energy_j']
    if (et.ndim!=1 or modified.shape!=et.shape or len(et)<2 or et[0]!=0. or et[-1]!=duration
            or not np.isfinite(et).all() or not np.isfinite(modified).all()
            or native['dt_s']!=duration/level['steps']):
        raise ValueError('time-study energy times or declared increment changed')
    after = modified[et>plan['settings']['cycles']/plan['settings']['frequency_hz']+2*native['dt_s']]
    if not np.isfinite(after).all() or len(after)<2 or after.mean()<=0 or np.ptp(after)/abs(after.mean())>plan['settings']['energy_tolerance']:
        raise ValueError('time-study level fails its raw postburst energy check')
    with np.load(root/'native/operator-comparison.npz',allow_pickle=False) as arrays:
        coefficients = {k:arrays[k] for k in ('native_to_topology','points_m','nodal_mass_kg')}
    with np.load(root/'native/mechanical-patches.npz',allow_pickle=False) as arrays:
        coefficients.update({k:arrays[k] for k in ('load','probes')})
    return dict(root=root,plan=plan,parent=parent,native=native,time=time,trace=trace,coefficients=coefficients,
        identities={str(p.relative_to(root)):sha256_file(p) for p in
            [root/'enriched-gpu-plan.json',root/'monitor-result.json',root/'native/wave-report.json',
             *[root/'native'/name for name in artifacts]]})


def assess(plan_path,output,guard):
    plan_path,output = Path(plan_path).resolve(),Path(output).resolve()
    criteria = checked_plan(json.loads(plan_path.read_text()))
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('time refinement assessment needs a fresh output directory')
    output.mkdir(parents=True,exist_ok=True)
    report = dict(completed=False,passed=False,ready_for_training_data=False,training_updates=0,
        criteria=criteria,criteria_sha256=sha256_file(plan_path),driver_sha256=sha256_file(Path(__file__)))
    try:
        levels = [read_level(level,criteria['duration_s'],guard) for level in criteria['levels']]
        if levels[0]['parent']['fingerprint']!=criteria['reference_result_fingerprint']:
            raise ValueError('time-study coarse reference changed')
        same_problem([r['plan'] for r in levels])
        base = levels[0]['plan']
        if (base['source_sha256']!=reference_source_hashes() or base['driver_sha256']!=wave_driver_identity()
                or base['runtime']!=distribution_identity(guard) or base['gpu_packages']!=gpu_package_identity(guard)):
            raise ValueError('time-study numerical source/runtime no longer matches the retained levels')
        if any(sha256_file(Path(base['capacity_directory'])/name)!=digest for name,digest in base['input_sha256'].items()):
            raise ValueError('time-study retained geometry inputs changed')
        for level in levels[1:]:
            if any(level['native'][key]!=levels[0]['native'][key] for key in ('vector_dofs','stable_dt_s')):
                raise ValueError('time study changed coefficient count or mesh-specific stability bound')
            if any(not np.array_equal(value,level['coefficients'][key]) for key,value in levels[0]['coefficients'].items()):
                raise ValueError('time study changed coefficient maps, mass or physical footprints')
            if any(level['native']['device'][key]!=levels[0]['native']['device'][key] for key in
                    ('name','compute_capability','total_bytes','cupy_version','precision')):
                raise ValueError('time study changed CUDA device or precision')
        rows,aligned = compare_traces([r['time'] for r in levels],[r['trace'] for r in levels],criteria)
        atomic_write(output/'comparison.npz',lambda stream:np.savez_compressed(stream,time_s=levels[0]['time'],
            coarse_m=aligned[0],half_m=aligned[1],quarter_m=aligned[2]))
        for level in levels:
            if any(sha256_file(level['root']/name)!=digest for name,digest in level['identities'].items()):
                raise ValueError('time-study inputs changed during assessment')
        report.update(completed=True,passed=all(r['passed'] for r in rows),receivers=rows,
            comparison_sha256=sha256_file(output/'comparison.npz'),source_sha256=base['source_sha256'],
            levels=[dict(directory=str(r['root']),identities=r['identities'],dt_s=r['native']['dt_s'],
                steps=r['native']['steps_completed'],propagation_seconds=r['native']['propagation_seconds'],
                resources=r['parent']['resources']) for r in levels],
            scope='Full-window temporal convergence on one fixed intact mesh only; no spatial, defect, PZT or physical acceptance.')
    except Exception as error:
        report['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        report['resources'] = guard.report()
        report['fingerprint'] = canonical_hash(report)
        write_json(output/'time-refinement-report.json',report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path,default=Path('configs/enriched-time-refinement.json'))
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    guard = ResourceGuard(ResourceLimits(max_seconds=90,max_rss_mib=1024,max_system_memory_percent=85,min_available_mib=2048))
    result = assess(args.plan,args.output,guard)
    print(json.dumps({k:result[k] for k in ('completed','passed','ready_for_training_data','receivers')}))
    return 0 if result['passed'] else 2


if __name__=='__main__':
    raise SystemExit(main())
