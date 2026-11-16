"""Validate retained source snapshots before comparing historical wave runs."""
import hashlib
import json
import math
from pathlib import Path
import zipfile

from examples.embedded_wave import FULL_LEVELS, analysis_paths, checked_case, screen
from examples.tensor_wave import PROJECT, checked_receipt, identities, read_json
from traxis.gpu_verification import gpu_package_identity
from traxis.highorder_verification import distribution_identity
from traxis.storage import sha256_file


COMPATIBILITY = PROJECT/'configs/embedded-wave-source-review.json'


def check_snapshot(plan, path):
    """Check every archived byte against its original plan without extraction."""
    path=Path(path)
    if path.stat().st_size>16*1024**2 or sha256_file(path)!=plan['source_snapshot_sha256']:
        raise ValueError('embedded source snapshot size or hash changed')
    expected={folder+'/'+name:digest for key,folder in
        (('source_sha256','traxis'),('driver_sha256','examples')) for name,digest in plan[key].items()}
    if set(expected)&set(plan['analysis_sources']):
        raise ValueError('embedded source manifests overlap')
    expected.update(plan['analysis_sources'])
    with zipfile.ZipFile(path) as archive:
        entries=archive.infolist()
        if (len(entries)!=len(expected)+1 or {p.filename for p in entries}!=set(expected)|{'case.json'}
                or any(p.file_size>4*1024**2 or p.is_dir() for p in entries)
                or sum(p.file_size for p in entries)>16*1024**2):
            raise ValueError('embedded source snapshot members or expansion bounds changed')
        for name,digest in expected.items():
            if hashlib.sha256(archive.read(name)).hexdigest()!=digest:
                raise ValueError('embedded archived source differs from its recorded hash: '+name)
        if json.loads(archive.read('case.json'))!=plan['case']:
            raise ValueError('embedded archived physical case changed')


def check_source_review(plan, current, current_analysis, review):
    """Accept only exact current sources or explicitly reviewed hash transitions."""
    if review.get('kind')!='embedded_wave_source_review_v1':
        raise ValueError('unknown embedded source review')
    changed=[]
    groups=[('source_sha256','traxis/'),('driver_sha256','examples/'),('analysis_sources','')]
    for key,prefix in groups:
        actual=current_analysis if key=='analysis_sources' else current[key]
        if set(plan[key])!=set(actual):
            raise ValueError('embedded source manifest names changed')
        for name,old in plan[key].items():
            new=actual[name]
            if old==new:continue
            item=review['files'].get(prefix+name,{})
            if new!=item.get('current_sha256') or old not in item.get('historical_sha256',[]):
                raise ValueError('unreviewed embedded source change: '+prefix+name)
            changed.append(dict(path=prefix+name,run_sha256=old,current_sha256=new,review=item['reason']))
    return changed


def validate_saved_plan(plan, root, guard=None):
    """Retain original identities; never rewrite a completed run to current hashes."""
    checked_receipt(plan); checked_case(plan['case'])
    if (plan['kind']!='embedded_mechanical_window' or plan['mesh_screen']!=screen(plan['case'])
            or plan['declared_full_levels']!=[list(p) for p in FULL_LEVELS]
            or plan['runtime']!=distribution_identity(guard) or plan['gpu_packages']!=gpu_package_identity(guard)):
        raise ValueError('embedded saved case, mesh or runtime changed')
    analysis=analysis_paths(plan['case'],'execution' in plan)
    current_analysis={name:sha256_file(PROJECT/name) for name in analysis}
    changed=check_source_review(plan,identities(),current_analysis,read_json(COMPATIBILITY))
    check_snapshot(plan,Path(root)/'source-snapshot.zip')
    return dict(source_snapshot_sha256=plan['source_snapshot_sha256'],reviewed_changes=changed,
        review_sha256=sha256_file(COMPATIBILITY),validator_sha256=sha256_file(__file__))


def _check_integrator_receipt(report, changes):
    expected={label+'_'+name for label in ('dense','sparse') for name in ('time_s','displacement_m','energy_time_s',
        'physical_energy_j','modified_energy_j','final_displacement_m','final_velocity_m_s')}
    if (not report['completed'] or not report['passed'] or report['ready_for_training_data'] or report['training_updates']!=0
            or report['checker_sha256']!=sha256_file(PROJECT/'examples/check_embedded_source_transition.py')
            or report['source_validation']['review_sha256']!=sha256_file(COMPATIBILITY)
            or report['source_validation']['validator_sha256']!=sha256_file(__file__)
            or [case['nonzero_initial_state'] for case in report['cases']]!=[False,True]
            or any(row['run_sha256']!=report['archived_integrator_sha256']
                or row['current_sha256']!=report['current_integrator_sha256'] for row in changes)
            or any(not case['passed'] or set(case['maximum_absolute_errors'])!=expected
                or any(not math.isfinite(error) or not 0<=error<1e-12 for error in case['maximum_absolute_errors'].values())
                for case in report['cases'])):
        raise ValueError('embedded CUDA source-transition evidence changed or failed')


def check_integrator_transition(source_checks, path):
    changes=[row for check in source_checks for row in check['reviewed_changes']
        if row['path']=='traxis/gpu_enriched_verlet.py']
    if not changes:return None
    if path is None:
        raise ValueError('historical integrator comparison requires the passing CUDA source-transition check')
    path=Path(path).resolve();report=checked_receipt(read_json(path))
    if report.get('kind')=='embedded_cuda_source_transitions':
        versions=report.get('versions',[])
        required={row['run_sha256'] for row in changes}
        if (not report.get('completed') or not report.get('passed') or report.get('ready_for_training_data') is not False
                or report.get('training_updates')!=0 or not 1<=len(versions)<=4
                or report.get('checker_sha256')!=sha256_file(PROJECT/'examples/check_embedded_source_transitions.py')
                or report.get('current_integrator_sha256')!=sha256_file(PROJECT/'traxis/gpu_enriched_verlet.py')
                or len({v['archived_integrator_sha256'] for v in versions})!=len(versions)
                or not required.issubset({v['archived_integrator_sha256'] for v in versions})):
            raise ValueError('embedded CUDA transition bundle is missing an archived version or repeats one')
        for version in versions:
            child=(path.parent/version['path']).resolve()
            if not child.is_relative_to(path.parent) or child==path or sha256_file(child)!=version['sha256']:
                raise ValueError('embedded CUDA version proof escaped its bundle or changed')
            proof=checked_receipt(read_json(child))
            if proof['fingerprint']!=version['fingerprint'] or proof['archived_integrator_sha256']!=version['archived_integrator_sha256']:
                raise ValueError('embedded CUDA version binding changed')
            subset=[row for row in changes if row['run_sha256']==version['archived_integrator_sha256']]
            if not subset:
                subset=[dict(run_sha256=version['archived_integrator_sha256'],
                    current_sha256=report['current_integrator_sha256'])]
            _check_integrator_receipt(proof,subset)
    else:
        _check_integrator_receipt(report,changes)
    return dict(path=str(path),sha256=sha256_file(path),fingerprint=report['fingerprint'])
