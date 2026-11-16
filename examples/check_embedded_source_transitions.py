"""Bind actual CUDA parity checks for every archived integrator version."""
import argparse
import json
from pathlib import Path

from examples.check_embedded_source_transition import verify as verify_version
from examples.tensor_wave import PROJECT, identities, sealed
from traxis.highorder_verification import distribution_identity
from traxis.resources import ResourceGuard, ResourceLimits
from traxis.storage import sha256_file, write_json


def verify(references, output, guard):
    output=Path(output).resolve()
    if not output.is_relative_to(PROJECT/'outputs') or not 1<=len(references)<=4:
        raise ValueError('CUDA transition bundle needs one to four references and a fresh project output')
    output.mkdir(parents=True,exist_ok=False)
    sources=identities();runtime=distribution_identity(guard)
    analysis={name:sha256_file(PROJECT/name) for name in ('examples/check_embedded_source_transitions.py',
        'examples/check_embedded_source_transition.py','examples/embedded_wave_evidence.py',
        'configs/embedded-wave-source-review.json')}
    report=dict(kind='embedded_cuda_source_transitions',completed=False,passed=False,
        ready_for_training_data=False,training_updates=0,current_integrator_sha256=sources['source_sha256']['gpu_enriched_verlet.py'],
        checker_sha256=sha256_file(__file__),analysis_sources=analysis,runtime=runtime,
        scope='Actual archived/current dense/sparse CUDA integration parity on bounded SPD systems; no continuum or physical validation.')
    try:
        versions=[]
        for i,reference in enumerate(references):
            child=output/('version-'+str(i));proof=verify_version(reference,child,guard)
            if not proof['completed'] or not proof['passed']:
                raise ValueError('an archived CUDA version did not pass its complete parity check')
            versions.append(dict(archived_integrator_sha256=proof['archived_integrator_sha256'],
                path=str((child/'source-transition-check.json').relative_to(output)),
                sha256=sha256_file(child/'source-transition-check.json'),fingerprint=proof['fingerprint']))
        if len({v['archived_integrator_sha256'] for v in versions})!=len(versions):
            raise ValueError('CUDA transition references repeat the same archived integrator version')
        if (identities()!=sources or distribution_identity(guard)!=runtime
                or any(sha256_file(PROJECT/name)!=digest for name,digest in analysis.items())
                or any(sha256_file(output/v['path'])!=v['sha256'] for v in versions)):
            raise ValueError('CUDA transition sources, runtime or per-version evidence changed')
        guard.check(force=True)
        report.update(completed=True,passed=True,versions=versions)
    except Exception as error:
        report['error']=f'{type(error).__name__}: {error}'
        raise
    finally:
        report['resources']=guard.report();write_json(output/'source-transitions-check.json',sealed(report))
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-runs',type=Path,nargs='+',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    guard=ResourceGuard(ResourceLimits(max_seconds=120,max_rss_mib=1024,
        max_system_memory_percent=89.9,min_available_mib=768))
    report=verify(args.reference_runs,args.output,guard)
    print(json.dumps({k:report[k] for k in ('completed','passed','versions','fingerprint')}))
    return 0 if report['passed'] else 2


if __name__=='__main__':
    raise SystemExit(main())
