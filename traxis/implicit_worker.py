"""Isolated sparse factorization/time integration for bounded reference jobs."""

import argparse
import json
from pathlib import Path

import numpy as np

from traxis.elmer_operators import read_elmer_sparse_operators
from traxis.fem import tone_burst
from traxis.implicit_fem import implicit_simulate
from traxis.resources import ResourceGuard, ResourceLimits
from traxis.storage import atomic_write, canonical_hash, sha256_file, write_json


def run_job(path: Path, output: Path, guard):
    path, output = Path(path).resolve(), Path(output).resolve()
    if path.stat().st_size > 1024**2:
        raise MemoryError('implicit worker job exceeds 1 MiB')
    job = json.loads(path.read_text(encoding='utf-8'))
    if canonical_hash({k: v for k, v in job.items() if k != 'fingerprint'}) != job['fingerprint']:
        raise ValueError('implicit worker job changed')
    if set(job['input_sha256']) != {'operator_sizes.dat', 'operator_a.dat', 'operator_perm.dat', 'loads-probes.npz'}:
        raise ValueError('worker job must bind every operator/load input')
    required_source = {'implicit_worker.py', 'implicit_fem.py', 'elmer_operators.py', 'fem.py'}
    if not required_source.issubset(job['source_sha256']):
        raise ValueError('worker job is missing source bindings')
    for name, expected in job['source_sha256'].items():
        if Path(name).name != name or sha256_file(Path(__file__).parent / name) != expected:
            raise ValueError('worker source changed since the reference plan')
    if not output.is_relative_to(path.parent) or (output.exists() and any(output.iterdir())):
        raise ValueError('worker output must be a new directory within the job directory')
    for name, expected in job['input_sha256'].items():
        source = (path.parent / name).resolve()
        if not source.is_relative_to(path.parent) or sha256_file(source) != expected:
            raise ValueError('implicit worker input changed or escaped the job directory')
    output.mkdir(parents=True, exist_ok=True)
    report = {'kind': 'bounded_implicit_elastic_worker', 'completed': False, 'ready_for_training_data': False,
              'job_fingerprint': job['fingerprint']}
    try:
        guard.check(force=True)
        stiffness, mass = read_elmer_sparse_operators(path.parent, job['nodes'],
                                                      max_entries=job['max_operator_entries'], max_file_mib=job['max_operator_file_mib'])
        guard.check(force=True)
        import zipfile
        loads = path.parent / 'loads-probes.npz'
        with zipfile.ZipFile(loads) as archive:
            if len(archive.infolist()) != 2 or sum(row.file_size for row in archive.infolist()) > 16 * 1024**2:
                raise MemoryError('implicit force/probe archive exceeds 16 MiB')
        with np.load(loads, allow_pickle=False) as saved:
            force, probes = saved['force_n'].copy(), saved['probes'].copy()
        if force.shape != (job['nodes'] * 3,) or not np.isfinite(force).all():
            raise ValueError('invalid worker force shape/values')
        result = implicit_simulate(stiffness, mass, job['steps'], job['dt_s'], probes,
            force=lambda t: force * tone_burst(float(t), job['frequency_hz'], job['cycles']),
            output_stride=job['output_stride'], guard=guard)
        atomic_write(output / 'traces.npz', lambda stream: np.savez_compressed(stream, time_s=result.time_s,
            displacement_m=result.displacement_m.T, energy_time_s=result.energy_time_s, energy_j=result.energy_j))
        after = result.energy_j[result.energy_time_s > job['cycles'] / job['frequency_hz']]
        report.update({'completed': True, 'stiffness_nonzeros': stiffness.nnz, 'mass_nonzeros': mass.nnz,
            'lu_factor_nonzeros': result.factor_nonzeros, 'maximum_relative_linear_residual': result.maximum_relative_residual,
            'factorization': 'SuperLU symmetric mode; MMD_AT_PLUS_A; diagonal pivot threshold 0; reported storage nnz',
            'postburst_relative_energy_range': float(np.ptp(after) / abs(after.mean())) if len(after) > 1 and after.mean() else None,
            'traces_sha256': sha256_file(output / 'traces.npz')})
    except Exception as exc:
        report['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        report['resources'] = guard.report()
        report['fingerprint'] = canonical_hash(report)
        write_json(output / 'worker-report.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-seconds', type=float, default=60)
    parser.add_argument('--max-rss-mib', type=float, default=512)
    args = parser.parse_args()
    guard = ResourceGuard(ResourceLimits(max_seconds=args.max_seconds, max_rss_mib=args.max_rss_mib))
    try:
        result = run_job(args.job, args.output, guard)
        print(json.dumps({'completed': result['completed'], 'ready_for_training_data': False}))
    except Exception as exc:
        parser.exit(1, f'Implicit worker stopped: {type(exc).__name__}: {exc}\n')


if __name__ == '__main__':
    main()
