"""Run regression tests under a process-tree memory and temperature guard."""

import argparse
import os
from pathlib import Path
import sys

from traxis.elmer_reference import run_monitored
from traxis.resources import ResourceGuard, ResourceLimits
from traxis.storage import sha256_file, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--elmer-home', type=Path)
    parser.add_argument('--highorder', action='store_true')
    parser.add_argument('--max-seconds', type=float, default=300)
    parser.add_argument('--max-rss-mib', type=float, default=1024)
    parser.add_argument('--max-system-memory-percent', type=float, default=85)
    parser.add_argument('--min-available-mib', type=float, default=2048)
    parser.add_argument('tests', nargs='*', help='optional unittest module names')
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        parser.error('test output needs a new directory')
    if args.elmer_home and not (args.elmer_home / 'bin' / 'ElmerSolver.exe').is_file():
        parser.error('Elmer home must contain bin/ElmerSolver.exe')
    guard = ResourceGuard(ResourceLimits(max_seconds=args.max_seconds,
        max_rss_mib=args.max_rss_mib,
        max_system_memory_percent=args.max_system_memory_percent,
        min_available_mib=args.min_available_mib))
    root = Path(__file__).resolve().parents[1]
    command = [sys.executable, '-m', 'unittest']
    command += ['-v', *args.tests] if args.tests else ['discover', '-v']
    env = os.environ.copy()
    env.update({'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1'})
    if args.elmer_home:
        env['TRAXIS_ELMER_HOME'] = str(args.elmer_home.resolve())
    if args.highorder:
        env['TRAXIS_HIGHORDER_CHECKS'] = '1'
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / 'invocation.json', {'command': command,
        'limits': guard.report()['limits'], 'elmer_home': env.get('TRAXIS_ELMER_HOME'),
        'highorder_checks': env.get('TRAXIS_HIGHORDER_CHECKS') == '1',
        'wrapper_source_sha256': sha256_file(Path(__file__))})
    result = {'completed': False, 'full_training_cycle_run': False}
    try:
        result.update(run_monitored(command, root, env, guard, output / 'tests.log'))
        result['completed'] = True
    except Exception as exc:
        result['error'] = f'{type(exc).__name__}: {exc}'
    finally:
        result['resources'] = guard.report()
        write_json(output / 'monitor-result.json', result)
    print(f"Test exit status: {result.get('exit_code', 'guard stopped')}; output: {output}")
    return result.get('exit_code', 1)


if __name__ == '__main__':
    raise SystemExit(main())
