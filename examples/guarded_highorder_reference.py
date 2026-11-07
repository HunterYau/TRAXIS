"""Run a higher-order directional or CAD study under a process-tree guard."""

import argparse
import os
from pathlib import Path
import sys

from traxis.elmer_reference import run_monitored
from traxis.resources import ResourceGuard, ResourceLimits
from traxis.storage import sha256_file, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('plan', 'settings', 'output', 'monitor-output', 'cache-dir'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--cad', action='store_true', help='use CAD mesh-size/time refinement')
    parser.add_argument('--index', type=int, default=0)
    parser.add_argument('--state', choices=['intact', 'current'], default='current', help='geometry state for CAD studies')
    parser.add_argument('--max-seconds', type=float, default=600)
    parser.add_argument('--max-rss-mib', type=float, default=1024)
    parser.add_argument('--max-system-memory-percent', type=float, default=85)
    parser.add_argument('--min-available-mib', type=float, default=2048)
    args = parser.parse_args()
    root, monitor = Path(__file__).resolve().parents[1], args.monitor_output.resolve()
    if monitor.exists() and any(monitor.iterdir()):
        parser.error('monitor output needs a new directory')
    guard = ResourceGuard(ResourceLimits(max_seconds=args.max_seconds+10, max_rss_mib=args.max_rss_mib,
        max_system_memory_percent=args.max_system_memory_percent, min_available_mib=args.min_available_mib))
    command = [sys.executable, '-m', 'traxis', 'check-highorder-cad-waveforms' if args.cad else 'check-highorder-waveforms', '--plan', str(args.plan.resolve()),
        '--settings', str(args.settings.resolve()), '--output', str(args.output.resolve()), '--cache-dir', str(args.cache_dir.resolve()),
        '--index', str(args.index), '--max-seconds', str(args.max_seconds), '--max-rss-mib', str(args.max_rss_mib)]
    if args.cad:
        command += ['--state', args.state]
    monitor.mkdir(parents=True, exist_ok=True)
    write_json(monitor / 'invocation.json', {'command': command, 'outer_resource_limits': guard.report()['limits'],
        'wrapper_source_sha256': sha256_file(Path(__file__)), 'ready_for_training_data': False})
    env = os.environ.copy()
    env.update({'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1'})
    result = {'completed': False, 'ready_for_training_data': False}
    try:
        result.update(run_monitored(command, root, env, guard, monitor / 'study.log'))
        result['completed'] = result['exit_code'] in (0, 2)
    except Exception as exc:
        result['error'] = f'{type(exc).__name__}: {exc}'
    finally:
        result['resources'] = guard.report()
        write_json(monitor / 'monitor-result.json', result)
    print(f"Study exit status: {result.get('exit_code', 'guard stopped')}; monitor: {monitor}")
    return result.get('exit_code', 1)


if __name__ == '__main__':
    raise SystemExit(main())
