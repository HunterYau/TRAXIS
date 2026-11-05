"""Run the seven-case intact P2 study under an outer process-tree guard."""

import argparse
from pathlib import Path
import sys

from traxis.elmer_reference import elmer_environment, run_monitored
from traxis.resources import ResourceGuard, ResourceLimits
from traxis.storage import sha256_file, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('plan', 'settings', 'elmer-home', 'output', 'monitor-output', 'cache-dir'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--max-seconds', type=float, default=600)
    parser.add_argument('--max-rss-mib', type=float, default=1280)
    parser.add_argument('--max-system-memory-percent', type=float, default=85)
    parser.add_argument('--min-available-mib', type=float, default=2048)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    monitor = args.monitor_output.resolve()
    if monitor.exists() and any(monitor.iterdir()):
        parser.error('monitor output needs a new or empty directory')
    limits = ResourceLimits(max_seconds=args.max_seconds + 10, max_rss_mib=args.max_rss_mib,
                           max_system_memory_percent=args.max_system_memory_percent, min_available_mib=args.min_available_mib)
    _, env = elmer_environment(args.elmer_home.resolve())
    command = [sys.executable, '-m', 'traxis', 'check-directional-waveforms', '--plan', str(args.plan.resolve()),
        '--settings', str(args.settings.resolve()), '--elmer-home', str(args.elmer_home.resolve()),
        '--output', str(args.output.resolve()), '--cache-dir', str(args.cache_dir.resolve()),
        '--max-seconds', str(args.max_seconds), '--max-rss-mib', str(args.max_rss_mib)]
    monitor.mkdir(parents=True, exist_ok=True)
    guard = ResourceGuard(limits)
    write_json(monitor / 'invocation.json', {'command': command, 'outer_resource_limits': guard.report()['limits'],
        'wrapper_source_sha256': sha256_file(Path(__file__)), 'ready_for_training_data': False})
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
