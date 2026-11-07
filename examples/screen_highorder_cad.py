"""Count actual CAD/H1 mesh costs in an isolated worker without wave solves."""

import argparse
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import sys

from traxis.elmer_plate_refinement import reference_source_hashes
from traxis.elmer_reference import run_monitored
from traxis.highorder import highorder_dofs
from traxis.highorder_cad_refinement import HighOrderCADRefinementSettings
from traxis.highorder_waveforms import reference_actions, reference_mesh, waveform_runtime
from traxis.resources import ResourceGuard, ResourceLimits
from traxis.scenarios import load_scenario_plan
from traxis.storage import canonical_hash, sha256_file, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('plan', 'settings', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--index', type=int, default=0)
    parser.add_argument('--state', choices=['intact', 'current'], default='current')
    parser.add_argument('--max-seconds', type=float, default=60)
    parser.add_argument('--max-rss-mib', type=float, default=256)
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    output = args.output.resolve()
    guard = ResourceGuard(ResourceLimits(max_seconds=args.max_seconds,
        max_rss_mib=args.max_rss_mib, max_system_memory_percent=85, min_available_mib=2048))
    if not args.worker:
        if output.exists() and any(output.iterdir()):
            parser.error('topology screen needs a new output directory')
        output.mkdir(parents=True, exist_ok=True)
        command = [sys.executable, '-m', 'examples.screen_highorder_cad', '--worker',
            '--plan', str(args.plan.resolve()), '--settings', str(args.settings.resolve()),
            '--output', str(output), '--index', str(args.index), '--state', args.state,
            '--max-seconds', str(args.max_seconds), '--max-rss-mib', str(args.max_rss_mib)]
        result = {'completed': False, 'forward_solves': 0, 'training_updates': 0,
            'wrapper_source_sha256': sha256_file(Path(__file__))}
        try:
            result.update(run_monitored(command, Path(__file__).resolve().parents[1],
                {**os.environ, 'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1'}, guard, output/'screen.log'))
            result['completed'] = result['exit_code'] in (0, 2)
        except Exception as exc:
            result['error'] = f'{type(exc).__name__}: {exc}'
        finally:
            result['resources'] = guard.report()
            write_json(output/'monitor-result.json', result)
        print(f"Topology screen status: {result.get('exit_code', 'guard stopped')}; output: {output}")
        return result.get('exit_code', 1)
    inputs = {str(p.resolve()): sha256_file(p) for p in (args.plan, args.settings)}
    if args.settings.stat().st_size > 1024**2:
        raise MemoryError('topology settings exceed 1 MiB')
    scenarios = load_scenario_plan(args.plan)
    if not 0 <= args.index < len(scenarios):
        raise ValueError('scenario index is outside the plan')
    scenario = scenarios[args.index]
    settings = HighOrderCADRefinementSettings.from_dict(json.loads(args.settings.read_text(encoding='utf-8-sig')))
    if settings.compare_scattering and (args.state != 'current' or not scenario.recesses
            or any(g != 1. for g in scenario.nuisance.current_gain_drift)):
        raise ValueError('paired topology screening needs current recess geometry and unity drift')
    source = reference_source_hashes()
    runtime = waveform_runtime(settings.reference, guard)
    report = {'kind': 'cad_highorder_topology_screen', 'completed': False,
        'scenario': asdict(scenario), 'settings': asdict(settings), 'input_sha256': inputs,
        'source_sha256': source, 'runtime': runtime, 'forward_solves': 0, 'training_updates': 0,
        'ready_for_training_data': False, 'rows': [],
        'note': 'Actual H1 counts and assembly screen only; sparse fill, factor cost and wave accuracy are unmeasured.'}
    try:
        for size in settings.mesh_sizes_m:
            for state in ((args.state, 'intact') if settings.compare_scattering else (args.state,)):
                case = replace(settings.reference, mesh_size_m=size)
                reference_actions(scenario, case, state=state)
                mesh, geometry = reference_mesh(scenario, case, state, guard)
                counts = highorder_dofs(mesh, case.order)
                report['rows'].append({'mesh_size_m': size, 'state': state, 'geometry': geometry, 'counts': counts,
                    'dof_cap_passed': counts['vector_dofs'] <= case.max_dofs,
                    'assembly_cap_passed': counts['assembly_screen_mib'] <= case.max_assembly_mib})
        if reference_source_hashes() != source or waveform_runtime(settings.reference, guard) != runtime:
            raise ValueError('source/runtime changed during topology screening')
        if any(sha256_file(p) != digest for p, digest in inputs.items()):
            raise ValueError('input settings changed during topology screening')
        report['all_meshes_fit'] = all(r['dof_cap_passed'] and r['assembly_cap_passed'] for r in report['rows'])
        guard.check(force=True)
        report['completed'] = True
    except Exception as exc:
        report['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        report['resources'] = guard.report()
        report['fingerprint'] = canonical_hash(report)
        write_json(output/'topology-report.json', report)
    return 0 if report['all_meshes_fit'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
