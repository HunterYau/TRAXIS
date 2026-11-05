"""Separate in-plane, thickness and time discrepancy screens for H1 pulses."""

from dataclasses import asdict, dataclass, field, replace
import json
import math
from pathlib import Path
import time

import numpy as np

from traxis.config import positive
from traxis.elmer_plate_refinement import reference_source_hashes
from traxis.highorder_waveforms import HighOrderWaveSettings, _load_traces, reference_actions, run_highorder_waveform_case, screen_highorder_reference
from traxis.resources import ResourceGuard
from traxis.storage import canonical_hash, sha256_file, write_json
from traxis.waveform_checks import band_resolution, trace_discrepancies


@dataclass(frozen=True)
class HighOrderDirectionalSettings:
    in_plane_cells: tuple[int, ...] = (3, 4, 5)
    thickness_cells: tuple[int, ...] = (1, 2, 3)
    substeps_per_output: tuple[int, ...] = (4, 8, 16)
    in_plane_relative_l2_tolerance: float = .05
    thickness_relative_l2_tolerance: float = .05
    temporal_relative_l2_tolerance: float = .01
    reference_rms_floor: float = 1e-18
    max_total_solves: int = 7
    reference: HighOrderWaveSettings = field(default_factory=HighOrderWaveSettings)

    def __post_init__(self):
        for name in ('in_plane_cells', 'thickness_cells', 'substeps_per_output'):
            values = tuple(getattr(self, name))
            object.__setattr__(self, name, values)
            if len(values) != 3 or any(type(n) is not int or n < 1 for n in values) or any(a >= b for a, b in zip(values, values[1:])):
                raise ValueError(f'{name} requires three increasing positive integer levels')
        steps = self.substeps_per_output
        if steps[-1] > 64 or steps[1] % steps[0] or steps[2]*steps[0] != steps[1]**2:
            raise ValueError('time refinements must be geometric integers with at most 64 substeps')
        for name in ('in_plane_relative_l2_tolerance', 'thickness_relative_l2_tolerance', 'temporal_relative_l2_tolerance', 'reference_rms_floor'):
            positive(name, getattr(self, name))
        if type(self.max_total_solves) is not int or not 1 <= self.max_total_solves <= 7 or not isinstance(self.reference, HighOrderWaveSettings):
            raise ValueError('choose bounded high-order settings and at most seven solves')

    @classmethod
    def from_dict(cls, values):
        values = dict(values)
        if 'reference' in values:
            values['reference'] = HighOrderWaveSettings(**values['reference'])
        return cls(**values)

    def cases(self):
        plane, thickness, steps = self.in_plane_cells, self.thickness_cells, self.substeps_per_output
        definitions = [((n, n, thickness[-1]), steps[-1]) for n in plane]
        definitions += [((plane[-1], plane[-1], n), steps[-1]) for n in thickness[:2]]
        definitions += [((plane[-1], plane[-1], thickness[-1]), n) for n in steps[:2]]
        return tuple(replace(self.reference, cells=cells, substeps_per_output=n) for cells, n in definitions)


def _checked_case(scenario, settings, case, guard, plan, cache_dir, runtime):
    if reference_source_hashes() != plan['source_sha256']:
        raise ValueError('high-order source changed during the frozen study')
    action = plan['actions'][0]
    values, measured = run_highorder_waveform_case(scenario, settings, case, guard,
        transmitter=action['transmitter'], frequency_hz=action['frequency_hz'], cache_dir=cache_dir)
    case_plan = json.loads((case / 'case-plan.json').read_text())
    runtime = case_plan['runtime'] if runtime is None else runtime
    if (canonical_hash({k: v for k, v in case_plan.items() if k != 'fingerprint'}) != case_plan['fingerprint']
            or runtime != case_plan['runtime'] or case_plan['source_sha256'] != plan['source_sha256']
            or case_plan['actions'] != plan['actions'] or canonical_hash(case_plan['scenario']) != canonical_hash(plan['scenario'])
            or canonical_hash(case_plan['settings']) != canonical_hash(asdict(settings)) or case_plan['state'] != 'intact'
            or not measured['completed'] or measured['plan_fingerprint'] != case_plan['fingerprint']):
        raise ValueError('high-order case changed its frozen source/runtime/scenario/settings/actions')
    path = case / 'solver-traces.npz'
    if sha256_file(path) != measured['trace_sha256']:
        raise ValueError('saved high-order comparison traces changed')
    saved = _load_traces(path, scenario.acquisition.samples, len(plan['actions']), scenario.acquisition.sample_dt_s)
    if not np.array_equal(saved, values):
        raise ValueError('returned high-order traces disagree with saved comparison arrays')
    row = {'directory': case.name, 'substeps_per_output': settings.substeps_per_output,
        **{k: measured[k] for k in ('order', 'cells', 'dofs', 'dt_internal_s', 'internal_steps', 'linear_mesh',
            'geometry_load_fingerprint', 'geometry_load_sha256', 'cache_hit', 'forward_solves', 'native_directory',
            'worker_report_sha256', 'stiffness_nonzeros', 'mass_nonzeros', 'lu_factor_nonzeros',
            'maximum_relative_linear_residual', 'postburst_relative_energy_range', 'wall_seconds')},
        'case_plan_sha256': sha256_file(case / 'case-plan.json'), 'case_report_sha256': sha256_file(case / 'waveform-case.json'),
        'float64_traces_sha256': measured['trace_sha256']}
    return values, row, runtime


def run_highorder_directional(scenario, settings, output, guard=None, *, transmitter=0, frequency_hz=None, cache_dir=None):
    if settings.max_total_solves < 7:
        raise ValueError('high-order directional refinement exceeds its solve budget')
    definitions = settings.cases()
    for reference in definitions:
        reference_actions(scenario, reference, transmitter, frequency_hz)
        screen_highorder_reference(reference)
    actions = reference_actions(scenario, definitions[0], transmitter, frequency_hz)
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('high-order refinement needs a new output directory')
    output.mkdir(parents=True, exist_ok=True)
    guard = guard or ResourceGuard()
    plan = {'kind': 'predeclared_highorder_directional_refinement', 'scenario': asdict(scenario), 'settings': asdict(settings),
        'actions': [asdict(a) for a in actions], 'state': 'intact', 'source_sha256': reference_source_hashes(),
        'requested_forward_solves': 7, 'definitions': [asdict(case) for case in definitions],
        'comparison_indices': {'in_plane': [[0, 1], [1, 2]], 'thickness': [[3, 4], [4, 2]], 'temporal': [[5, 6], [6, 2]]},
        'ready_for_training_data': False}
    plan['fingerprint'] = canonical_hash(plan)
    write_json(output / 'refinement-plan.json', plan)
    report = {'kind': 'highorder_directional_refinement', 'solver_label': f'P{settings.reference.order} NGSolve',
        'completed': False, 'discretization_screen_passed': False, 'ready_for_training_data': False,
        'training_updates': 0, 'plan_fingerprint': plan['fingerprint'], 'cases': []}
    raw, runtime = [], None
    started = time.perf_counter()
    try:
        for index, reference in enumerate(definitions):
            values, row, runtime = _checked_case(scenario, reference, output / f'case-{index:02d}', guard, plan, cache_dir, runtime)
            report['cases'].append(row)
            raw.append(values)
        if reference_source_hashes() != plan['source_sha256']:
            raise ValueError('high-order source changed during the frozen study')
        if len({report['cases'][i]['geometry_load_fingerprint'] for i in (2, 5, 6)}) != 1:
            raise ValueError('high-order temporal comparison changed its mesh/loads/operators')
        times = np.arange(scenario.acquisition.samples)*scenario.acquisition.sample_dt_s
        passed = {}
        for axis, pairs in plan['comparison_indices'].items():
            rows = [trace_discrepancies(raw[a], raw[b], times, [action.frequency_hz for action in actions],
                rms_floor=settings.reference_rms_floor) for a, b in pairs]
            passed[axis] = bool(all(row['all_records_resolved'] for row in rows)
                and rows[-1]['maximum_relative_l2'] <= getattr(settings, f'{axis}_relative_l2_tolerance'))
            report[f'{axis}_comparisons'] = rows
            report[f'{axis}_screen_passed'] = passed[axis]
            ratio = rows[0]['global_difference_norm']/rows[1]['global_difference_norm'] if rows[1]['global_difference_norm'] else None
            report[f'{axis}_change_ratio'] = ratio
        ratio = report['temporal_change_ratio']
        report['observed_temporal_order'] = math.log(ratio)/math.log(settings.substeps_per_output[1]/settings.substeps_per_output[0]) if ratio and ratio > 0 else None
        # Thickness levels need not be geometric. A change ratio alone does
        # not imply a convergence order or justify Richardson extrapolation.
        report['observed_thickness_order'] = None
        guard.check(force=True)
        report.update({'completed': True, 'spatial_screen_passed': passed['in_plane'] and passed['thickness'],
            'discretization_screen_passed': all(passed.values()),
            'band_resolution': band_resolution(replace(scenario, recesses=()), report['cases'][2]['linear_mesh']),
            'note': 'One intact transmitter/frequency, fixed polynomial degree. Separate raw in-plane/thickness/time discrepancies; no continuum bound or PZT/physical acceptance.'})
    except Exception as exc:
        report['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        report['wall_seconds'] = time.perf_counter()-started
        report['resources'] = guard.report()
        report['fingerprint'] = canonical_hash(report)
        write_json(output / 'directional-report.json', report)
    return report
