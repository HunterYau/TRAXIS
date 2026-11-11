"""Independent in-plane, thickness and time refinements of an intact P2 box."""

from dataclasses import asdict, dataclass, field, replace
import math
from pathlib import Path
import time

import numpy as np

from traxis.acquisition import action_catalog
from traxis.config import positive
from traxis.elmer_plate_refinement import checked_reference_case, reference_source_hashes
from traxis.elmer_waveforms import ElmerWaveSettings, screen_structured_reference
from traxis.resources import ResourceGuard
from traxis.storage import canonical_hash, write_json
from traxis.waveform_checks import band_resolution, trace_discrepancies


@dataclass(frozen=True)
class ElmerDirectionalSettings:
    in_plane_cells: tuple[int, ...] = (10, 15, 20)
    thickness_cells: tuple[int, ...] = (1, 2, 4)
    substeps_per_output: tuple[int, ...] = (4, 8, 16)
    in_plane_relative_l2_tolerance: float = .05
    thickness_relative_l2_tolerance: float = .05
    temporal_relative_l2_tolerance: float = .01
    reference_rms_floor: float = 1e-18
    max_total_solves: int = 7
    reference: ElmerWaveSettings = field(default_factory=lambda: ElmerWaveSettings(mesh_backend='structured', max_matrix_screen_mib=280))

    def __post_init__(self):
        for name in ('in_plane_cells', 'thickness_cells', 'substeps_per_output'):
            values = tuple(getattr(self, name))
            object.__setattr__(self, name, values)
            if (len(values) != 3 or any(type(n) is not int or n < 1 for n in values)
                    or any(a >= b for a, b in zip(values, values[1:]))):
                raise ValueError(f'{name} requires three strictly increasing positive integers')
        for name in ('thickness_cells', 'substeps_per_output'):
            n = getattr(self, name)
            if n[1] % n[0] or n[2] * n[0] != n[1]**2:
                raise ValueError(f'{name} must use geometric integer refinements')
        if self.substeps_per_output[-1] > 64:
            raise ValueError('substeps must not exceed 64')
        for name in ('in_plane_relative_l2_tolerance', 'thickness_relative_l2_tolerance',
                     'temporal_relative_l2_tolerance', 'reference_rms_floor'):
            positive(name, getattr(self, name))
        if type(self.max_total_solves) is not int or not 1 <= self.max_total_solves <= 7:
            raise ValueError('total solve budget must lie in [1,7]')
        if not isinstance(self.reference, ElmerWaveSettings) or self.reference.mesh_backend != 'structured':
            raise ValueError('directional refinement requires structured reference settings')

    @classmethod
    def from_dict(cls, values):
        values = dict(values)
        if 'reference' in values:
            values['reference'] = ElmerWaveSettings(**values['reference'])
        return cls(**values)

    def cases(self):
        plane, thickness, steps = self.in_plane_cells, self.thickness_cells, self.substeps_per_output
        definitions = [((n, n, thickness[-1]), steps[-1]) for n in plane]
        definitions += [((plane[-1], plane[-1], n), steps[-1]) for n in thickness[:2]]
        definitions += [((plane[-1], plane[-1], thickness[-1]), n) for n in steps[:2]]
        return tuple(replace(self.reference, cells=cells, substeps_per_output=n) for cells, n in definitions)


def run_elmer_directional_refinement(scenario, settings, home, output, guard=None, *,
                                     transmitter=0, frequency_hz=None, cache_dir=None):
    if scenario.nuisance.relative_noise_std or scenario.nuisance.absolute_noise_std_m or scenario.nuisance.support != 'free':
        raise ValueError('directional refinement requires zero stochastic noise and a free plate')
    if type(transmitter) is not int or not 0 <= transmitter < scenario.acquisition.sensor_count:
        raise ValueError('invalid directional transmitter')
    frequency = scenario.acquisition.frequencies_hz[0] if frequency_hz is None else frequency_hz
    actions = tuple(a for a in action_catalog(scenario.acquisition.sensor_count, scenario.acquisition.frequencies_hz,
                                             scenario.acquisition.reciprocal)
                    if a.transmitter == transmitter and a.frequency_hz == frequency)
    if not actions:
        raise ValueError('requested transmitter/frequency has no eligible receive records')
    if settings.max_total_solves < 7:
        raise ValueError('directional refinement exceeds the total solve budget')
    if (scenario.acquisition.samples - 1) * settings.substeps_per_output[-1] > settings.reference.max_internal_steps:
        raise ValueError('directional refinement exceeds the internal-step limit before meshing')
    definitions = settings.cases()
    for reference in definitions:
        screen_structured_reference(reference)
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('directional refinement needs a new or empty output directory')
    output.mkdir(parents=True, exist_ok=True)
    guard = guard or ResourceGuard()
    plan = {'kind': 'predeclared_elmer_p2_directional_refinement', 'scenario': scenario.to_dict(), 'settings': asdict(settings),
            'actions': [asdict(a) for a in actions], 'state': 'intact', 'source_sha256': reference_source_hashes(),
            'requested_forward_solves': 7, 'definitions': [asdict(case) for case in definitions],
            'comparison_indices': {'in_plane': [[0, 1], [1, 2]], 'thickness': [[3, 4], [4, 2]], 'temporal': [[5, 6], [6, 2]]},
            'ready_for_training_data': False}
    plan['fingerprint'] = canonical_hash(plan)
    write_json(output / 'refinement-plan.json', plan)
    report = {'kind': 'elmer_p2_directional_refinement', 'completed': False, 'discretization_screen_passed': False,
              'ready_for_training_data': False, 'training_updates': 0, 'plan_fingerprint': plan['fingerprint'], 'cases': []}
    raw, kernel = [], None
    started = time.perf_counter()
    try:
        for index, reference in enumerate(definitions):
            values, row, kernel = checked_reference_case(scenario, reference, home, output / f'case-{index:02d}',
                                                         guard, plan, cache_dir, kernel)
            report['cases'].append(row)
            raw.append(values)
        if reference_source_hashes() != plan['source_sha256']:
            raise ValueError('reference source changed during the frozen directional study')
        if len({report['cases'][i]['geometry_load_fingerprint'] for i in (2, 5, 6)}) != 1:
            raise ValueError('temporal comparison changed the spatial mesh or physical apertures')
        times = np.arange(scenario.acquisition.samples) * scenario.acquisition.sample_dt_s
        comparisons = {axis: [trace_discrepancies(raw[a], raw[b], times, [a.frequency_hz for a in actions],
                       rms_floor=settings.reference_rms_floor) for a, b in pairs]
                       for axis, pairs in plan['comparison_indices'].items()}
        passed = {}
        for axis, rows in comparisons.items():
            tolerance = getattr(settings, f'{axis}_relative_l2_tolerance')
            passed[axis] = all(row['all_records_resolved'] for row in rows) and rows[-1]['maximum_relative_l2'] <= tolerance
            report[f'{axis}_comparisons'] = rows
            report[f'{axis}_screen_passed'] = passed[axis]
        for axis, levels in (('thickness', settings.thickness_cells), ('temporal', settings.substeps_per_output)):
            rows = comparisons[axis]
            ratio = rows[0]['global_difference_norm'] / rows[1]['global_difference_norm'] if rows[1]['global_difference_norm'] else None
            report[f'{axis}_change_ratio'] = ratio
            report[f'observed_{axis}_order'] = math.log(ratio) / math.log(levels[1] / levels[0]) if ratio and ratio > 0 else None
        guard.check(force=True)
        report.update({'completed': True, 'spatial_screen_passed': passed['in_plane'] and passed['thickness'],
            'discretization_screen_passed': all(passed.values()),
            'band_resolution': band_resolution(replace(scenario, recesses=()), report['cases'][2]['linear_mesh']),
            'note': 'One intact transmitter/frequency. Independent in-plane/thickness/time discrepancy screens, not a continuum bound or PZT/physical validation.'})
    except Exception as exc:
        report['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        report['wall_seconds'] = time.perf_counter() - started
        report['resources'] = guard.report()
        report['fingerprint'] = canonical_hash(report)
        write_json(output / 'directional-report.json', report)
    return report
