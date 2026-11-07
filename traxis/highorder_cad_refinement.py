"""Independent CAD mesh and time discrepancy screens for high-order pulses."""

from dataclasses import asdict, dataclass, field, replace
import math
from pathlib import Path
import time

import numpy as np

from traxis.config import positive
from traxis.elmer_plate_refinement import reference_source_hashes
from traxis.highorder_directional import _checked_case
from traxis.highorder_waveforms import HighOrderWaveSettings, reference_actions
from traxis.resources import ResourceGuard
from traxis.storage import canonical_hash, write_json
from traxis.waveform_checks import band_resolution, trace_discrepancies


@dataclass(frozen=True)
class HighOrderCADRefinementSettings:
    mesh_sizes_m: tuple[float, ...] = (.003, .0025, .002)
    substeps_per_output: tuple[int, ...] = (4, 8, 16)
    spatial_relative_l2_tolerance: float = .05
    temporal_relative_l2_tolerance: float = .01
    reference_rms_floor: float = 1e-18
    max_total_solves: int = 5
    reference: HighOrderWaveSettings = field(default_factory=lambda: HighOrderWaveSettings(order=3, mesh_backend='cad'))

    def __post_init__(self):
        sizes, steps = tuple(self.mesh_sizes_m), tuple(self.substeps_per_output)
        object.__setattr__(self, 'mesh_sizes_m', sizes)
        object.__setattr__(self, 'substeps_per_output', steps)
        if (len(sizes) != 3 or any(not math.isfinite(h) or h <= 0 for h in sizes)
                or any(a <= b for a, b in zip(sizes, sizes[1:]))):
            raise ValueError('CAD refinement needs three decreasing positive mesh sizes')
        if (len(steps) != 3 or any(type(n) is not int or not 1 <= n <= 64 for n in steps)
                or steps[1] <= steps[0] or steps[1] % steps[0] or steps[2]*steps[0] != steps[1]**2):
            raise ValueError('CAD refinement needs three geometric integer time refinements')
        for name in ('spatial_relative_l2_tolerance', 'temporal_relative_l2_tolerance', 'reference_rms_floor'):
            positive(name, getattr(self, name))
        if type(self.max_total_solves) is not int or not 1 <= self.max_total_solves <= 5:
            raise ValueError('CAD refinement solve budget must lie in [1,5]')
        if not isinstance(self.reference, HighOrderWaveSettings) or self.reference.mesh_backend != 'cad':
            raise ValueError('mesh-size refinement requires CAD high-order settings')

    @classmethod
    def from_dict(cls, values):
        values = dict(values)
        if 'reference' in values:
            values['reference'] = HighOrderWaveSettings(**values['reference'])
        return cls(**values)

    def cases(self):
        return tuple(replace(self.reference, mesh_size_m=h, substeps_per_output=n)
            for h, n in [(h, self.substeps_per_output[-1]) for h in self.mesh_sizes_m]
            + [(self.mesh_sizes_m[-1], n) for n in self.substeps_per_output[:2]])


def run_highorder_cad_refinement(scenario, settings, output, guard=None, *,
                                transmitter=0, frequency_hz=None, state='current', cache_dir=None):
    if settings.max_total_solves < 5:
        raise ValueError('CAD refinement exceeds its five-solve budget')
    definitions = settings.cases()
    for case in definitions:
        reference_actions(scenario, case, transmitter, frequency_hz, state=state)
    actions = reference_actions(scenario, definitions[0], transmitter, frequency_hz, state=state)
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('CAD refinement needs a new output directory')
    output.mkdir(parents=True, exist_ok=True)
    guard = guard or ResourceGuard()
    plan = {'kind': 'predeclared_highorder_cad_refinement', 'scenario': asdict(scenario),
        'settings': asdict(settings), 'actions': [asdict(a) for a in actions], 'state': state,
        'source_sha256': reference_source_hashes(), 'requested_forward_solves': 5,
        'definitions': [asdict(case) for case in definitions], 'ready_for_training_data': False}
    plan['fingerprint'] = canonical_hash(plan)
    write_json(output/'refinement-plan.json', plan)
    report = {'kind': 'highorder_cad_refinement', 'completed': False,
        'discretization_screen_passed': False, 'ready_for_training_data': False,
        'training_updates': 0, 'plan_fingerprint': plan['fingerprint'], 'cases': []}
    raw, runtime = [], None
    started = time.perf_counter()
    try:
        for i, case in enumerate(definitions):
            values, row, runtime = _checked_case(scenario, case, output/f'case-{i:02d}',
                guard, plan, cache_dir, runtime)
            raw.append(values)
            report['cases'].append(row)
        if reference_source_hashes() != plan['source_sha256']:
            raise ValueError('high-order source changed during the frozen CAD study')
        if len({report['cases'][i]['geometry_load_fingerprint'] for i in (2, 3, 4)}) != 1:
            raise ValueError('CAD time refinement changed the spatial mesh, footprints or operators')
        times = np.arange(scenario.acquisition.samples)*scenario.acquisition.sample_dt_s
        def compare(a, b):
            return trace_discrepancies(raw[a], raw[b], times, [a.frequency_hz for a in actions],
                rms_floor=settings.reference_rms_floor)
        spatial = [compare(0, 1), compare(1, 2)]
        temporal = [compare(3, 4), compare(4, 2)]
        def screen(rows, tolerance):
            return bool(all(r['all_records_resolved'] for r in rows) and rows[-1]['maximum_relative_l2'] <= tolerance)
        space_pass = screen(spatial, settings.spatial_relative_l2_tolerance)
        time_pass = screen(temporal, settings.temporal_relative_l2_tolerance)
        ratio = temporal[0]['global_difference_norm']/temporal[1]['global_difference_norm'] if temporal[1]['global_difference_norm'] else None
        guard.check(force=True)
        report.update({'completed': True, 'spatial_comparisons': spatial, 'temporal_comparisons': temporal,
            'spatial_screen_passed': space_pass, 'temporal_screen_passed': time_pass,
            'discretization_screen_passed': space_pass and time_pass, 'temporal_change_ratio': ratio,
            'observed_temporal_order': math.log(ratio)/math.log(settings.substeps_per_output[1]/settings.substeps_per_output[0]) if ratio and ratio > 0 else None,
            'band_resolution': band_resolution(replace(scenario, recesses=()) if state == 'intact' else scenario, report['cases'][2]['linear_mesh']),
            'note': 'One mechanical group and geometry; CAD faceting and wave discretization both change under mesh refinement. No continuum, scattered-signal, PZT or physical acceptance.'})
    except Exception as exc:
        report['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        report['wall_seconds'] = time.perf_counter()-started
        report['resources'] = guard.report()
        report['fingerprint'] = canonical_hash(report)
        write_json(output/'refinement-report.json', report)
    return report
