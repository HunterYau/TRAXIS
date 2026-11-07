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
from traxis.storage import atomic_write, canonical_hash, sha256_file, write_json
from traxis.waveform_checks import band_resolution, trace_discrepancies


@dataclass(frozen=True)
class HighOrderCADRefinementSettings:
    mesh_sizes_m: tuple[float, ...] = (.003, .0025, .002)
    substeps_per_output: tuple[int, ...] = (4, 8, 16)
    spatial_relative_l2_tolerance: float = .05
    temporal_relative_l2_tolerance: float = .01
    reference_rms_floor: float = 1e-18
    max_total_solves: int = 5
    compare_scattering: bool = False
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
        if type(self.max_total_solves) is not int or not 1 <= self.max_total_solves <= 10:
            raise ValueError('CAD refinement solve budget must lie in [1,10]')
        if type(self.compare_scattering) is not bool:
            raise ValueError('choose explicitly whether to compare defect-scattered traces')
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
    solves = 10 if settings.compare_scattering else 5
    if settings.max_total_solves < solves:
        raise ValueError('CAD refinement exceeds its declared solve budget')
    if settings.compare_scattering and (state != 'current' or not scenario.recesses
            or any(g != 1. for g in scenario.nuisance.current_gain_drift)):
        raise ValueError('scattering refinement needs current recess geometry and unity coupling drift')
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
        'source_sha256': reference_source_hashes(), 'requested_forward_solves': solves,
        'definitions': [asdict(case) for case in definitions], 'ready_for_training_data': False}
    plan['fingerprint'] = canonical_hash(plan)
    write_json(output/'refinement-plan.json', plan)
    report = {'kind': 'highorder_cad_refinement', 'completed': False,
        'discretization_screen_passed': False, 'ready_for_training_data': False,
        'training_updates': 0, 'plan_fingerprint': plan['fingerprint'], 'cases': []}
    raw, scattered, runtime = [], [], None
    if settings.compare_scattering:
        report.update({'baseline_cases': [], 'scattered_cases': []})
    started = time.perf_counter()
    try:
        for i, case in enumerate(definitions):
            values, row, runtime = _checked_case(scenario, case, output/f'case-{i:02d}',
                guard, plan, cache_dir, runtime)
            raw.append(values)
            report['cases'].append(row)
            if settings.compare_scattering:
                baseline, baseline_row, runtime = _checked_case(scenario, case,
                    output/'baseline'/f'case-{i:02d}', guard, {**plan, 'state': 'intact'}, cache_dir, runtime)
                baseline_row['directory'] = f'baseline/case-{i:02d}'
                report['baseline_cases'].append(baseline_row)
                difference = values-baseline
                scattered.append(difference)
                directory = output/'scattered'/f'case-{i:02d}'
                directory.mkdir(parents=True, exist_ok=True)
                path = directory/'solver-traces.npz'
                atomic_write(path, lambda stream: np.savez_compressed(stream,
                    time_s=np.arange(scenario.acquisition.samples)*scenario.acquisition.sample_dt_s,
                    displacement_m=difference))
                report['scattered_cases'].append({**row, 'directory': f'scattered/case-{i:02d}',
                    'float64_traces_sha256': sha256_file(path),
                    'geometry_load_fingerprint': canonical_hash({'current': row['geometry_load_fingerprint'],
                        'intact': baseline_row['geometry_load_fingerprint']}),
                    'current_traces_sha256': row['float64_traces_sha256'],
                    'intact_traces_sha256': baseline_row['float64_traces_sha256']})
        if reference_source_hashes() != plan['source_sha256']:
            raise ValueError('high-order source changed during the frozen CAD study')
        if len({report['cases'][i]['geometry_load_fingerprint'] for i in (2, 3, 4)}) != 1:
            raise ValueError('CAD time refinement changed the spatial mesh, footprints or operators')
        if settings.compare_scattering and len({report['scattered_cases'][i]['geometry_load_fingerprint'] for i in (2, 3, 4)}) != 1:
            raise ValueError('scattered time refinement changed an intact/current mesh or operator')
        times = np.arange(scenario.acquisition.samples)*scenario.acquisition.sample_dt_s
        def compare(data, a, b):
            return trace_discrepancies(data[a], data[b], times, [a.frequency_hz for a in actions],
                rms_floor=settings.reference_rms_floor)
        spatial = [compare(raw, 0, 1), compare(raw, 1, 2)]
        temporal = [compare(raw, 3, 4), compare(raw, 4, 2)]
        def screen(rows, tolerance):
            return bool(all(r['all_records_resolved'] for r in rows) and rows[-1]['maximum_relative_l2'] <= tolerance)
        space_pass = screen(spatial, settings.spatial_relative_l2_tolerance)
        time_pass = screen(temporal, settings.temporal_relative_l2_tolerance)
        ratio = temporal[0]['global_difference_norm']/temporal[1]['global_difference_norm'] if temporal[1]['global_difference_norm'] else None
        if settings.compare_scattering:
            scatter_space = [compare(scattered, 0, 1), compare(scattered, 1, 2)]
            scatter_time = [compare(scattered, 3, 4), compare(scattered, 4, 2)]
            scatter_space_pass = screen(scatter_space, settings.spatial_relative_l2_tolerance)
            scatter_time_pass = screen(scatter_time, settings.temporal_relative_l2_tolerance)
            scatter_ratio = scatter_time[0]['global_difference_norm']/scatter_time[1]['global_difference_norm'] if scatter_time[1]['global_difference_norm'] else None
            report.update({'current_spatial_screen_passed': space_pass, 'current_temporal_screen_passed': time_pass,
                'scattered_spatial_comparisons': scatter_space, 'scattered_temporal_comparisons': scatter_time,
                'scattered_spatial_screen_passed': scatter_space_pass, 'scattered_temporal_screen_passed': scatter_time_pass,
                'scattered_observed_temporal_order': math.log(scatter_ratio)/math.log(settings.substeps_per_output[1]/settings.substeps_per_output[0]) if scatter_ratio and scatter_ratio > 0 else None})
            space_pass, time_pass = space_pass and scatter_space_pass, time_pass and scatter_time_pass
        guard.check(force=True)
        report.update({'completed': True, 'spatial_comparisons': spatial, 'temporal_comparisons': temporal,
            'spatial_screen_passed': space_pass, 'temporal_screen_passed': time_pass,
            'discretization_screen_passed': space_pass and time_pass, 'temporal_change_ratio': ratio,
            'observed_temporal_order': math.log(ratio)/math.log(settings.substeps_per_output[1]/settings.substeps_per_output[0]) if ratio and ratio > 0 else None,
            'band_resolution': band_resolution(replace(scenario, recesses=()) if state == 'intact' else scenario, report['cases'][2]['linear_mesh']),
            'note': 'One mechanical group; current and optional paired defect-scattered discrepancies. CAD faceting and field discretization change together. No continuum, full-size, PZT or physical acceptance.'})
    except Exception as exc:
        report['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        report['wall_seconds'] = time.perf_counter()-started
        report['resources'] = guard.report()
        report['fingerprint'] = canonical_hash(report)
        write_json(output/'refinement-report.json', report)
    if settings.compare_scattering:
        view_plan = {**plan, 'signal': 'current_minus_intact', 'parent_plan_fingerprint': plan['fingerprint']}
        view_plan.pop('fingerprint')
        view_plan['fingerprint'] = canonical_hash(view_plan)
        view_report = {'kind': 'highorder_scattered_refinement', 'completed': True,
            'plan_fingerprint': view_plan['fingerprint'], 'parent_report_sha256': sha256_file(output/'refinement-report.json'),
            'spatial_screen_passed': report['scattered_spatial_screen_passed'],
            'temporal_screen_passed': report['scattered_temporal_screen_passed'],
            'discretization_screen_passed': report['scattered_spatial_screen_passed'] and report['scattered_temporal_screen_passed'],
            'spatial_comparisons': report['scattered_spatial_comparisons'],
            'temporal_comparisons': report['scattered_temporal_comparisons'],
            'observed_temporal_order': report['scattered_observed_temporal_order'],
            'ready_for_training_data': False, 'cases': [{**r, 'directory': Path(r['directory']).name} for r in report['scattered_cases']]}
        view_report['fingerprint'] = canonical_hash(view_report)
        write_json(output/'scattered/refinement-plan.json', view_plan)
        write_json(output/'scattered/refinement-report.json', view_report)
    return report
