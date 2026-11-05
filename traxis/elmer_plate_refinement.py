"""Independent mesh/time axes for a single bounded P2 mechanical group."""

from dataclasses import asdict, dataclass, field, replace
import json
import math
from pathlib import Path
import time
import zipfile

import numpy as np

from traxis.acquisition import action_catalog
from traxis.config import positive
from traxis.elmer_waveforms import ElmerWaveSettings, run_elmer_waveform_case
from traxis.resources import ResourceGuard
from traxis.storage import canonical_hash, sha256_file, write_json
from traxis.waveform_checks import band_resolution, trace_discrepancies


@dataclass(frozen=True)
class ElmerPlateRefinementSettings:
    mesh_sizes_m: tuple[float, ...] = (.003, .002, .0015)
    substeps_per_output: tuple[int, ...] = (4, 8, 16)
    spatial_relative_l2_tolerance: float = .05
    temporal_relative_l2_tolerance: float = .01
    reference_rms_floor: float = 1e-18
    max_total_solves: int = 5
    reference: ElmerWaveSettings = field(default_factory=ElmerWaveSettings)

    def __post_init__(self):
        for name in ('mesh_sizes_m', 'substeps_per_output'):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        if (len(self.mesh_sizes_m) != 3 or any(not math.isfinite(h) or h <= 0 for h in self.mesh_sizes_m)
                or any(a <= b for a, b in zip(self.mesh_sizes_m, self.mesh_sizes_m[1:]))):
            raise ValueError('need three strictly decreasing positive mesh sizes')
        n = self.substeps_per_output
        if (len(n) != 3 or any(type(v) is not int or not 1 <= v <= 64 for v in n)
                or n[1] <= n[0] or n[1] % n[0] or n[2] * n[0] != n[1]**2):
            raise ValueError('need three geometric integer substep refinements within [1,64]')
        for name in ('spatial_relative_l2_tolerance', 'temporal_relative_l2_tolerance', 'reference_rms_floor'):
            positive(name, getattr(self, name))
        if type(self.max_total_solves) is not int or not 1 <= self.max_total_solves <= 5:
            raise ValueError('total solve budget must lie in [1,5]')
        if not isinstance(self.reference, ElmerWaveSettings):
            raise ValueError('reference settings must be validated ElmerWaveSettings')

    @classmethod
    def from_dict(cls, values):
        values = dict(values)
        if 'reference' in values:
            values['reference'] = ElmerWaveSettings(**values['reference'])
        return cls(**values)


def run_elmer_plate_refinement(scenario, settings, home, output, guard=None, *,
                               transmitter=0, frequency_hz=None, state='intact', cache_dir=None):
    if scenario.nuisance.relative_noise_std or scenario.nuisance.absolute_noise_std_m or scenario.nuisance.support != 'free':
        raise ValueError('P2 refinement requires zero stochastic noise and a free plate')
    if state not in ('intact', 'current') or type(transmitter) is not int or not 0 <= transmitter < scenario.acquisition.sensor_count:
        raise ValueError('invalid reference state or transmitter')
    frequency = scenario.acquisition.frequencies_hz[0] if frequency_hz is None else frequency_hz
    actions = tuple(a for a in action_catalog(scenario.acquisition.sensor_count, scenario.acquisition.frequencies_hz,
                                             scenario.acquisition.reciprocal)
                    if a.transmitter == transmitter and a.frequency_hz == frequency)
    if not actions:
        raise ValueError('requested transmitter/frequency has no eligible receive records')
    if settings.max_total_solves < 5:
        raise ValueError('P2 refinement exceeds the total solve budget')
    if (scenario.acquisition.samples - 1) * settings.substeps_per_output[-1] > settings.reference.max_internal_steps:
        raise ValueError('P2 refinement exceeds the internal-step limit before meshing')
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('P2 refinement needs a new or empty output directory')
    output.mkdir(parents=True, exist_ok=True)
    guard = guard or ResourceGuard()
    def sources():
        return {p.name: sha256_file(p) for p in sorted(Path(__file__).parent.glob('*.py'))}
    definitions = [(h, settings.substeps_per_output[-1]) for h in settings.mesh_sizes_m]
    definitions += [(settings.mesh_sizes_m[-1], n) for n in settings.substeps_per_output[:2]]
    plan = {'kind': 'predeclared_elmer_p2_plate_refinement', 'scenario': asdict(scenario), 'settings': asdict(settings),
            'actions': [asdict(a) for a in actions], 'state': state, 'source_sha256': sources(),
            'requested_forward_solves': 5, 'ready_for_training_data': False,
            'definitions': [{'mesh_size_m': h, 'substeps_per_output': n} for h, n in definitions]}
    plan['fingerprint'] = canonical_hash(plan)
    write_json(output / 'refinement-plan.json', plan)
    report = {'kind': 'elmer_p2_plate_refinement', 'completed': False, 'discretization_screen_passed': False,
              'ready_for_training_data': False, 'training_updates': 0, 'plan_fingerprint': plan['fingerprint'], 'cases': []}
    started = time.perf_counter()
    raw = []
    kernel = None
    try:
        for index, (size, substeps) in enumerate(definitions):
            guard.check(force=True)
            if sources() != plan['source_sha256']:
                raise ValueError('reference source changed during the frozen study')
            case = output / f'case-{index:02d}'
            case_settings = replace(settings.reference, mesh_size_m=size, substeps_per_output=substeps)
            traces, measured = run_elmer_waveform_case(scenario, case_settings, home, case, guard,
                transmitter=transmitter, frequency_hz=frequency, state=state, cache_dir=cache_dir)
            case_plan = json.loads((case / 'case-plan.json').read_text(encoding='utf-8'))
            if kernel is None:
                kernel = (case_plan['kernel_sha256'], case_plan['runtime'])
            if (canonical_hash({k: v for k, v in case_plan.items() if k != 'fingerprint'}) != case_plan['fingerprint']
                    or (case_plan['kernel_sha256'], case_plan['runtime']) != kernel
                    or case_plan['settings'] != asdict(case_settings) or case_plan['source_sha256'] != plan['source_sha256']
                    or case_plan['actions'] != plan['actions'] or canonical_hash(case_plan['scenario']) != canonical_hash(plan['scenario'])
                    or case_plan['state'] != state or not measured['completed']):
                raise ValueError('reference case changed the frozen source/kernel/scenario/actions/state')
            if traces.shape != (len(actions), scenario.acquisition.samples) or not np.isfinite(traces).all():
                raise ValueError('reference traces disagree with the frozen actions/samples')
            trace_path = case / 'solver-traces.npz'
            if sha256_file(trace_path) != measured['trace_sha256']:
                raise ValueError('saved reference traces changed')
            with zipfile.ZipFile(trace_path) as archive:
                if len(archive.infolist()) != 2 or sum(e.file_size for e in archive.infolist()) > 16 * 1024**2:
                    raise MemoryError('reference trace archive exceeds 16 MiB or has invalid entries')
            with np.load(trace_path, allow_pickle=False) as saved:
                expected_times = np.arange(scenario.acquisition.samples) * scenario.acquisition.sample_dt_s
                if not np.array_equal(saved['displacement_m'], traces) or not np.array_equal(saved['time_s'], expected_times):
                    raise ValueError('saved reference traces/time differ from the compared values')
            geometry = {p.name: sha256_file(p) for p in sorted((case / 'mesh').iterdir()) if p.is_file()}
            geometry['loads-probes.npz'] = sha256_file(case / 'loads-probes.npz')
            report['cases'].append({'directory': case.name, 'mesh_size_m': size, 'substeps_per_output': substeps,
                **{k: measured[k] for k in ('dt_internal_s', 'internal_steps', 'quadratic_nodes', 'tetrahedra',
                    'forward_solves', 'cache_hit', 'wall_seconds', 'computation', 'linear_mesh')},
                'geometry_load_fingerprint': canonical_hash(geometry), 'geometry_load_sha256': geometry,
                'case_plan_sha256': sha256_file(case / 'case-plan.json'),
                'case_report_sha256': sha256_file(case / 'waveform-case.json'),
                'float64_traces_sha256': sha256_file(case / 'solver-traces.npz')})
            raw.append(traces)
        if sources() != plan['source_sha256']:
            raise ValueError('reference source changed during the frozen study')
        if len({report['cases'][i]['geometry_load_fingerprint'] for i in (2, 3, 4)}) != 1:
            raise ValueError('temporal comparison changed the spatial mesh or physical apertures')
        times = np.arange(scenario.acquisition.samples) * scenario.acquisition.sample_dt_s
        def compare(a, b):
            return trace_discrepancies(raw[a], raw[b], times, [a.frequency_hz for a in actions],
                                       rms_floor=settings.reference_rms_floor)
        spatial, temporal = [compare(0, 1), compare(1, 2)], [compare(3, 4), compare(4, 2)]
        def screen(rows, tolerance):
            return all(row['all_records_resolved'] for row in rows) and rows[-1]['maximum_relative_l2'] <= tolerance
        spatial_pass = screen(spatial, settings.spatial_relative_l2_tolerance)
        temporal_pass = screen(temporal, settings.temporal_relative_l2_tolerance)
        ratio = temporal[0]['global_difference_norm'] / temporal[1]['global_difference_norm'] if temporal[1]['global_difference_norm'] else None
        band_scenario = replace(scenario, recesses=()) if state == 'intact' else scenario
        report.update({'completed': True, 'spatial_comparisons': spatial, 'temporal_comparisons': temporal,
            'spatial_screen_passed': spatial_pass, 'temporal_screen_passed': temporal_pass,
            'discretization_screen_passed': spatial_pass and temporal_pass, 'temporal_change_ratio': ratio,
            'observed_temporal_order': math.log(ratio) / math.log(settings.substeps_per_output[1] / settings.substeps_per_output[0]) if ratio and ratio > 0 else None,
            'band_resolution': band_resolution(band_scenario, report['cases'][2]['linear_mesh']),
            'note': 'One transmitter/frequency and one geometry. Provisional discrepancy screen, not a continuum bound, complete bank, PZT or physical validation.'})
        guard.check(force=True)
    except Exception as exc:
        report['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        report['wall_seconds'] = time.perf_counter() - started
        report['resources'] = guard.report()
        report['fingerprint'] = canonical_hash(report)
        write_json(output / 'refinement-report.json', report)
    return report
