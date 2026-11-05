"""Finite-plate trace refinement diagnostics; no training-data promotion."""

from dataclasses import asdict, dataclass, field, replace
import math
from pathlib import Path
import time

import numpy as np
from scipy.signal import hilbert

from traxis.acquisition import Action, action_catalog
from traxis.config import positive
from traxis.dispersion import mode_velocity
from traxis.forward import ForwardSettings, generate_waveform_bank, load_cached_group, minimum_stable_substeps
from traxis.resources import ResourceGuard
from traxis.scenarios import Scenario
from traxis.storage import atomic_write, canonical_hash, sha256_file, write_json


@dataclass(frozen=True)
class WaveformSweepSettings:
    mesh_sizes_m: tuple[float, ...] = (.003, .002, .0015)
    time_refinements: tuple[int, ...] = (1, 2, 4)
    spatial_relative_l2_tolerance: float = .05
    temporal_relative_l2_tolerance: float = .01
    reference_rms_floor: float = 1e-18
    max_total_solves: int = 64
    max_raw_trace_mib: float = 16
    forward: ForwardSettings = field(default_factory=lambda: ForwardSettings(max_internal_steps=50000))

    def __post_init__(self):
        object.__setattr__(self, "mesh_sizes_m", tuple(self.mesh_sizes_m))
        object.__setattr__(self, "time_refinements", tuple(self.time_refinements))
        if len(self.mesh_sizes_m) != 3 or any(not math.isfinite(h) or h <= 0 for h in self.mesh_sizes_m):
            raise ValueError("provide exactly three positive mesh sizes")
        if any(a <= b for a, b in zip(self.mesh_sizes_m, self.mesh_sizes_m[1:])):
            raise ValueError("mesh sizes must strictly decrease")
        factors = self.time_refinements
        if (len(factors) != 3 or factors[0] != 1 or any(type(n) is not int or not 1 <= n <= 16 for n in factors)
                or factors[1] <= 1 or factors[2] != factors[1]**2):
            raise ValueError("time refinements must be (1,r,r²), with integer r>1 and r²<=16")
        for name in ("spatial_relative_l2_tolerance", "temporal_relative_l2_tolerance", "reference_rms_floor", "max_raw_trace_mib"):
            positive(name, getattr(self, name))
        if type(self.max_total_solves) is not int or not 1 <= self.max_total_solves <= 128:
            raise ValueError("total solve limit must lie in [1,128]")
        if self.forward.mesh_backend != "cad" or self.forward.time_step_refinement != 1 or self.forward.fixed_substeps_per_output is not None:
            raise ValueError("this sweep uses CAD mesh-size and time-refinement axes; keep the base time refinement at one")

    @classmethod
    def from_dict(cls, values):
        data = dict(values)
        if "forward" in data:
            data["forward"] = ForwardSettings(**data["forward"])
        return cls(**data)


def trace_discrepancies(candidate, reference, time_s, frequencies_hz, *, rms_floor=1e-18):
    """Compare aligned traces with no amplitude fit, shift or interpolation.

    Envelope-peak differences are waveform features, not verified mode arrivals.
    Phase is the complex carrier component on the entire supplied time window.
    """
    candidate, reference, times = [np.asarray(v, dtype=float) for v in (candidate, reference, time_s)]
    positive("reference RMS floor", rms_floor)
    if (candidate.ndim != 2 or candidate.shape != reference.shape or times.shape != (candidate.shape[1],)
            or len(times) < 8 or len(frequencies_hz) != len(candidate)
            or not all(np.isfinite(v).all() for v in (candidate, reference, times))):
        raise ValueError("need finite matching (records,time) arrays, time vector and record frequencies")
    if np.any(np.diff(times) <= 0) or not np.allclose(np.diff(times), times[1] - times[0], rtol=1e-7, atol=1e-15):
        raise ValueError("trace comparison requires a uniform increasing time grid")
    for frequency in frequencies_hz:
        positive("frequency", frequency)
        if frequency >= .5 / (times[1] - times[0]):
            raise ValueError("comparison carrier must be below sampling Nyquist")
    ce, re = abs(hilbert(candidate, axis=1)), abs(hilbert(reference, axis=1))
    rows = []
    for c, r, ec, er, frequency in zip(candidate, reference, ce, re, frequencies_hz):
        norm, cnorm = float(np.linalg.norm(r)), float(np.linalg.norm(c))
        reference_rms = norm / math.sqrt(len(times))
        defined = reference_rms > rms_floor
        carrier = np.exp(-2j * np.pi * frequency * times)
        rc, cc = r @ carrier, c @ carrier
        phase_defined = defined and abs(rc) > 1e-8 * norm * math.sqrt(len(times)) and abs(cc) > 1e-8 * cnorm * math.sqrt(len(times))
        peaks_defined = defined and cnorm and np.ptp(er) > 1e-8 * er.mean() and np.ptp(ec) > 1e-8 * ec.mean()
        rows.append({"reference_rms": reference_rms, "candidate_rms": cnorm / math.sqrt(len(times)),
                     "rms_error": float(np.sqrt(np.mean((c - r)**2))),
                     "relative_l2": float(np.linalg.norm(c - r) / norm) if defined else None,
                     "relative_rms_error": abs(cnorm / norm - 1) if defined else None,
                     "cosine_correlation": float(c @ r / (cnorm * norm)) if defined and cnorm else None,
                     "envelope_relative_l2": float(np.linalg.norm(ec - er) / np.linalg.norm(er)) if defined else None,
                     "envelope_peak_shift_s": float(times[ec.argmax()] - times[er.argmax()]) if peaks_defined else None,
                     "carrier_phase_difference_rad": float(np.angle(cc * rc.conjugate())) if phase_defined else None})
    values = [r["relative_l2"] for r in rows if r["relative_l2"] is not None]
    return {"rows": rows, "records": len(rows), "defined_records": len(values),
            "maximum_relative_l2": max(values) if values else None,
            "global_difference_norm": float(np.linalg.norm(candidate - reference)),
            "all_records_resolved": len(values) == len(rows),
            "note": "Raw finite-plate displacements include reflections and rigid motion. Agreement is not independent wave/PZT/physical validation."}


def _raw_traces(scenario, bank, generated, cache):
    if scenario.nuisance.relative_noise_std or scenario.nuisance.absolute_noise_std_m:
        raise ValueError("numerical refinement requires noise-free source/control settings")
    lookup = {(row["kind"], row["transmitter"], row["frequency_hz"]): row["key"] for row in generated["groups"]}
    loaded = {}
    result = {}
    for channel, kind in (("current", "current" if scenario.recesses else "intact"), ("baseline", "intact")):
        values = np.empty(bank.waveforms.shape, np.float64)
        for i, action in enumerate(bank.actions):
            key = lookup[(kind, action.transmitter, action.frequency_hz)]
            if key not in loaded:
                loaded[key] = load_cached_group(cache, key, (scenario.acquisition.sensor_count, scenario.acquisition.samples))
            n = scenario.nuisance
            gain = n.peak_force_n * n.sensor_gains[action.transmitter] * n.sensor_gains[action.receiver]
            values[i] = loaded[key][action.receiver] * gain
            if channel == "current":
                values[i] = values[i] * n.current_gain_drift[action.transmitter] * n.current_gain_drift[action.receiver]
        stored = bank.waveforms if channel == "current" else bank.baseline
        if not np.array_equal(values.astype(np.float32), stored):
            raise ArithmeticError("float64 solver trace extraction disagrees with the saved bank")
        result[channel] = values
    return result


def band_resolution(scenario: Scenario, mesh_report: dict) -> dict:
    """Fundamental-mode main-lobe screen; not a mesh recommendation."""
    thicknesses = sorted({scenario.plate.thickness_m} | {scenario.plate.thickness_m - r.depth_m for r in scenario.recesses})
    rows = []
    for thickness in thicknesses:
        for frequency in sorted(set(scenario.acquisition.frequencies_hz)
                                | {f * (1 + 2 / scenario.acquisition.cycles) for f in scenario.acquisition.frequencies_hz}):
            for mode in ("A0", "S0"):
                try:
                    velocity = mode_velocity(frequency, thickness, mode, scenario.nuisance.material)
                    rows.append({**asdict(velocity), "wavelength_to_maximum_edge_ratio": velocity.wavelength_m / mesh_report["maximum_edge_m"]})
                except ValueError as exc:
                    rows.append({"mode": mode, "frequency_hz": frequency, "thickness_m": thickness, "unsupported": str(exc)})
    ratios = [row["wavelength_to_maximum_edge_ratio"] for row in rows if "wavelength_to_maximum_edge_ratio" in row]
    return {"rows": rows, "minimum_wavelength_to_edge_ratio": min(ratios) if ratios else None,
            "stock_thickness_to_maximum_z_span": scenario.plate.thickness_m / mesh_report["maximum_element_z_span_m"],
            "note": "Carrier and first-main-lobe upper zeros only; tails/higher modes and finite-plate arrivals are not validated. A geometric ratio is not proof of bending accuracy."}


def run_waveform_sweep(scenario: Scenario, settings: WaveformSweepSettings, output: Path,
                         guard: ResourceGuard | None = None, *, cache_dir: Path | None = None) -> dict:
    if scenario.nuisance.relative_noise_std or scenario.nuisance.absolute_noise_std_m:
        raise ValueError("turn off stochastic noise before a numerical refinement study")
    actions = action_catalog(scenario.acquisition.sensor_count, scenario.acquisition.frequencies_hz, scenario.acquisition.reciprocal)
    count = len({(a.transmitter, a.frequency_hz) for a in actions})
    requested_solves = count * (2 if scenario.recesses else 1) * 5
    if requested_solves > settings.max_total_solves:
        raise ValueError("refinement sweep exceeds its total forward-solve limit")
    records = len(actions)
    if records * scenario.acquisition.samples * 16 > settings.max_raw_trace_mib * 1024**2:
        raise MemoryError("float64 refinement traces exceed the per-case limit")
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("waveform sweep needs a new or empty output directory")
    output.mkdir(parents=True, exist_ok=True)
    cache = Path(cache_dir).resolve() if cache_dir is not None else (output / "groups").resolve()
    plan = {"kind": "predeclared_waveform_refinement", "scenario": asdict(scenario), "settings": asdict(settings),
            "cache_directory": str(cache),
            "requested_forward_solves": requested_solves, "ready_for_training_data": False,
            "source_sha256": {p.name: sha256_file(p) for p in sorted(Path(__file__).parent.glob("*.py"))}}
    plan["fingerprint"] = canonical_hash(plan)
    write_json(output / "refinement-plan.json", plan)
    started = time.perf_counter()
    report = {"kind": "finite_plate_waveform_refinement", "completed": False, "discretization_screen_passed": False,
              "ready_for_training_data": False, "plan_fingerprint": plan["fingerprint"], "cases": []}
    raw, banks = [], []
    try:
        minimums = [minimum_stable_substeps(scenario, replace(settings.forward, mesh_size_m=size), guard)
                    for size in settings.mesh_sizes_m]
        common_base = max(minimums)
        if common_base * settings.time_refinements[-1] * (scenario.acquisition.samples - 1) > settings.forward.max_internal_steps:
            raise ValueError("common refined time grid exceeds the internal-step limit before propagation")
        report["minimum_substeps_by_mesh"] = minimums
        report["common_base_substeps"] = common_base
        # The spatial axis keeps the finest time grid fixed on all meshes.
        # The temporal axis then uses the finest mesh at N, rN and r²N steps.
        definitions = [(size, settings.time_refinements[-1]) for size in settings.mesh_sizes_m]
        definitions += [(settings.mesh_sizes_m[-1], factor) for factor in settings.time_refinements[:2]]
        for index, (size, factor) in enumerate(definitions):
            if guard:
                guard.check()
            forward = replace(settings.forward, mesh_size_m=size, fixed_substeps_per_output=common_base * factor)
            bank, generated = generate_waveform_bank(scenario, forward, cache_dir=cache, guard=guard)
            traces = _raw_traces(scenario, bank, generated, cache)
            path = output / f"case-{index:02d}"
            path.mkdir()
            bank.save(path / "bank.npz")
            atomic_write(path / "solver-traces.npz", lambda stream: np.savez_compressed(stream, time_s=bank.time_s, **traces))
            write_json(path / "forward.json", generated)
            report["cases"].append({"directory": path.name, "mesh_size_m": size, "relative_time_refinement": factor,
                                     "dt_internal_s": generated["dt_internal_s"], "internal_steps": generated["internal_steps"],
                                     "mesh": generated["current_mesh"], "forward_solves": generated["forward_solves"],
                                     "cache_hits": generated["cache_hits"], "bank_sha256": sha256_file(path / "bank.npz"),
                                     "float64_traces_sha256": sha256_file(path / "solver-traces.npz")})
            raw.append(traces["current"])
            banks.append(bank)
        frequencies = [a.frequency_hz for a in banks[0].actions]
        def compare(a, b):
            return trace_discrepancies(raw[a], raw[b], banks[0].time_s, frequencies, rms_floor=settings.reference_rms_floor)
        spatial, temporal = [compare(0, 1), compare(1, 2)], [compare(3, 4), compare(4, 2)]
        def screen(rows, tolerance):
            return all(row["all_records_resolved"] for row in rows) and rows[-1]["maximum_relative_l2"] <= tolerance
        spatial_pass, temporal_pass = screen(spatial, settings.spatial_relative_l2_tolerance), screen(temporal, settings.temporal_relative_l2_tolerance)
        ratio = temporal[0]["global_difference_norm"] / temporal[1]["global_difference_norm"] if temporal[1]["global_difference_norm"] > 0 else None
        report.update({"spatial_comparisons": spatial, "temporal_comparisons": temporal,
                       "spatial_screen_passed": spatial_pass, "temporal_screen_passed": temporal_pass,
                       "temporal_change_ratio": ratio,
                       "observed_temporal_order": math.log(ratio, settings.time_refinements[1]) if ratio and ratio > 0 else None,
                       "band_resolution": band_resolution(scenario, report["cases"][2]["mesh"]),
                       "completed": True, "discretization_screen_passed": spatial_pass and temporal_pass,
                       "note": "Provisional discrepancy screens, not independent wave/PZT/physical validation. Raw traces can include rigid motion and overlapping reflected modes; no automatic data promotion."})
        if guard:
            guard.check(force=True)
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        report["wall_seconds"] = time.perf_counter() - started
        report["resources"] = guard.report() if guard else None
        report["fingerprint"] = canonical_hash(report)
        write_json(output / "waveform-report.json", report)
    return report


def plot_waveform_sweep(directory: Path, output: Path, records=(0, 12), guard=None, *, reference=False, directional=False) -> dict:
    """Plot preselected probe records without new solves or fitted alignment."""
    import json
    import zipfile
    from traxis.reporting import _figure_api, _new_figure, _save_figure
    directory, output = Path(directory).resolve(), Path(output)
    reference = reference or directional
    report_name = 'directional-report.json' if directional else ('refinement-report.json' if reference else 'waveform-report.json')
    path = directory / report_name
    if path.stat().st_size > 16 * 1024**2:
        raise MemoryError("waveform report exceeds 16 MiB")
    report = json.loads(path.read_text(encoding="utf-8-sig"))
    if canonical_hash({k: v for k, v in report.items() if k != "fingerprint"}) != report["fingerprint"] or not report["completed"]:
        raise ValueError("waveform report is incomplete or changed")
    plan_path = directory / "refinement-plan.json"
    if plan_path.stat().st_size > 1024**2:
        raise MemoryError("waveform plan exceeds 1 MiB")
    plan = json.loads(plan_path.read_text(encoding="utf-8-sig"))
    if canonical_hash({k: v for k, v in plan.items() if k != "fingerprint"}) != plan["fingerprint"] or plan["fingerprint"] != report["plan_fingerprint"]:
        raise ValueError("waveform plan changed")
    scenario = Scenario.from_dict(plan["scenario"])
    actions = action_catalog(scenario.acquisition.sensor_count, scenario.acquisition.frequencies_hz, scenario.acquisition.reciprocal)
    if reference:
        selected = tuple(Action(**row) for row in plan['actions'])
        if not selected or len(set(selected)) != len(selected) or any(a not in actions for a in selected):
            raise ValueError('reference plot actions disagree with the scenario catalog')
        actions = selected
    expected_cases = 7 if directional else 5
    if len(report['cases']) != expected_cases:
        raise ValueError(f'refinement figures require {expected_cases} completed cases')
    if not records or len(records) > 4 or any(type(i) is not int or not 0 <= i < len(actions) for i in records):
        raise ValueError("select 1–4 available probe records before reviewing errors")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("waveform figure output needs a new or empty directory")
    output.mkdir(parents=True, exist_ok=True)
    arrays = []
    for case in report["cases"]:
        if guard:
            guard.check()
        path = (directory / case["directory"] / "solver-traces.npz").resolve()
        if not path.is_relative_to(directory) or sha256_file(path) != case["float64_traces_sha256"]:
            raise ValueError("solver trace file changed or escaped the sweep directory")
        with zipfile.ZipFile(path) as archive:
            trace_limit_mib = 16 if reference else plan['settings']['max_raw_trace_mib']
            if sum(e.file_size for e in archive.infolist()) > trace_limit_mib * 1024**2 + 1024**2:
                raise MemoryError("solver trace archive exceeds its planned limit")
        with np.load(path, allow_pickle=False) as saved:
            traces, times = saved['displacement_m' if reference else 'current'].copy(), saved['time_s'].copy()
        if traces.shape != (len(actions), scenario.acquisition.samples) or times.shape != (scenario.acquisition.samples,) or not np.isfinite(traces).all():
            raise ValueError("solver trace shape/values disagree with the sweep plan")
        if not np.allclose(times, np.arange(scenario.acquisition.samples) * scenario.acquisition.sample_dt_s, rtol=1e-13, atol=1e-18):
            raise ValueError('solver trace time grid disagrees with the sweep plan')
        arrays.append(traces)
    api = _figure_api(output / ".matplotlib-cache")
    if directional:
        files = []
        axes_definitions = (('In-plane refinement', (0, 1, 2)), ('Thickness refinement', (3, 4, 2)), ('Time refinement', (5, 6, 2)))
        for differences in (False, True):
            figure = _new_figure(api, (14, 3.5 * len(records)))
            axes = figure.subplots(len(records), 3, squeeze=False)
            for row, index in enumerate(records):
                action = actions[index]
                for column, (title, levels) in enumerate(axes_definitions):
                    axis = axes[row, column]
                    for level in levels:
                        case = report['cases'][level]
                        label = f"{case['cells'][0]} x {case['cells'][1]} x {case['cells'][2]} cells" if column < 2 else f"dt={case['dt_internal_s'] * 1e9:g} ns"
                        values = arrays[level][index] - arrays[2][index] if differences else arrays[level][index]
                        axis.plot(times * 1e6, values * 1e9, label=label, linewidth=1)
                    axis.set(title=f"{title}\nTX {action.transmitter} to RX {action.receiver}, {action.frequency_hz/1000:g} kHz",
                             xlabel='Time (microseconds)', ylabel='Difference from common finest (nm)' if differences else 'Displacement (nm)')
                    axis.grid(alpha=.2)
                    axis.legend(fontsize=7)
            figure.suptitle(f"Directional {report.get('solver_label', 'P2')} intact-plate pilot - provisional spatial resolution", fontsize=13)
            figure.supxlabel('Float64 probes; fixed physical apertures; no fitted amplitude, phase or time shift. No PZT/physical validation.', fontsize=8)
            files.extend(_save_figure(figure, output, 'directional-differences' if differences else 'directional-waveforms', guard))
        result = {'kind': 'directional_refinement_figures', 'records': list(records), 'files': files,
                  'waveform_report_sha256': sha256_file(directory / report_name), 'ready_for_training_data': False,
                  'file_sha256': {name: sha256_file(output / name) for name in files}, 'resources': guard.report() if guard else None}
        write_json(output / 'figures.json', result)
        return result
    figure = _new_figure(api, (14, 3.5 * len(records)))
    axes = figure.subplots(len(records), 3, squeeze=False)
    for row, index in enumerate(records):
        action = actions[index]
        name = f"TX {action.transmitter} → RX {action.receiver}, {action.frequency_hz / 1000:g} kHz"
        for level in range(3):
            axes[row, 0].plot(times * 1e6, arrays[level][index] * 1e9,
                              label=f"h={report['cases'][level]['mesh_size_m'] * 1000:g} mm", linewidth=1)
        for level in range(2):
            axes[row, 1].plot(times * 1e6, (arrays[level][index] - arrays[2][index]) * 1e9,
                              label=f"h={report['cases'][level]['mesh_size_m'] * 1000:g} mm minus finest", linewidth=1)
        for level in (3, 4):
            axes[row, 2].plot(times * 1e6, (arrays[level][index] - arrays[2][index]) * 1e9,
                              label=f"dt={report['cases'][level]['dt_internal_s'] * 1e9:.3g} ns minus finest", linewidth=1)
        for axis, title in zip(axes[row], ("Mesh refinement", "Spatial difference", "Time-step difference")):
            axis.set(title=f"{title}\n{name}", xlabel="Time (μs)", ylabel="Displacement (nm)")
            axis.grid(alpha=.2)
            axis.legend(fontsize=7)
    figure.suptitle("Finite-plate mechanical pilot — no PZT/physical validation", fontsize=13)
    figure.supxlabel("Float64 probes; no amplitude normalization, time alignment or interpolation. Reflections and rigid motion remain in the traces.", fontsize=8)
    files = _save_figure(figure, output, "waveform-refinement", guard)
    result = {"kind": "waveform_refinement_figures", "records": list(records), "files": files,
              "waveform_report_sha256": sha256_file(directory / report_name),
              "file_sha256": {name: sha256_file(output / name) for name in files}, "ready_for_training_data": False,
              "resources": guard.report() if guard else None}
    write_json(output / "figures.json", result)
    return result
