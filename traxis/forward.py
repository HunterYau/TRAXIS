"""Bounded three-dimensional elastic waveform banks for local pilot checks.

The local source/receiver are reciprocal mechanical patches, not piezoelectric
domains or electrode voltages. Every bank remains plumbing_only until an
independent reference/physical validation workflow establishes readiness.
"""

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import time
import zipfile

import numpy as np
import scipy

from traxis.acquisition import action_catalog
from traxis.bank import BankMetadata, WaveformBank
from traxis.cad_mesh import cad_mesh
from traxis.config import positive
from traxis.fem import assemble, simulate, surface_patch_with_report, tone_burst
from traxis.geometry import depth_map_mm, estimate_mesh, structured_mesh
from traxis.resources import ResourceGuard
from traxis.scenarios import Scenario
from traxis.storage import atomic_write, canonical_hash, sha256_file


@dataclass(frozen=True)
class ForwardSettings:
    mesh_backend: str = "cad"
    mesh_size_m: float = 0.003
    mesh_optimization: str = "default"
    mesh_algorithm: int = 1
    cells: tuple[int, int, int] = (8, 8, 4)
    max_assembly_mib: float = 256
    max_volume_error: float = 0.05
    max_internal_steps: int = 10000
    max_forward_solves: int = 32
    max_bank_mib: float = 64
    stability_fraction: float = 0.8
    time_step_refinement: int = 1
    fixed_substeps_per_output: int | None = None
    label_size: int = 512

    def __post_init__(self):
        object.__setattr__(self, "cells", tuple(self.cells))
        if self.mesh_backend not in ("cad", "structured"):
            raise ValueError("choose cad or structured pilot meshing")
        if self.mesh_optimization not in ("default", "Netgen"):
            raise ValueError("mesh optimization must be default or Netgen")
        if type(self.mesh_algorithm) is not int or self.mesh_algorithm not in (1, 4, 10):
            raise ValueError("choose Gmsh Delaunay (1), Frontal (4) or HXT (10)")
        estimate_mesh(self.cells)
        for key in ("mesh_size_m", "max_assembly_mib", "max_bank_mib", "max_volume_error"):
            positive(key, getattr(self, key))
        if not 0 < self.stability_fraction <= 0.9 or self.max_volume_error > 1:
            raise ValueError("invalid stability fraction or geometric-error limit")
        if type(self.max_internal_steps) is not int or not 1 <= self.max_internal_steps <= 50000:
            raise ValueError("internal step limit must be in [1,50000]")
        if type(self.max_forward_solves) is not int or not 1 <= self.max_forward_solves <= 128:
            raise ValueError("forward solve limit must be in [1,128]")
        if type(self.time_step_refinement) is not int or not 1 <= self.time_step_refinement <= 16:
            raise ValueError("time-step refinement must be an integer in [1,16]")
        if self.fixed_substeps_per_output is not None and (type(self.fixed_substeps_per_output) is not int
                or not 1 <= self.fixed_substeps_per_output <= 50000 or self.time_step_refinement != 1):
            raise ValueError("fixed substeps must lie in [1,50000]; choose fixed substeps or a refinement factor")
        if self.label_size not in (128, 256, 512):
            raise ValueError("label size must be 128, 256 or 512")


def _system(scenario: Scenario, settings: ForwardSettings, recesses, guard):
    if settings.mesh_backend == "cad":
        mesh, report = cad_mesh(scenario.plate, recesses, settings.mesh_size_m,
                                max_assembly_mib=settings.max_assembly_mib,
                                seed=scenario.geometry_seed, optimization=settings.mesh_optimization,
                                algorithm=settings.mesh_algorithm, guard=guard)
    else:
        mesh = structured_mesh(scenario.plate, settings.cells, recesses,
                                max_assembly_bytes=int(settings.max_assembly_mib * 1024**2))
        report = {"backend": "structured_centroid_removal", "cells": list(settings.cells),
                  "nodes": len(mesh.points_m), "tetrahedra": len(mesh.tetrahedra)}
    requested, removed = mesh.requested_removed_volume_m3, mesh.removed_volume_m3
    error = abs(removed / requested - 1) if requested else 0.
    report.update({"requested_removed_volume_m3": requested, "meshed_removed_volume_m3": removed,
                   "removed_volume_relative_error": error, "maximum_allowed_volume_error": settings.max_volume_error})
    if error > settings.max_volume_error:
        raise ValueError(f"meshed recess volume error {error:.1%} exceeds {settings.max_volume_error:.1%}")
    points = mesh.points_m[mesh.tetrahedra]
    edge_limits = [(float(np.linalg.norm(points[:, i] - points[:, j], axis=1).min()),
                    float(np.linalg.norm(points[:, i] - points[:, j], axis=1).max()))
                   for i, j in ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))]
    report.update({"minimum_edge_m": min(pair[0] for pair in edge_limits),
                   "maximum_edge_m": max(pair[1] for pair in edge_limits),
                   "maximum_element_z_span_m": float(np.ptp(points[:, :, 2], axis=1).max()),
                   "degrees_of_freedom": len(mesh.points_m) * 3})
    return assemble(mesh, scenario.nuisance.material, guard), report


def _trace_hash(values):
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def _cached_group(cache: Path | None, key: str, expected_shape, compute):
    path = cache / f"{key}.npz" if cache else None
    if path and path.exists():
        with zipfile.ZipFile(path) as archive:
            names = [e.filename for e in archive.infolist()]
            if len(names) != 2 or set(names) != {"displacement_m.npy", "metadata.npy"} or sum(e.file_size for e in archive.infolist()) > 32 * 1024**2:
                raise ValueError("invalid or oversized forward group cache")
        with np.load(path, allow_pickle=False) as saved:
            values = saved["displacement_m"]
            metadata = json.loads(str(saved["metadata"].item()))
        if metadata["key"] != key or values.shape != expected_shape or values.dtype != np.float64 or not np.isfinite(values).all() or metadata["sha256"] != _trace_hash(values):
            raise ValueError("forward group cache integrity check failed")
        return values, True
    values = np.asarray(compute(), dtype=np.float64)
    if values.shape != expected_shape or not np.isfinite(values).all():
        raise ValueError("forward solver returned invalid probe traces")
    if path:
        metadata = np.array(json.dumps({"key": key, "sha256": _trace_hash(values)}))
        atomic_write(path, lambda stream: np.savez_compressed(stream, displacement_m=values, metadata=metadata))
    return values, False


def load_cached_group(cache: Path, key: str, expected_shape) -> np.ndarray:
    """Read checked float64 solver probes for numerical error studies."""
    if len(key) != 64 or any(c not in "0123456789abcdef" for c in key):
        raise ValueError("invalid forward group key")
    def missing():
        raise FileNotFoundError("requested forward group was not generated")
    return _cached_group(Path(cache), key, expected_shape, missing)[0]


def minimum_stable_substeps(scenario: Scenario, settings: ForwardSettings,
                             guard: ResourceGuard | None = None) -> int:
    """Preflight the mechanical operators without propagating or storing fields."""
    if scenario.nuisance.support != "free":
        raise ValueError("local pilot implements traction-free supports only")
    healthy, _ = _system(scenario, settings, (), guard)
    bound = healthy.stable_dt_s
    if scenario.recesses:
        damaged, _ = _system(scenario, settings, scenario.recesses, guard)
        bound = min(bound, damaged.stable_dt_s)
    return int(math.ceil(scenario.acquisition.sample_dt_s / (bound * settings.stability_fraction)))


def generate_waveform_bank(scenario: Scenario, settings: ForwardSettings = ForwardSettings(),
                            *, cache_dir: Path | None = None, guard: ResourceGuard | None = None) -> tuple[WaveformBank, dict]:
    if scenario.nuisance.support != "free":
        raise ValueError("local pilot implements traction-free supports only; use an external solver for other supports")
    if guard:
        guard.check(force=True)
    started = time.perf_counter()
    acquisition = scenario.acquisition
    actions = action_catalog(acquisition.sensor_count, acquisition.frequencies_hz, acquisition.reciprocal)
    groups = sorted({(a.transmitter, a.frequency_hz) for a in actions})
    estimated_solves = len(groups) * (2 if scenario.recesses else 1)
    if estimated_solves > settings.max_forward_solves:
        raise ValueError("requested transmitter-frequency solves exceed the local pilot limit")
    bank_bytes = len(actions) * acquisition.samples * 4 * 2
    if bank_bytes > settings.max_bank_mib * 1024**2:
        raise MemoryError("requested waveform bank exceeds pilot storage limit")
    healthy, healthy_mesh = _system(scenario, settings, (), guard)
    damaged, damaged_mesh = _system(scenario, settings, scenario.recesses, guard) if scenario.recesses else (healthy, healthy_mesh)
    stable = min(healthy.stable_dt_s, damaged.stable_dt_s) * settings.stability_fraction
    minimum_substeps = int(math.ceil(acquisition.sample_dt_s / stable))
    substeps = settings.fixed_substeps_per_output or minimum_substeps * settings.time_step_refinement
    if substeps < minimum_substeps:
        raise ValueError("requested fixed time step exceeds the conservative stability bound")
    dt = acquisition.sample_dt_s / substeps
    steps = (acquisition.samples - 1) * substeps
    if steps > settings.max_internal_steps:
        raise ValueError("sample-aligned internal steps exceed the configured pilot limit")
    sensors = acquisition.sensors(scenario.plate)
    systems = {"intact": healthy, "current": damaged}
    patches, patch_reports = {}, {}
    for kind, system in systems.items():
        integrated = [surface_patch_with_report(system, tuple(xy), acquisition.patch_radius_m) for xy in sensors]
        patches[kind] = np.stack([vector for vector, _report in integrated])
        patch_reports[kind] = [report for _vector, report in integrated]
        if any(report["full_disk_area_relative_error"] > 1e-8 or report["centroid_error_m"] > 1e-8 * acquisition.patch_radius_m
               for report in patch_reports[kind]):
            raise ArithmeticError("circular source/probe footprint integration failed its area/centroid check")
    source = {p.name: sha256_file(p) for p in [Path(__file__), Path(__file__).with_name("fem.py"),
              Path(__file__).with_name("geometry.py"), Path(__file__).with_name("cad_mesh.py"), Path(__file__).with_name("patches.py")]}
    cache_dir = Path(cache_dir) if cache_dir else None
    waveforms = np.empty((len(actions), acquisition.samples), dtype=np.float32)
    baselines = np.empty_like(waveforms)
    solves, hits, group_records = 0, 0, []
    for tx, frequency in groups:
        raw = {}
        for kind, system in systems.items():
            if kind == "current" and not scenario.recesses:
                raw[kind] = raw["intact"]
                continue
            if guard:
                guard.check()
            recipe = {"kind": kind, "plate": asdict(scenario.plate), "material": asdict(scenario.nuisance.material),
                      "recesses": [asdict(r) for r in scenario.recesses] if kind == "current" else [],
                      "sensors_xy_m": sensors.tolist(), "patch_radius_m": acquisition.patch_radius_m,
                      "cycles": acquisition.cycles, "dt_s": dt, "substeps": substeps, "steps": steps,
                      "transmitter": tx, "frequency_hz": frequency,
                      "mesh": healthy_mesh if kind == "intact" else damaged_mesh,
                      "source_sha256": source, "numpy": np.__version__, "scipy": scipy.__version__}
            key = canonical_hash(recipe)
            def compute():
                # Solve unit force once; linear coupling/drive gains are applied
                # below, without pretending they are piezoelectric domains.
                load = patches[kind][tx]
                run = simulate(system, steps, dt, patches[kind],
                               force=lambda t: load * tone_burst(t, frequency, acquisition.cycles), guard=guard)
                return run.displacement_m[::substeps].T
            values, cached = _cached_group(cache_dir, key, (acquisition.sensor_count, acquisition.samples), compute)
            raw[kind] = values
            hits += int(cached)
            solves += int(not cached)
            group_records.append({"transmitter": tx, "frequency_hz": frequency, "kind": kind, "key": key, "cache_hit": cached})
        for i, action in enumerate(actions):
            if action.transmitter != tx or action.frequency_hz != frequency:
                continue
            rx = action.receiver
            nuisance = scenario.nuisance
            gain = nuisance.peak_force_n * nuisance.sensor_gains[tx] * nuisance.sensor_gains[rx]
            baseline = raw["intact"][rx] * gain
            current = raw["current"][rx] * gain * nuisance.current_gain_drift[tx] * nuisance.current_gain_drift[rx]
            sigma = math.hypot(nuisance.absolute_noise_std_m, nuisance.relative_noise_std * float(np.sqrt(np.mean(baseline**2))))
            for destination, signal, kind in ((baselines, baseline, "baseline"), (waveforms, current, "current")):
                seed = int(canonical_hash([scenario.nuisance_seed, asdict(action), kind])[:16], 16)
                destination[i] = signal + np.random.default_rng(seed).normal(0, sigma, acquisition.samples)
    baseline_id = "paired-intact-" + canonical_hash({"plate": asdict(scenario.plate), "material": asdict(scenario.nuisance.material),
                                                   "acquisition": asdict(acquisition), "gains": scenario.nuisance.sensor_gains})[:16]
    metadata = BankMetadata(scenario.specimen_id, scenario.parent_specimen_id, scenario.split, "simulation", "displacement_m",
                            baseline_id, "local linear-tetrahedral elastic FEM", "plumbing_only", scenario.nuisance_seed,
                            {"scenario": asdict(scenario), "scenario_fingerprint": scenario.fingerprint,
                             "forward": asdict(settings), "intact_mesh": healthy_mesh, "current_mesh": damaged_mesh,
                             "dt_internal_s": dt, "substeps_per_output": substeps, "internal_steps": steps,
                             "mechanical_patches": patch_reports,
                             "source_sha256": source, "source": "unit-total-force normal mechanical disk patch",
                             "baseline": "matched intact plate with same material/coupling; not a separate physical control",
                             "approximations": ["no PZT/adhesive/electrical domains", "scalar coupling/drift gains only",
                                                "undamped free edges", "guided-wave convergence not established"]})
    bank = WaveformBank(metadata, scenario.plate, sensors, actions, np.arange(acquisition.samples) * acquisition.sample_dt_s,
                        waveforms, baselines, depth_map_mm(scenario.plate, scenario.recesses, (settings.label_size,) * 2, 2))
    report = {"kind": "local_elastic_bank_pilot", "scenario_fingerprint": scenario.fingerprint,
              "records": len(actions), "samples": acquisition.samples, "forward_solves": solves, "cache_hits": hits,
              "groups": group_records, "intact_mesh": healthy_mesh, "current_mesh": damaged_mesh,
              "dt_internal_s": dt, "substeps_per_output": substeps, "internal_steps": steps,
              "mechanical_patches": patch_reports,
              "wall_seconds": time.perf_counter() - started, "ready_for_training_data": False,
              "validation_status": bank.metadata.validation_status}
    if guard:
        guard.check(force=True)
    return bank, report
