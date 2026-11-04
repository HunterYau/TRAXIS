"""Compile one explicit external-solver task without launching a solver."""

from dataclasses import asdict
from pathlib import Path

import numpy as np

from traxis.acquisition import action_catalog
from traxis.bank import BankMetadata
from traxis.geometry import depth_map_mm
from traxis.scenarios import Scenario
from traxis.storage import atomic_write, sha256_file, write_json


def export_solver_request(scenario: Scenario, directory: Path) -> dict:
    """Write SI geometry, label/time anchors and a mapped scalar CSV template.

    This exports a mechanical reference task. PZT voltage simulation needs
    additional measured piezoelectric/adhesive/electrical inputs, left explicit
    in the task instead of being invented or replaced by displacement units.
    """
    directory = Path(directory)
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError("solver request needs a new or empty directory")
    directory.mkdir(parents=True, exist_ok=True)
    acq = scenario.acquisition
    actions = action_catalog(acq.sensor_count, acq.frequencies_hz, acq.reciprocal)
    depth = depth_map_mm(scenario.plate, scenario.recesses, (512, 512), 2)
    times = np.arange(acq.samples) * acq.sample_dt_s
    atomic_write(directory / "depth-mm.npy", lambda stream: np.save(stream, depth, allow_pickle=False))
    atomic_write(directory / "sample-time-s.npy", lambda stream: np.save(stream, times, allow_pickle=False))
    records, groups = [], []
    for action in actions:
        suffix = f"tx-{action.transmitter}-rx-{action.receiver}-f-{action.frequency_hz:g}.csv"
        records.append({"action": asdict(action), "waveform_csv": f"current/{suffix}", "baseline_csv": f"intact/{suffix}"})
    for state in ("intact", "current"):
        for tx, frequency in sorted({(a.transmitter, a.frequency_hz) for a in actions}):
            receivers = [a.receiver for a in actions if a.transmitter == tx and a.frequency_hz == frequency]
            groups.append({"state": state, "transmitter": tx, "frequency_hz": frequency,
                           "eligible_receivers": receivers, "record_all_receivers_together": True,
                           "intact_geometry_reusable": not bool(scenario.recesses)})
    geometry = ["// Coordinates are meters; set downstream import units explicitly.", 'SetFactory("OpenCASCADE");',
                f"Box(1) = {{0,0,0,{scenario.plate.width_m:.17g},{scenario.plate.height_m:.17g},{scenario.plate.thickness_m:.17g}}};"]
    for index, r in enumerate(scenario.recesses, 2):
        geometry.append(f"Cylinder({index}) = {{{r.x_m:.17g},{r.y_m:.17g},{-scenario.plate.thickness_m:.17g},0,0,{scenario.plate.thickness_m + r.depth_m:.17g},{r.radius_m:.17g}}};")
    if scenario.recesses:
        tools = ",".join(str(i) for i in range(2, len(scenario.recesses) + 2))
        geometry.append(f"v() = BooleanDifference{{ Volume{{1}}; Delete; }}{{ Volume{{{tools}}}; Delete; }};")
        geometry.append('Physical Volume("plate") = {v()};')
    else:
        geometry.append('Physical Volume("plate") = {1};')
    atomic_write(directory / "geometry.geo", lambda stream: stream.write(("\n".join(geometry) + "\n").encode()))
    metadata = BankMetadata(scenario.specimen_id, scenario.parent_specimen_id, scenario.split, "simulation", "displacement_m",
                            "paired-intact-" + scenario.parent_specimen_id, "external solver export required", "plumbing_only", scenario.nuisance_seed,
                            {"scenario_recipe": asdict(scenario), "scenario_fingerprint": scenario.fingerprint,
                             "nuisance_application_status": "unverified external outputs; importer does not apply gains/noise",
                             "reference_validation_status": "unsolved; verify convergence and control agreement separately"})
    template = {"schema_version": 1, "metadata": asdict(metadata), "plate": asdict(scenario.plate),
                "sensors_xy_m": acq.sensors(scenario.plate).tolist(), "records": records,
                "depth_mm_npy": "depth-mm.npy", "expected_label_sha256": sha256_file(directory / "depth-mm.npy"),
                "expected_time_s_npy": "sample-time-s.npy", "expected_time_sha256": sha256_file(directory / "sample-time-s.npy")}
    write_json(directory / "import-manifest.json", template)
    task = {"schema_version": 1, "kind": "unsolved_external_mechanical_reference_task", "scenario": asdict(scenario),
            "scenario_fingerprint": scenario.fingerprint, "units": {"coordinates": "m", "time": "s", "depth_label": "mm", "force": "N", "probe": "m"},
            "source": {"type": "normal top-surface finite-area mechanical patch", "radius_m": acq.patch_radius_m,
                       "waveform": "Hann-windowed sinusoid", "cycles": acq.cycles, "peak_force_n": scenario.nuisance.peak_force_n},
            "probe": "area-averaged normal displacement at the receiving patch",
            "jobs": groups, "record_count": len(actions), "csv_header": "time_s,value",
            "import_manifest_sha256": sha256_file(directory / "import-manifest.json"),
            "ready_for_training_data": False,
            "required_before_voltage_model": ["PZT geometry/material/poling", "adhesive geometry/properties", "electrodes and electrical drive/loading", "receiver circuit/observable"],
            "required_before_research_dataset": ["appropriate band/mesh/time convergence", "intact physical agreement", "support/coupling validation", "measured solver cost"],
            "note": "No solver has run. No fake/zero CSV traces are supplied. Gmsh GEO coordinates are unitless numbers assigned SI meters by this recipe."}
    write_json(directory / "solver-task.json", task)
    return task
