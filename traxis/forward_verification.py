"""Monitored recipe/CAD/elastic-bank pilot; no network training."""

from pathlib import Path
import time

import numpy as np

from traxis.bank import WaveformBank
from traxis.cad_mesh import cad_mesh
from traxis.config import CircularRecess, Material, Plate
from traxis.dataset import BankDataset
from traxis.forward import ForwardSettings, generate_waveform_bank
from traxis.geometry import structured_mesh
from traxis.resources import ResourceGuard
from traxis.scenarios import AcquisitionSettings, NuisanceSettings, SamplingSettings, Scenario, sample_scenarios, save_scenario_plan
from traxis.solver_requests import export_solver_request
from traxis.storage import sha256_file, write_json


def verify_forward(output: Path, guard: ResourceGuard) -> dict:
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("forward pilot needs a new or empty output directory")
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    report = {"kind": "cad_to_elastic_bank_pilot", "passed": False, "ready_for_training_data": False,
              "note": "Geometry/data-pipeline checks only. No coupled PZT, guided-wave convergence or physical agreement claim.",
              "source_sha256": {p.name: sha256_file(p) for p in sorted(Path(__file__).parent.glob("*.py"))}}
    try:
        guard.check(force=True)
        plan = save_scenario_plan(output / "study-recipes.json", sample_scenarios(SamplingSettings(count=1000)))
        report["recipes"] = {"parents": plan["parent_count"], "split_counts": plan["split_counts"],
                             "stratum_counts": plan["stratum_counts"], "fingerprint": plan["fingerprint"], "forward_solves": 0}
        plate = Plate(0.04, 0.04, 0.003)
        recess = CircularRecess(0.02, 0.02, 0.008, 0.0015)
        old = structured_mesh(plate, (8, 8, 4), (recess,))
        coarse_error = abs(old.removed_volume_m3 / old.requested_removed_volume_m3 - 1)
        levels = []
        for size in (0.003, 0.002, 0.0015):
            _mesh, refinement = cad_mesh(plate, (recess,), size, guard=guard)
            levels.append(refinement)
        report["geometry_refinement"] = {"plate_m": [0.04, 0.04, 0.003], "radius_m": 0.008,
                                         "depth_m": 0.0015, "structured_volume_relative_error": coarse_error, "cad_levels": levels}
        if levels[-1]["removed_volume_relative_error"] >= 0.01 or levels[-1]["removed_volume_relative_error"] >= coarse_error:
            raise AssertionError("CAD geometric refinement did not reach the pilot's 1% volume criterion")
        acquisition = AcquisitionSettings(inset_m=0.006, patch_radius_m=0.003, frequencies_hz=(100000.,), samples=128, cycles=2)
        nuisance = NuisanceSettings(Material(), 1., (1.,) * 8, (1.,) * 8, relative_noise_std=0.)
        scenarios = [Scenario("elastic-intact", "elastic-parent-intact", "train", 2026, 44, plate, (), acquisition, nuisance),
                     Scenario("elastic-recess", "elastic-parent-recess", "test", 2026, 45, plate,
                              (CircularRecess(0.02, 0.02, 0.01, 0.0015),), acquisition, nuisance)]
        paths, banks, reports = [], [], []
        settings = ForwardSettings(mesh_backend="cad", mesh_size_m=0.003)
        for scenario in scenarios:
            bank, generated = generate_waveform_bank(scenario, settings, cache_dir=output / "groups", guard=guard)
            path = output / f"{scenario.specimen_id}.npz"
            bank.save(path)
            loaded = WaveformBank.load(path)
            np.testing.assert_array_equal(bank.waveforms, loaded.waveforms)
            paths.append(path)
            banks.append(bank)
            reports.append(generated)
        # Verify completed physics groups are replayed rather than recomputed.
        recovered, recovery = generate_waveform_bank(scenarios[1], settings, cache_dir=output / "groups", guard=guard)
        np.testing.assert_array_equal(recovered.waveforms, banks[1].waveforms)
        if recovery["forward_solves"] != 0:
            raise AssertionError("cached recovery repeated completed elastic solves")
        dataset = BankDataset.build(paths, output / "dataset.json", guard)
        request = export_solver_request(scenarios[1], output / "reference-request")
        intact_residual = float(np.linalg.norm(banks[0].waveforms - banks[0].baseline))
        defect_change = float(np.linalg.norm(banks[1].waveforms - banks[1].baseline) / np.linalg.norm(banks[1].baseline))
        report.update({"banks": reports, "cache_recovery": recovery, "dataset_fingerprint": dataset.fingerprint,
                       "reference_request_jobs": len(request["jobs"]), "intact_residual_norm_m": intact_residual,
                       "defect_relative_response_change": defect_change})
        report["passed"] = intact_residual == 0 and defect_change > 0.001 and recovery["forward_solves"] == 0
        guard.check(force=True)
    except Exception as exc:
        report["passed"] = False
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        report["wall_seconds"] = time.perf_counter() - started
        report["resources"] = guard.report()
        write_json(output / "forward-report.json", report)
    return report
