from dataclasses import asdict
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from tests.test_forward import local_scenario
from traxis.importers import import_trace_bundle
from traxis.solver_requests import export_solver_request


class SolverRequestTests(unittest.TestCase):
    def test_exported_trace_mapping_and_label_time_anchors(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scenario = local_scenario()
            task = export_solver_request(scenario, root)
            self.assertEqual(task["record_count"], 28)
            self.assertEqual(len(task["jobs"]), 14)
            self.assertFalse(task["ready_for_training_data"])
            spec = json.loads((root / "import-manifest.json").read_text())
            times = np.load(root / "sample-time-s.npy")
            # Format fixture only, not a claimed reference-solver response.
            for record in spec["records"]:
                for name in ("waveform_csv", "baseline_csv"):
                    path = root / record[name]
                    path.parent.mkdir(parents=True, exist_ok=True)
                    np.savetxt(path, np.column_stack([times, np.sin(2 * np.pi * 100000 * times) * 1e-9]),
                               delimiter=",", header="time_s,value", comments="")
            bank = import_trace_bundle(root / "import-manifest.json")
            self.assertEqual(bank.metadata.specimen_id, scenario.specimen_id)
            self.assertEqual(bank.metadata.validation_status, "plumbing_only")
            self.assertEqual(bank.waveforms.shape, (28, 64))
            np.testing.assert_array_equal(bank.depth_mm, np.load(root / "depth-mm.npy"))
            trace = root / spec["records"][0]["waveform_csv"]
            original_trace = trace.read_bytes()
            np.savetxt(trace, np.column_stack([times + 1e-6, np.zeros_like(times)]), delimiter=",", header="time_s,value", comments="")
            with self.assertRaisesRegex(ValueError, "requested sample-time anchor"):
                import_trace_bundle(root / "import-manifest.json")
            trace.write_bytes(original_trace)
            original = (root / "depth-mm.npy").read_bytes()
            np.save(root / "depth-mm.npy", np.zeros((512, 512)))
            with self.assertRaisesRegex(ValueError, "geometry anchor"):
                import_trace_bundle(root / "import-manifest.json")
            (root / "depth-mm.npy").write_bytes(original)
            np.save(root / "sample-time-s.npy", times + 1e-6)
            with self.assertRaisesRegex(ValueError, "time anchor fingerprint"):
                import_trace_bundle(root / "import-manifest.json")

    @unittest.skipUnless(importlib.util.find_spec("gmsh"), "optional Gmsh verifies the exported CAD script")
    def test_exported_geo_parses_to_the_requested_cad_volume(self):
        import gmsh
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scenario = local_scenario()
            export_solver_request(scenario, root)
            gmsh.initialize([], readConfigFiles=False, run=False)
            try:
                gmsh.option.setNumber("General.Terminal", 0)
                gmsh.open(str(root / "geometry.geo"))
                volume = sum(gmsh.model.occ.getMass(dim, tag) for dim, tag in gmsh.model.getEntities(3))
                expected = 0.04 * 0.04 * 0.003 - scenario.recesses[0].volume_m3
                self.assertAlmostEqual(volume / expected, 1, places=10)
            finally:
                gmsh.finalize()
