from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from tests.fixtures import analytic_bank
from traxis.importers import import_trace_bundle


class ImportTests(unittest.TestCase):
    def make_bundle(self, directory):
        bank = analytic_bank()
        directory = Path(directory)
        np.save(directory / "label.npy", bank.depth_mm)
        records = []
        for index in range(2):
            for prefix, signal in (("current", bank.waveforms), ("baseline", bank.baseline)):
                np.savetxt(directory / f"{prefix}-{index}.csv", np.column_stack([bank.time_s, signal[index]]),
                           delimiter=",", header="time_s,value", comments="")
            records.append({"action": asdict(bank.actions[index]), "waveform_csv": f"current-{index}.csv",
                            "baseline_csv": f"baseline-{index}.csv"})
        manifest = {"schema_version": 1, "metadata": asdict(bank.metadata), "plate": asdict(bank.plate),
                    "sensors_xy_m": bank.sensors_xy_m.tolist(), "records": records, "depth_mm_npy": "label.npy"}
        path = directory / "manifest.json"
        path.write_text(json.dumps(manifest), encoding="utf-8")
        return bank, path, manifest

    def test_import_roundtrip_and_file_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            bank, path, _ = self.make_bundle(directory)
            imported = import_trace_bundle(path)
            np.testing.assert_array_equal(imported.waveforms, bank.waveforms[:2])
            np.testing.assert_array_equal(imported.depth_mm, bank.depth_mm)
            self.assertEqual(len(imported.metadata.settings["import_source_sha256"]), 5)
            self.assertIn("import_manifest_sha256", imported.metadata.settings)

    def test_time_mismatch_wrong_header_and_byte_limit_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            bank, path, _ = self.make_bundle(directory)
            with self.assertRaises(MemoryError):
                import_trace_bundle(path, max_input_mib=0.00001)
            target = Path(directory) / "baseline-1.csv"
            original = target.read_text()
            target.write_text(original.replace("time_s,value", "time_ms,value"))
            with self.assertRaisesRegex(ValueError, "header"):
                import_trace_bundle(path)
            np.savetxt(target, np.column_stack([bank.time_s + 1e-5, bank.baseline[1]]),
                       delimiter=",", header="time_s,value", comments="")
            with self.assertRaisesRegex(ValueError, "time grid"):
                import_trace_bundle(path)

    def test_bundle_paths_cannot_escape_manifest_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            _, path, manifest = self.make_bundle(directory)
            manifest["depth_mm_npy"] = "../outside.npy"
            path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "within"):
                import_trace_bundle(path)
