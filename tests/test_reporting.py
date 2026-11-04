import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from tests.test_dataset import fixture_dataset
from traxis.reporting import create_report, load_case, load_evaluation, _surface_grid


HAS_DEPS = all(importlib.util.find_spec(name) is not None for name in ("torch", "matplotlib"))


@unittest.skipUnless(HAS_DEPS, "optional CPU PyTorch and Matplotlib are required")
class ReportingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from traxis.das_baseline import fit_das_baseline
        from traxis.evaluation import freeze_evaluation_plan, run_frozen_evaluation
        from traxis.imaging import DASSettings
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        cls.dataset = fixture_dataset(cls.root, splits=("train", "train", "validation", "test"))
        artifact = cls.root / "das.json"
        fit_das_baseline(cls.dataset, DASSettings({100000.: 2500.}, 25e-6, (128, 128)), (2,)).save(artifact)
        cls.plan = cls.root / "plan.json"
        freeze_evaluation_plan(cls.plan, cls.dataset, [artifact], split="test", budgets=(2, 4))
        cls.evaluation = cls.root / "evaluation"
        cls.result = run_frozen_evaluation(cls.plan, cls.dataset, cls.evaluation)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_saved_maps_match_metrics_and_physical_geometry(self):
        report = load_evaluation(self.evaluation)
        plate, maps = load_case(self.evaluation, report["episodes"][0], 0, 2)
        self.assertEqual(plate.width_m, .5)
        self.assertEqual(plate.thickness_m, .003)
        self.assertEqual(maps["prediction"].shape, (128, 128))
        self.assertEqual(maps["truth"].shape, (512, 512))
        self.assertEqual(report["predictors"]["model-0"]["neural_parameters"], 0)

    def test_headless_report_exports_figures_units_counts_and_index(self):
        from traxis.cli import main
        from unittest.mock import patch
        output = self.root / "report"
        with patch("sys.argv", ["traxis", "report", "--evaluations", str(self.evaluation), "--output", str(output),
                                "--labels", "DAS fixture", "--case-records", "2", "--max-seconds", "60"]):
            main()
        result = json.loads((output / "report.json").read_text())
        self.assertTrue(result["passed"])
        self.assertTrue(result["debug_only"])
        self.assertEqual(len([p for p in result["files"] if p.endswith(".png")]), 3)
        self.assertIn("DEBUG FIXTURES", (output / "index.html").read_text(encoding="utf-8"))
        table = (output / "summary.csv").read_text()
        self.assertIn("independent_parents", table)
        self.assertIn("defect_region_mae_mm", table)
        self.assertIn("inference_calls_cumulative", (output / "computation.csv").read_text())
        with self.assertRaises(FileExistsError):
            create_report([self.evaluation], output)

    def test_changed_reports_maps_and_unmatched_protocols_fail(self):
        import shutil
        from traxis.reporting import _matched_runs
        report = load_evaluation(self.evaluation)
        changed = json.loads(json.dumps(report))
        changed["plan"]["initial_actions"] = [0]
        with self.assertRaisesRegex(ValueError, "same dataset"):
            _matched_runs([report, changed])
        changed_dir = self.root / "changed"
        shutil.copytree(self.evaluation, changed_dir)
        archive = changed_dir / "case-00000" / "maps.npz"
        archive.write_bytes(archive.read_bytes() + b"changed")
        with self.assertRaisesRegex(ValueError, "checksum"):
            load_case(changed_dir, report["episodes"][0], 0, 2)
        changed["fingerprint"] = "changed"
        (changed_dir / "evaluation.json").write_text(json.dumps(changed))
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            load_evaluation(changed_dir)

    def test_map_metric_disagreement_is_detected_even_with_updated_checksum(self):
        from traxis.storage import sha256_file
        directory = self.root / "inconsistent" / "case-00000"
        directory.mkdir(parents=True)
        episode = json.loads(json.dumps(self.result["episodes"][0]))
        # A corrupt exporter could hash internally consistent bytes but report
        # errors from a different prediction. Recompute errors to catch it.
        np.savez_compressed(directory / "maps.npz", truth_mm=np.zeros((512, 512)), depth_2_mm=np.ones((128, 128)))
        episode["maps_sha256"] = sha256_file(directory / "maps.npz")
        (directory / "metrics.json").write_text(json.dumps(episode))
        with self.assertRaisesRegex(ValueError, "reproduce"):
            load_case(directory.parent, episode, 0, 2)

    def test_surface_display_preserves_mean_and_x_y_asymmetry(self):
        grid = np.zeros((512, 256))
        grid[256:, :128] = 2
        averaged = _surface_grid(grid)
        self.assertEqual(averaged.shape, (64, 64))
        self.assertEqual(averaged.mean(), grid.mean())
        self.assertTrue(np.all(averaged[32:, :32] == 2))
        self.assertTrue(np.all(averaged[:32] == 0))

    def test_reporting_does_not_import_optional_torch_runtime(self):
        import subprocess
        import sys
        script = ("import sys; from pathlib import Path; from traxis.reporting import create_report; "
                  "create_report([Path(sys.argv[1])], Path(sys.argv[2]), case_count=0); "
                  "assert 'torch' not in sys.modules")
        subprocess.run([sys.executable, "-c", script, str(self.evaluation), str(self.root / "plots-only")],
                       check=True, capture_output=True, text=True, timeout=30)
