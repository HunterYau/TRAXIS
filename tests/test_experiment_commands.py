import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tests.test_dataset import fixture_dataset

HAS_TORCH = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(HAS_TORCH, "optional CPU PyTorch is required")
class ExperimentCommandTests(unittest.TestCase):
    def invoke(self, arguments):
        from traxis.cli import main
        captured = io.StringIO()
        with patch("sys.argv", ["traxis", *map(str, arguments)]), contextlib.redirect_stdout(captured):
            main()
        return json.loads(captured.getvalue())

    def test_cli_train_resume_calibrate_freeze_and_replay(self):
        import torch
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = fixture_dataset(root, ("train", "train", "validation", "test"))
            spec = {"model": {"architecture": "direct", "width": 4, "embedding": 16},
                    "training": {"max_steps": 3, "min_records": 2, "max_records": 2}}
            config = root / "config.json"
            config.write_text(json.dumps(spec))
            out = root / "run"
            common = ["--dataset", root / "dataset.json", "--output", out, "--steps", 1]
            first = self.invoke(["train", *common, "--config", config])
            self.assertEqual(first["updates_total"], 1)
            resumed = self.invoke(["train", *common, "--resume", out / "latest.pt"])
            self.assertEqual(resumed["updates_total"], 2)
            settings = root / "das-settings.json"
            settings.write_text(json.dumps({"group_velocity_m_s": {"100000": 2500}, "time_offset_s": 25e-6}))
            baseline = root / "baseline.json"
            self.invoke(["calibrate-das", "--dataset", root / "dataset.json", "--settings", settings,
                         "--budgets", 2, "--output", baseline])
            plan = root / "plan.json"
            self.invoke(["freeze-evaluation", "--dataset", root / "dataset.json", "--models", resumed["model"], baseline,
                         "--split", "test", "--budgets", 2, 4, "--random-seeds", 5, "--output", plan])
            result = self.invoke(["evaluate", "--dataset", root / "dataset.json", "--plan", plan,
                                  "--output", root / "evaluation"])
            self.assertEqual(result["episodes"], 4)
            self.assertTrue(result["debug_only"])

    def test_cli_index_preserves_fingerprint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = fixture_dataset(root)
            indexed = self.invoke(["index-data", "--banks-dir", root, "--output", root / "indexed.json"])
            self.assertEqual(indexed["dataset_fingerprint"], dataset.fingerprint)
