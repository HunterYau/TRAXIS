import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tests.test_dataset import fixture_dataset
from traxis.resources import ResourceLimitError
from traxis.storage import sha256_file

HAS_TORCH = importlib.util.find_spec("torch") is not None
if HAS_TORCH:
    import torch
    from traxis.studies import advance_study, create_study, freeze_study_comparisons, load_study, study_lock
    from traxis.training import TrainingSession


def small_spec(steps=1):
    return {"matrix": {"architectures": ["direct", "das_unet"], "output_sizes": [128],
                       "extra_decoder_blocks": [0], "seeds": [101, 202], "subset_seeds": [37, 53]},
            "model": {"width": 4, "embedding": 16},
            "training": {"mode": "debug", "max_steps": steps, "min_records": 2, "max_records": 4,
                         "validation_records": 4},
            "das": {"group_velocity_m_s": {"100000": 2500.}, "time_offset_s": 25e-6},
            "baseline": {"budgets": [2, 4]}}


@unittest.skipUnless(HAS_TORCH, "optional CPU PyTorch is required")
class StudyTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.dataset = fixture_dataset(self.root, ("train", "train", "validation", "uncertainty_calibration", "test"))
        self.manifest = self.root / "dataset.json"

    def test_full_factorial_plan_reads_no_reserved_or_test_labels(self):
        from traxis.dataset import BankDataset
        read = BankDataset.read
        roles = []
        def audit(dataset, entry):
            roles.append(entry.split)
            self.assertIn(entry.split, ("train", "validation"))
            return read(dataset, entry)
        specification = json.loads(Path("configs/study-debug.json").read_text())
        with patch.object(BankDataset, "read", audit):
            plan = create_study(self.manifest, specification, self.root / "study")
        self.assertEqual(len(plan["jobs"]), 20)
        self.assertEqual(len(plan["controlled_pairs"]), 24)
        self.assertTrue(roles)
        self.assertEqual({p["factor"] for p in plan["controlled_pairs"]}, {"architecture", "output_size", "extra_decoder_blocks"})
        self.assertEqual(plan["reference_shape"], [512, 512])
        self.assertFalse((self.root / "study" / "jobs").exists())
        with self.assertRaisesRegex(ValueError, "complete every"):
            freeze_study_comparisons(self.root / "study" / "study.json", self.root / "premature")

    def test_validation_intact_control_cannot_replace_a_training_control(self):
        folder = self.root / "missing-training-control"
        folder.mkdir()
        fixture_dataset(folder)  # Damaged train, intact validation, damaged test.
        with self.assertRaisesRegex(ValueError, "both intact and damaged training"):
            create_study(folder / "dataset.json", small_spec(), self.root / "invalid-study")

    def test_bounded_progress_exact_recovery_completed_idempotence_and_frozen_ensembles(self):
        create_study(self.manifest, small_spec(2), self.root / "study")
        plan_path = self.root / "study" / "study.json"
        first = advance_study(plan_path)
        self.assertEqual(first["new_updates"], 1)
        self.assertEqual(first["baseline_fits"], 0)
        self.assertFalse(first["jobs"][0]["completed"])
        job_id = first["jobs"][0]["id"]
        second = advance_study(plan_path, job_ids=(job_id,))
        self.assertEqual(second["new_updates"], 1)
        artifact = self.root / "study" / "jobs" / job_id / "selected.pt"
        digest = sha256_file(artifact)
        repeated = advance_study(plan_path, job_ids=(job_id,))
        self.assertEqual(repeated["new_updates"], 0)
        self.assertEqual(sha256_file(artifact), digest)
        rest = advance_study(plan_path, max_new_updates=6, baseline_fits=2)
        self.assertEqual(rest["new_updates"], 6)
        self.assertEqual(rest["baseline_fits"], 2)
        result = freeze_study_comparisons(plan_path, self.root / "comparisons", budgets=(2, 4), random_seeds=(17,))
        self.assertEqual(len(result["predictors"]), 6)
        self.assertEqual(len(result["ensembles"]), 2)
        states = [json.loads(path.read_text()) for path in (self.root / "study" / "jobs").glob("*/job.json")]
        neural = [state for state in states if state["steps"]]
        self.assertTrue(all(len(state["exposure_prefixes"]) == 2 for state in neural))
        self.assertTrue(all(len(state["validation_history"]) == 2 for state in neural))

    def test_final_update_validation_interruption_recovers_with_zero_new_updates(self):
        spec = small_spec()
        spec["matrix"].update(architectures=["direct"], seeds=[101], subset_seeds=[37])
        spec.pop("baseline")
        create_study(self.manifest, spec, self.root / "study")
        plan_path = self.root / "study" / "study.json"
        with patch.object(TrainingSession, "validate", side_effect=ResourceLimitError("interrupted final validation")):
            with self.assertRaises(ResourceLimitError):
                advance_study(plan_path)
        state = json.loads(next((self.root / "study" / "jobs").glob("*/job.json")).read_text())
        self.assertEqual(state["steps"], 1)
        self.assertTrue(state["pending_validation"])
        recovered = advance_study(plan_path, max_new_updates=0)
        self.assertEqual(recovered["new_updates"], 0)
        self.assertTrue(recovered["jobs"][0]["completed"])

    def test_foreign_checkpoint_with_identical_hyperparameters_is_rejected(self):
        import shutil
        spec = small_spec(2)
        spec["matrix"].update(architectures=["direct"], seeds=[101], subset_seeds=[37])
        spec.pop("baseline")
        for name in ("first", "second"):
            create_study(self.manifest, spec, self.root / name)
            advance_study(self.root / name / "study.json")
        first = next((self.root / "first" / "jobs").glob("*/latest.pt"))
        second = next((self.root / "second" / "jobs").glob("*/latest.pt"))
        shutil.copyfile(first, second)
        with self.assertRaisesRegex(ValueError, "different study job"):
            advance_study(self.root / "second" / "study.json")

    def test_extra_rng_consumption_in_one_architecture_invalidates_comparison(self):
        spec = small_spec()
        spec["matrix"].update(seeds=[101], subset_seeds=[37])
        spec.pop("baseline")
        create_study(self.manifest, spec, self.root / "study")
        original = TrainingSession.train_step
        def mismatched(session):
            if session.model_config.architecture == "das_unet":
                session.rng.integers(1000000)
            return original(session)
        with patch.object(TrainingSession, "train_step", mismatched):
            with self.assertRaisesRegex(ValueError, "different training observations"):
                advance_study(self.root / "study" / "study.json", max_new_updates=2)
        self.assertFalse((self.root / "study" / "jobs" / "B-r128-extra0-seed101" / "selected.pt").exists())

    def test_study_lock_blocks_another_process_and_releases_normally(self):
        import subprocess
        import sys
        folder = self.root / "locked"
        folder.mkdir()
        script = """from pathlib import Path
import sys
from traxis.studies import study_lock
try:
    with study_lock(Path(sys.argv[1])):
        sys.exit(2)
except RuntimeError:
    sys.exit(0)
"""
        with study_lock(folder):
            result = subprocess.run([sys.executable, "-c", script, str(folder)], capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)
        with study_lock(folder):
            pass

    def test_plan_source_and_completed_artifact_changes_are_detected(self):
        from traxis.cli import main
        config = self.root / "config.json"
        config.write_text(json.dumps(small_spec()))
        with patch("sys.argv", ["traxis", "plan-study", "--dataset", str(self.manifest), "--config", str(config),
                                "--output", str(self.root / "study")]):
            main()
        path = self.root / "study" / "study.json"
        with patch("traxis.studies._source_hash", return_value="changed"):
            with self.assertRaisesRegex(ValueError, "source"):
                load_study(path)
        advance_study(path)
        artifact = next((self.root / "study" / "jobs").glob("*/selected.pt"))
        artifact.write_bytes(artifact.read_bytes() + b"changed")
        with self.assertRaisesRegex(ValueError, "artifact changed"):
            advance_study(path)
