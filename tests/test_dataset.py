from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

from tests.fixtures import analytic_bank
from traxis.dataset import BankDataset


def fixture_dataset(directory: Path, splits=("train", "validation", "test"), prefix="") -> BankDataset:
    directory = Path(directory)
    paths = []
    for index, split in enumerate(splits):
        domain = "physical" if split in ("adaptation", "physical_holdout", "calibration") else "simulation"
        bank = analytic_bank(intact=index % 2 == 1, specimen_id=f"{prefix}scan-{index}", parent_specimen_id=f"{prefix}parent-{index}",
                             split=split, domain=domain)
        path = directory / f"bank-{index}.npz"
        bank.save(path)
        paths.append(path)
    return BankDataset.build(paths, directory / "dataset.json")


class DatasetTests(unittest.TestCase):
    def test_roundtrip_lazy_read_and_changed_bank_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = fixture_dataset(directory)
            loaded = BankDataset.load(Path(directory) / "dataset.json")
            self.assertEqual(dataset.fingerprint, loaded.fingerprint)
            self.assertEqual(len(loaded.select("train")), 1)
            entry = loaded.select("train")[0]
            bank = loaded.read(entry)
            bank.save(Path(directory) / entry.path, overwrite=True)
            # Equal content may serialize identically, so change an actual signal.
            altered = replace(bank, waveforms=bank.waveforms * 0.5)
            altered.save(Path(directory) / entry.path, overwrite=True)
            with self.assertRaisesRegex(ValueError, "changed"):
                loaded.read(entry)

    def test_parent_split_and_duplicate_id_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory)
            a = analytic_bank(specimen_id="a", parent_specimen_id="same")
            b = analytic_bank(specimen_id="b", parent_specimen_id="same", split="test")
            a.save(p / "a.npz")
            b.save(p / "b.npz")
            with self.assertRaisesRegex(ValueError, "crosses"):
                BankDataset.build([p / "a.npz", p / "b.npz"], p / "manifest.json")

    def test_manifest_is_bound_to_entries(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = fixture_dataset(directory)
            with self.assertRaises(ValueError):
                dataset.read(replace(dataset.entries[0], parent_specimen_id="different"))
