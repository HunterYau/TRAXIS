from dataclasses import fields, replace
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from tests.fixtures import analytic_bank
from traxis.bank import Observation, ReplayEnvironment, WaveformBank, validate_specimen_splits


class BankTests(unittest.TestCase):
    def test_cache_roundtrip_and_no_implicit_overwrite(self):
        bank = analytic_bank()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.npz"
            bank.save(path)
            loaded = WaveformBank.load(path)
            self.assertEqual(bank.metadata, loaded.metadata)
            self.assertEqual(bank.actions, loaded.actions)
            for name in bank._arrays():
                np.testing.assert_array_equal(getattr(bank, name), getattr(loaded, name))
            with self.assertRaises(FileExistsError):
                bank.save(path)
            bank.save(path, overwrite=True)
            self.assertEqual(len(list(Path(directory).iterdir())), 1)

    def test_corrupt_arrays_and_oversized_archives_fail(self):
        bank = analytic_bank()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.npz"
            bank.save(path)
            with self.assertRaises(MemoryError):
                WaveformBank.load(path, max_uncompressed_mib=0.001)
            with np.load(path, allow_pickle=False) as archive:
                arrays = {key: archive[key] for key in archive.files}
            arrays["waveforms"][0, 0] += 1
            np.savez_compressed(path, **arrays)
            with self.assertRaisesRegex(ValueError, "integrity"):
                WaveformBank.load(path)

    def test_invalid_records_labels_and_times_are_rejected(self):
        bank = analytic_bank()
        cases = [{"actions": bank.actions + bank.actions[:1]},
                 {"time_s": bank.time_s[::-1]},
                 {"waveforms": bank.waveforms[:, :-1]},
                 {"depth_mm": np.full((4, 4), 4.)},
                 {"baseline": np.full_like(bank.baseline, np.nan)}]
        for changes in cases:
            with self.subTest(changes=list(changes)), self.assertRaises(ValueError):
                replace(bank, **changes)

    def test_parent_variants_cannot_cross_simulation_or_physical_splits(self):
        a = analytic_bank(specimen_id="scan-1")
        b = analytic_bank(specimen_id="scan-2", split="test")
        with self.assertRaisesRegex(ValueError, "crosses splits"):
            validate_specimen_splits([a, b])
        b = analytic_bank(specimen_id="scan-2")
        validate_specimen_splits([a, b])
        a = analytic_bank(specimen_id="real-1", domain="physical", split="adaptation")
        b = analytic_bank(specimen_id="real-2", domain="physical", split="physical_holdout")
        with self.assertRaises(ValueError):
            validate_specimen_splits([a, b])


class RevealTests(unittest.TestCase):
    def test_only_selected_records_are_revealed(self):
        bank = analytic_bank()
        env = ReplayEnvironment(bank)
        initial = env.state()
        self.assertEqual(initial.count, 0)
        self.assertFalse(initial.waveforms.any())
        self.assertFalse(initial.baseline.any())
        state = env.reveal(5)
        np.testing.assert_array_equal(state.waveforms[5], bank.waveforms[5])
        self.assertFalse(state.waveforms[~state.observed].any())
        self.assertEqual(state.count, 1)
        self.assertEqual({f.name for f in fields(Observation)},
                         {"plate", "observable", "sensors_xy_m", "actions", "time_s", "waveforms", "baseline", "observed"})
        with self.assertRaises(ValueError):
            env.reveal(5)
        self.assertEqual(env.reset().count, 0)

    def test_mutating_observation_cannot_change_cache_or_environment(self):
        bank = analytic_bank()
        env = ReplayEnvironment(bank)
        state = env.reveal(0)
        state.waveforms.setflags(write=True)
        state.observed.setflags(write=True)
        state.waveforms[:] = 999
        state.observed[:] = True
        self.assertEqual(env.state().count, 1)
        np.testing.assert_array_equal(env.state().waveforms[0], bank.waveforms[0])
