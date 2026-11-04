import unittest
from unittest.mock import patch

from traxis.resources import ResourceGuard, ResourceLimitError, ResourceLimits


class ResourceTests(unittest.TestCase):
    def setUp(self):
        self.sample = {"process_rss_mib": 100, "system_memory_percent": 50,
                       "available_mib": 2000, "cpu_temperatures_c": None, "gpus": []}

    def test_memory_and_temperature_stops(self):
        guard = ResourceGuard()
        for key, value in (("process_rss_mib", 600), ("system_memory_percent", 99),
                           ("available_mib", 100), ("cpu_temperatures_c", [95])):
            with self.assertRaises(ResourceLimitError):
                guard.enforce({**self.sample, key: value}, [])
        for used, temp in ((95, 40), (20, 85)):
            with self.assertRaises(ResourceLimitError):
                guard.enforce(self.sample, [{"used_mib": used, "total_mib": 100, "temperature_c": temp}])

    def test_missing_temperatures_are_not_reported_as_safe_readings(self):
        guard = ResourceGuard()
        with patch("traxis.resources.snapshot", return_value=self.sample.copy()):
            guard.check(force=True)
        self.assertIsNone(guard.report()["samples"][0]["cpu_temperatures_c"])
        self.assertIn("not verified", guard.report()["cpu_temperature_note"])

    def test_time_budget_is_enforced(self):
        guard = ResourceGuard()
        guard.started -= 61
        with self.assertRaises(ResourceLimitError):
            guard.check()

    def test_invalid_limits(self):
        with self.assertRaises(ValueError):
            ResourceLimits(max_seconds=float("inf"))
