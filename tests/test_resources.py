import unittest
from dataclasses import fields
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

    def test_default_system_ram_guard_stops_at_ninety_percent(self):
        guard = ResourceGuard()
        guard.enforce({**self.sample, "system_memory_percent": 89.9}, [])
        for percent in (90., 91., 99.):
            with self.assertRaisesRegex(ResourceLimitError, 'system RAM'):
                guard.enforce({**self.sample, "system_memory_percent": percent}, [])

    def test_missing_temperatures_are_not_reported_as_safe_readings(self):
        guard = ResourceGuard()
        with patch("traxis.resources.snapshot", return_value=self.sample.copy()):
            guard.check(force=True)
        self.assertIsNone(guard.report()["samples"][0]["cpu_temperatures_c"])
        self.assertIn("not verified", guard.report()["cpu_temperature_note"])

    def test_latest_live_sample_stays_fresh_after_history_cap(self):
        guard = ResourceGuard()
        with patch("traxis.resources.snapshot", side_effect=[{**self.sample, "sample_id": i}
                for i in range(10002)]):
            for _ in range(10002):
                guard.check(force=True)
        self.assertEqual(len(guard.history), 10000)
        self.assertEqual(guard.latest_sample["sample_id"], 10001)
        self.assertIsNot(guard.latest_sample, guard.history[-1])
        self.assertEqual(guard.report()["latest_sample"]["sample_id"], 10001)

    def test_time_budget_is_enforced(self):
        guard = ResourceGuard()
        guard.started -= 61
        with self.assertRaises(ResourceLimitError):
            guard.check()

    def test_invalid_limits(self):
        with self.assertRaises(ValueError):
            ResourceLimits(max_seconds=float("inf"))

    def test_unlimited_ram_keeps_telemetry_and_non_ram_stops(self):
        limits = ResourceLimits(max_rss_mib=None, max_system_memory_percent=None,
                                min_available_mib=None)
        guard = ResourceGuard(limits)
        sample = {**self.sample, 'process_rss_mib': 65536,
                  'system_memory_percent': 100., 'available_mib': 0.}
        with patch('traxis.resources.snapshot', return_value=sample.copy()):
            guard.check(force=True)
        self.assertEqual(guard.report()['latest_sample']['system_memory_percent'], 100.)
        self.assertIsNone(guard.report()['limits']['max_rss_mib'])
        with self.assertRaisesRegex(ResourceLimitError, 'GPU memory'):
            guard.enforce(sample, [{'used_mib': 95, 'total_mib': 100, 'temperature_c': 40}])
        with self.assertRaisesRegex(ResourceLimitError, 'GPU temperature'):
            guard.enforce(sample, [{'used_mib': 20, 'total_mib': 100, 'temperature_c': 85}])
        with self.assertRaisesRegex(ResourceLimitError, 'CPU temperature'):
            guard.enforce({**sample, 'cpu_temperatures_c': [95]}, [])
        guard.started -= 61
        with self.assertRaisesRegex(ResourceLimitError, 'wall-time'):
            guard.check(force=True)

    def test_all_resource_stops_can_be_disabled_without_losing_telemetry(self):
        limits=ResourceLimits(**{field.name:None for field in fields(ResourceLimits)})
        guard=ResourceGuard(limits)
        guard.started-=1e9
        sample={**self.sample,'process_rss_mib':65536,'system_memory_percent':100.,
                'available_mib':0.,'cpu_temperatures_c':[100.],
                'gpus':[{'used_mib':100,'total_mib':100,'temperature_c':100.}]}
        with patch('traxis.resources.snapshot',return_value=sample):
            guard.check(force=True)
        self.assertEqual(guard.latest_sample['gpus'][0]['temperature_c'],100.)
        self.assertTrue(all(v is None for v in guard.report()['limits'].values()))

    def test_nonfatal_telemetry_failure_recovers_on_next_reading(self):
        guard=ResourceGuard(telemetry_errors_fatal=False)
        with patch('traxis.resources.snapshot',side_effect=[OSError('sensor unavailable'),self.sample.copy()]):
            with self.assertWarnsRegex(RuntimeWarning,'simulation continues'):
                guard.check(force=True)
            self.assertIsNone(guard.latest_sample)
            self.assertEqual(guard.last_gpus,[])
            self.assertEqual(guard.telemetry_error,'sensor unavailable')
            guard.check(force=True)
        self.assertIsNone(guard.telemetry_error)
        self.assertEqual(guard.latest_sample['system_memory_percent'],50)

    def test_telemetry_failure_still_propagates_by_default(self):
        with patch('traxis.resources.snapshot',side_effect=OSError('sensor unavailable')):
            with self.assertRaises(OSError):ResourceGuard().check(force=True)

    def test_process_memory_stop_reports_measurement_and_limit(self):
        guard = ResourceGuard(ResourceLimits(max_rss_mib=50))
        with self.assertRaisesRegex(ResourceLimitError, r'100\.0 MiB > 50\.0 MiB'):
            guard.enforce(self.sample, [])
