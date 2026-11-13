"""The execution profile extends wall time while binding every actual guard."""
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
import unittest
from unittest.mock import patch

from examples.embedded_wave import case_config
from examples.embedded_wave_budget import ExtendedResourceGuard,check_execution,execution_profile
from traxis.resources import ResourceGuard,ResourceLimitError,ResourceLimits
from traxis.storage import sha256_file


class EmbeddedBudgetTests(unittest.TestCase):
    def plan(self):
        case=case_config('full',4,294912)
        profile=execution_profile(case)
        plan=dict(case=case,execution=profile,analysis_sources={
            'examples/embedded_wave_budget.py':sha256_file(Path(__file__).resolve().parents[1]/'examples/embedded_wave_budget.py')})
        return plan,ResourceGuard(ResourceLimits(**profile['limits']))

    def test_actual_guard_matches_recorded_profile_and_physics_stays_fixed(self):
        plan,guard=self.plan();check_execution(plan,guard)
        original=asdict(ResourceLimits(max_seconds=2400,max_rss_mib=2048,max_system_memory_percent=85,min_available_mib=2048))
        actual=asdict(guard.limits)
        self.assertEqual(actual['max_seconds'],3600)
        self.assertEqual({k:v for k,v in actual.items() if k!='max_seconds'},
                         {k:v for k,v in original.items() if k!='max_seconds'})
        self.assertEqual(plan['case'],case_config('full',4,294912))

    def test_changed_memory_thermal_time_source_and_actual_guard_reject(self):
        plan,guard=self.plan()
        for key,value in [('max_seconds',7200),('max_rss_mib',4096),('max_gpu_temperature_c',90),('min_available_mib',512)]:
            changed=deepcopy(plan);changed['execution']['limits'][key]=value
            with self.assertRaises(ValueError):check_execution(changed,guard)
        changed=deepcopy(plan);changed['analysis_sources']['examples/embedded_wave_budget.py']='0'*64
        with self.assertRaises(ValueError):check_execution(changed,guard)
        other=ResourceGuard(ResourceLimits(max_seconds=3600,max_rss_mib=4096,max_system_memory_percent=85,min_available_mib=2048))
        with self.assertRaises(ValueError):check_execution(plan,other)

    def test_unlisted_or_small_case_is_rejected(self):
        with self.assertRaises(ValueError):execution_profile(case_config('smoke'))
        plan,guard=self.plan();plan['case']['steps']=500000
        with self.assertRaises(ValueError):check_execution(plan,guard)


class ExtendedTelemetryTests(unittest.TestCase):
    def sample(self,index):
        return dict(process_rss_mib=100.,system_memory_percent=50.,available_mib=6000.,cpu_temperatures_c=[35.],
            gpus=[dict(used_mib=500.,total_mib=8192.,temperature_c=40.)],gpu_status='available',elapsed_seconds=float(index))

    def test_compaction_keeps_all_extrema_and_late_mutable_process_sample(self):
        guard=ExtendedResourceGuard(ResourceLimits(max_seconds=3600,max_rss_mib=2048,max_system_memory_percent=85,min_available_mib=2048))
        guard.history=[self.sample(i) for i in range(10000)]
        guard.history[101]['process_rss_mib']=1000.
        guard.history[201]['system_memory_percent']=84.
        guard.history[301]['available_mib']=2200.
        guard.history[401]['gpus'][0]['used_mib']=6000.
        guard.history[501]['gpus'][0]['temperature_c']=79.
        guard.history[601]['cpu_temperatures_c']=[89.]
        last=guard.history[-1]
        with patch('traxis.resources.snapshot',return_value=self.sample(10000)):
            guard.check(force=True)
        self.assertLess(len(guard.history),8000);self.assertIsNot(guard.history[-1],last)
        self.assertEqual(guard.history[-2],last)
        for index in (101,201,301,401,501,601):
            self.assertTrue(any(s['elapsed_seconds']==index for s in guard.history))
        # The external-process monitor augments this fresh sample in place.
        guard.history[-1]['process_rss_mib']+=500
        self.assertEqual(last['process_rss_mib'],100.)
        self.assertEqual(guard.report()['history_compactions'],1)
        self.assertEqual(guard.report()['total_samples_observed'],1)

    def test_resource_failure_remains_enforced_and_counted(self):
        guard=ExtendedResourceGuard(ResourceLimits(max_seconds=3600,max_rss_mib=2048,max_system_memory_percent=85,min_available_mib=2048))
        bad=self.sample(0);bad['process_rss_mib']=5000.
        with patch('traxis.resources.snapshot',return_value=bad):
            with self.assertRaises(ResourceLimitError):guard.check(force=True)
        self.assertEqual(guard.total_samples,1);self.assertEqual(guard.history[-1]['process_rss_mib'],5000.)


class ComparedExecutionTests(unittest.TestCase):
    def test_original_and_extended_reports_must_match_both_actual_guards(self):
        from examples.embedded_wave_study import checked_resources
        for extended in (False,True):
            case=case_config('full',4,294912 if extended else 131072)
            profile=execution_profile(case) if extended else None
            limits=profile['limits'] if extended else asdict(ResourceLimits(max_seconds=2400,max_rss_mib=2048,max_system_memory_percent=85,min_available_mib=2048))
            level=dict(case=case,plan=dict(case=case),parent=dict(resources=dict(limits=deepcopy(limits))),native=dict(resources=dict(limits=deepcopy(limits))))
            if extended:level['plan']['execution']=deepcopy(profile);level['parent']['execution']=deepcopy(profile)
            self.assertEqual(checked_resources(level),limits)
            for key in ('native','parent'):
                changed=deepcopy(level);changed[key]['resources']['limits']['max_rss_mib']=4096
                with self.assertRaises(ValueError):checked_resources(changed)
            changed=deepcopy(level);changed['parent']['execution']=dict(kind='unbound')
            with self.assertRaises(ValueError):checked_resources(changed)


if __name__=='__main__':unittest.main()
