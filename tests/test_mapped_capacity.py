"""Explicit capacity guards; synthetic count fixtures are not physical meshes."""
from dataclasses import replace
import hashlib
import os
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from examples.embedded_wave import case_config,gpu_controls
from examples.tensor_wave import read_json
from tests.test_gpu_mapped_tensor import small_system
from traxis.gpu_mapped_tensor import KERNEL,build_gpu_mapped,storage_screen
from traxis.mapped_tensor import action


class CapacityTests(unittest.TestCase):
    def test_legacy_gpu_forecast_keeps_its_default_and_h_cases_request_the_explicit_bound(self):
        system=small_system()
        baseline=read_json(Path(__file__).parent/'fixtures/embedded-legacy-identities.json')
        self.assertEqual(hashlib.sha256(KERNEL.encode()).hexdigest(),baseline['cuda_kernel_sha256'])
        self.assertEqual(storage_screen(system),storage_screen(system,max_vector_dofs=12000000))
        self.assertNotIn('max_vector_dofs',gpu_controls(case_config('full',4,294912)))
        self.assertEqual(gpu_controls(case_config('h_refinement',h_level=2))['max_vector_dofs'],24000000)

    def test_larger_count_needs_opt_in_and_cannot_bypass_memory_limits(self):
        system=small_system();ns=4100000
        mesh=replace(system.mesh,points_m=np.broadcast_to(np.zeros((1,3)),(ns,3)))
        fixture=replace(system,mesh=mesh,mass_kg=np.broadcast_to(np.ones(1),(3*ns,)))
        kwargs=dict(max_device_mib=6144,max_host_mib=4096)
        with self.assertRaises(MemoryError):storage_screen(fixture,**kwargs)
        self.assertEqual(storage_screen(fixture,max_vector_dofs=24000000,**kwargs)['vector_dofs'],12300000)
        with self.assertRaises(MemoryError):storage_screen(fixture,max_vector_dofs=12299999,**kwargs)
        with self.assertRaises(MemoryError):storage_screen(fixture,max_vector_dofs=24000000,max_device_mib=512,max_host_mib=4096)
        with patch('builtins.__import__',side_effect=AssertionError('CUDA import attempted')):
            for cap in (True,0,24000001,24000000.,float('nan')):
                with self.assertRaises(ValueError):build_gpu_mapped(system,max_vector_dofs=cap)


@unittest.skipUnless(os.environ.get('TRAXIS_GPU_CHECKS')=='1','explicit CUDA check not enabled')
class CapacityCUDAParityTests(unittest.TestCase):
    def test_explicit_capacity_uses_the_same_float64_operator(self):
        system=small_system(order=4)
        state=np.random.default_rng(20261115).normal(size=len(system.mass_kg))
        original=build_gpu_mapped(system);expanded=build_gpu_mapped(system,max_vector_dofs=24000000)
        expected=action(system,state);a=original.stiffness@state;b=expanded.stiffness@state
        self.assertLess(np.linalg.norm(a-b)/np.linalg.norm(expected),1e-12)
        self.assertLess(np.linalg.norm(b-expected)/np.linalg.norm(expected),1e-11)
        np.testing.assert_array_equal(expanded.mass_diagonal,system.mass_kg)


if __name__=='__main__':unittest.main()
