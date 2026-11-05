from dataclasses import replace
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np

from traxis.config import Material, Plate
from traxis.geometry import structured_mesh
from traxis.highorder import assemble_highorder, highorder_dofs
from traxis.highorder_verification import run_native_job, verify_highorder
from traxis.resources import ResourceGuard, ResourceLimits
from traxis.storage import canonical_hash, write_json


class HighOrderPreflightTests(unittest.TestCase):
    def test_topological_counts_for_complete_conforming_tetrahedron_spaces(self):
        mesh = structured_mesh(Plate(.04, .03, .003), (2, 2, 1))
        for order, expected in zip((1, 2, 3, 4), (18, 75, 196, 405)):
            screen = highorder_dofs(mesh, order)
            self.assertEqual(screen['scalar_dofs'], expected)
            self.assertEqual(screen['vector_dofs'], 3 * expected)
        with self.assertRaisesRegex(MemoryError, 'screen'):
            assemble_highorder(mesh, Material(), 4, max_dofs=12)

    def test_invalid_geometry_and_degree_are_rejected_before_native_import(self):
        mesh = structured_mesh(Plate(.04, .03, .003), (1, 1, 1))
        for degree in (0, 5, 2.5, True):
            with self.assertRaises(ValueError):
                highorder_dofs(mesh, degree)
        bad = replace(mesh, tetrahedra=np.r_[mesh.tetrahedra, mesh.tetrahedra[:1]])
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            highorder_dofs(bad, 3)
        with self.assertRaises(ValueError):
            highorder_dofs(replace(mesh, points_m=mesh.points_m.astype(complex)), 3)

    def test_missing_job_input_bindings_and_tampered_job_fail_without_native_work(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            job = {'input_sha256': {}}
            job['fingerprint'] = canonical_hash(job)
            write_json(root / 'job.json', job)
            with self.assertRaisesRegex(ValueError, 'bind'):
                run_native_job(root / 'job.json', root / 'native', ResourceGuard())
            job['input_sha256'] = {'changed': 'value'}
            write_json(root / 'job.json', job, overwrite=True)
            with self.assertRaisesRegex(ValueError, 'changed'):
                run_native_job(root / 'job.json', root / 'native', ResourceGuard())
            self.assertFalse((root / 'native').exists())


@unittest.skipUnless(os.environ.get('TRAXIS_HIGHORDER_CHECKS') == '1' and os.environ.get('TRAXIS_ELMER_HOME')
                     and importlib.util.find_spec('ngsolve'), 'enable optional isolated NGSolve/Elmer reference checks')
class NativeHighOrderTests(unittest.TestCase):
    def test_rectangular_and_skewed_p1_p2_basis_equivalence_and_p1_p4_energies(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'reference'
            result = verify_highorder(Path(os.environ['TRAXIS_ELMER_HOME']), output,
                ResourceGuard(ResourceLimits(max_seconds=60, max_rss_mib=1024)))
            self.assertTrue(result['completed'])
            self.assertTrue(result['passed'])
            self.assertFalse(result['ready_for_training_data'])
            self.assertEqual(len(result['cases']), 2)
            for case in result['cases']:
                self.assertEqual([row['order'] for row in case['orders']], [1, 2, 3, 4])
                for row in case['orders']:
                    self.assertTrue(row['passed'])
                    self.assertTrue(row['six_rigid_modes'])
                    self.assertTrue(row['ritz_nonincrease_passed'])
                    self.assertLess(row['energies']['maximum_mass_relative_error'], 1e-10)
                    self.assertLess(row['energies']['maximum_stiffness_relative_error'], 1e-10)
                basis = case['orders'][1]['p2_elmer_basis_comparison']
                self.assertEqual(basis['basis_rank'], 225)
                self.assertLess(max(basis['mass_relative_frobenius'], basis['stiffness_relative_frobenius']), 1e-10)
            report = json.loads((output / 'highorder-report.json').read_text())
            self.assertEqual(report['fingerprint'], canonical_hash({k: v for k, v in report.items() if k != 'fingerprint'}))
