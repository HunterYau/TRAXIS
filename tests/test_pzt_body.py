"""Independent mechanics checks for a resolved body attached to a plate."""
import unittest
import numpy as np
from scipy.linalg import solve

from tests.test_pzt_receiver import material
from traxis.config import Plate, Material
from traxis.gpu_tensor import make_layout, mass_diagonal
from traxis.pzt_body import attach_disk, merge_attachments
from traxis.pzt_receiver import DiskGeometry


class BodyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.layout = make_layout(Plate(.04, .04, .001), (2, 2, 1), 2)
        cls.geometry = DiskGeometry((.02, .02), .005, .0005, 'top', 1)
        cls.body = attach_disk(cls.layout, material(), cls.geometry,
            sectors=8, rings=1, layers=1, electrical='voltage')

    def state(self, field, body=None):
        body = self.body if body is None else body
        host = field(self.layout.points()).T.ravel()
        free = field(body.points_m).T.ravel()[body.free_body_dofs]
        return np.r_[host, free]

    def test_rigid_translation_mass_and_rotation_energy(self):
        b = self.body
        for component in range(3):
            def field(x):
                answer = np.zeros_like(x); answer[:, component] = 1
                return answer
            u = self.state(field)[b.indices]
            self.assertAlmostEqual(u@b.mass@u/b.report['body_mass_kg'], 1., places=12)
            self.assertLess(abs(u@b.stiffness@u)/(np.linalg.norm(b.stiffness)*np.dot(u,u)), 1e-13)
        def rotation(x): return np.cross(np.array([.7, -.3, .2]), x)
        u = self.state(rotation)[b.indices]
        self.assertLess(abs(u@b.stiffness@u)/(np.linalg.norm(b.stiffness)*np.dot(u,u)), 1e-13)
        self.assertLess(abs(b.coupling@u), 1e-14)

    def test_affine_motion_is_continuous_at_tied_face(self):
        matrix = np.array([[.1,.2,.3],[-.2,.4,.1],[.3,-.1,.6]])
        def field(x): return x@matrix.T+np.array([.01,.02,.03])
        b = self.body
        np.testing.assert_allclose(b.transform@self.state(field)[b.indices],
            field(b.points_m).T.ravel(), atol=3e-16, rtol=1e-13)

    def test_open_circuit_adds_correct_positive_backreaction(self):
        b = self.body
        opened = attach_disk(self.layout, material(), self.geometry,
            sectors=8, rings=1, layers=1, electrical='open', input_capacitance_f=1e-9)
        expected = np.outer(b.coupling,b.coupling)/(b.capacitance_f+1e-9)
        # Subtracting O(1e9) stiffness entries loses absolute precision in
        # small O(10) corrections; assess the complete energy operator.
        self.assertLess(np.linalg.norm(opened.stiffness-b.stiffness-expected)/np.linalg.norm(expected),1e-12)
        u = np.random.default_rng(12).normal(size=len(b.indices))
        self.assertGreaterEqual(u@expected@u, 0)

    def test_overlapping_bodies_use_a_joint_consistent_mass_inverse(self):
        b = self.body
        # Bodies do not overlap physically, but their host basis supports do.
        other = attach_disk(self.layout, material(),
            DiskGeometry((.032,.02),.005,.0005,'top',1), first_dof=b.next_dof,
            sectors=8,rings=1,layers=1)
        diagonal, patches = merge_attachments(mass_diagonal(self.layout,Material()),[b,other])
        self.assertEqual(len(patches),1)
        m = np.diag(np.r_[mass_diagonal(self.layout,Material()), np.zeros(len(diagonal)-self.layout.screen['vector_dofs'])])
        for body in (b,other): m[np.ix_(body.indices,body.indices)] += body.mass
        f = np.random.default_rng(44).normal(size=len(m)); actual=f/diagonal
        for p in patches: actual[p.indices]=p.inverse@f[p.indices]
        np.testing.assert_allclose(actual,solve(m,f,assume_a='pos'),rtol=2e-12,atol=1e-6)
        self.assertGreater(np.linalg.norm(m-np.diag(np.diag(m))),0)

    def test_unknown_glue_or_resistive_load_is_not_silently_invented(self):
        for kwargs in ({'bond':None},{'bond':'salol'},{'electrical':'resistive'}):
            with self.assertRaises(ValueError):
                attach_disk(self.layout,material(),self.geometry,**kwargs)

    def test_polygon_mass_converges_to_circular_mass(self):
        errors=[]
        for sectors in (8,16,32):
            b=attach_disk(self.layout,material(),self.geometry,sectors=sectors,rings=1,layers=1)
            errors.append(b.report['circular_area_relative_error'])
        self.assertTrue(errors[2]<errors[1]<errors[0])
        self.assertLess(errors[2],.0065)


if __name__ == '__main__': unittest.main()
