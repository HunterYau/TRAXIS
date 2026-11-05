import math
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np
from numpy.polynomial.legendre import leggauss

from traxis.config import Material, Plate
from traxis.elmer_reference import elmer_environment, operator_sif, read_elmer_operators, run_monitored, write_elmer_mesh
from traxis.geometry import structured_mesh
from traxis.quadratic_mass import exact_quadratic_mass, normalized_quadratic_mass
from traxis.quadratic_patches import TETRAHEDRON_EDGES, load_quadratic_elmer_mesh
from traxis.resources import ResourceGuard, ResourceLimits


def unit_quadratic_tetrahedron():
    corners = np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]])
    return np.r_[corners, corners[TETRAHEDRON_EDGES].mean(axis=1)], np.arange(10)[None]


class ExactQuadraticMassTests(unittest.TestCase):
    def test_exact_simplex_products_match_independent_four_point_duffy_rule(self):
        points, tets = unit_quadratic_tetrahedron()
        exact = exact_quadratic_mass(points, tets, 1.)[::3, ::3]
        errors = []
        for count in (3, 4):
            positions, weights = leggauss(count)
            positions, weights = (positions+1)/2, weights/2
            mass = np.zeros((10,10))
            for i, r in enumerate(positions):
                for j, s in enumerate(positions):
                    for k, t in enumerate(positions):
                        x, y, z = r*(1-s)*(1-t), s*(1-t), t
                        barycentric = np.array([1-x-y-z, x, y, z])
                        basis = np.r_[barycentric*(2*barycentric-1), 4*np.prod(barycentric[TETRAHEDRON_EDGES], axis=1)]
                        mass += weights[i]*weights[j]*weights[k]*(1-s)*(1-t)**2 * np.outer(basis,basis)
            errors.append(np.linalg.norm(mass-exact)/np.linalg.norm(exact))
        self.assertGreater(errors[0], 1e-3)
        self.assertLess(errors[1], 1e-12)
        normalized = normalized_quadratic_mass()
        np.testing.assert_allclose(normalized.sum(axis=1), np.r_[np.full(4,-1/20),np.full(6,1/5)], atol=1e-15)
        self.assertGreater(np.linalg.eigvalsh(normalized).min(), 0.)

    def test_global_node_and_midpoint_orderings_preserve_the_same_mass(self):
        points, tets = unit_quadratic_tetrahedron()
        expected = exact_quadratic_mass(points, tets, 2700.)
        order = np.array([4,0,9,2,7,6,8,1,5,3])
        inverse = np.argsort(order)
        shuffled = inverse[tets]
        shuffled[:,4:] = shuffled[:,4:][:,::-1]
        actual = exact_quadratic_mass(points[order], shuffled, 2700.)
        dofs = (3*inverse[:,None]+np.arange(3)).ravel()
        np.testing.assert_allclose(actual[dofs][:,dofs], expected, atol=1e-12)

    def test_curved_midpoints_and_bad_inputs_are_rejected(self):
        points, tets = unit_quadratic_tetrahedron()
        points[4,2] += .001
        with self.assertRaisesRegex(ValueError, 'straight'):
            exact_quadratic_mass(points, tets, 1.)
        with self.assertRaises(ValueError):
            exact_quadratic_mass(points, tets, math.nan)


@unittest.skipUnless(os.environ.get('TRAXIS_ELMER_HOME'), 'set TRAXIS_ELMER_HOME for native mass quadrature regression')
class NativeMassQuadratureTests(unittest.TestCase):
    def test_higher_rule_integrates_p2_mass_and_default_rule_exposes_old_error(self):
        solver, env = elmer_environment(Path(os.environ['TRAXIS_ELMER_HOME']))
        guard = ResourceGuard(ResourceLimits(max_seconds=15, max_rss_mib=512))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plate = Plate(.04, .03, .003)
            write_elmer_mesh(structured_mesh(plate,(1,1,1)),root/'linear-mesh')
            conversion = run_monitored([str(solver.with_name('ElmerGrid'+solver.suffix)), '2', '2', 'linear-mesh', '-increase', '-decimals', '17', '-out', 'mesh'],
                                       root, env, guard, root/'grid.log')
            self.assertEqual(conversion['exit_code'], 0)
            mesh = load_quadratic_elmer_mesh(root/'mesh',plate)
            expected = exact_quadratic_mass(mesh.points_m,mesh.tetrahedra,Material().density_kg_m3)
            errors = []
            for relative in (0,1):
                prefix = f'order{relative}'
                text = operator_sif(Material(), relative_integration_order=relative).replace('String "operator"', f'String "{prefix}"')
                (root/'operator.sif').write_text(text)
                process = run_monitored([str(solver),'operator.sif'],root,env,guard,root/f'{prefix}.log')
                self.assertEqual(process['exit_code'], 0)
                for suffix in ('sizes','a','perm'):
                    (root/f'operator_{suffix}.dat').write_bytes((root/f'{prefix}_{suffix}.dat').read_bytes())
                _, mass = read_elmer_operators(root,len(mesh.points_m))
                errors.append(np.linalg.norm(mass-expected)/np.linalg.norm(expected))
            self.assertGreater(errors[0],1e-3)
            self.assertLess(errors[1],1e-10)
