import importlib.util
from itertools import permutations
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np
from numpy.polynomial.legendre import leggauss

from traxis.config import Material,Plate
from traxis.enriched_tet import NODES,WEIGHTS,FACES,basis,barycentric_integral,build_topology,element_matrices,assemble_enriched
from traxis.fem import constitutive
from traxis.geometry import structured_mesh
from traxis.quadratic_mass import normalized_quadratic_mass
from traxis.resources import ResourceGuard,ResourceLimits


class EnrichedElementTests(unittest.TestCase):
    def test_nodal_partition_reproduction_and_required_positive_quadrature(self):
        np.testing.assert_allclose(basis(NODES),np.eye(15),rtol=0,atol=1e-14)
        rng = np.random.default_rng(2026)
        points = rng.dirichlet(np.ones(4),size=32)
        values,gradients = basis(points),basis(points,gradients=True)
        np.testing.assert_allclose(values.sum(axis=1),1.,rtol=0,atol=2e-14)
        np.testing.assert_allclose(gradients.sum(axis=1),0.,rtol=0,atol=2e-13)
        for i in range(4):
            for j in range(i,4):
                np.testing.assert_allclose(values@(NODES[:,i]*NODES[:,j]),points[:,i]*points[:,j],rtol=0,atol=2e-14)
                powers = tuple(int(k==i)+int(k==j) for k in range(4))
                self.assertAlmostEqual(float(WEIGHTS@np.prod(NODES**powers,axis=1)),barycentric_integral(powers),places=15)
        self.assertGreater(WEIGHTS.min(),0.)
        self.assertAlmostEqual(WEIGHTS.sum(),1/6,places=15)
        for j in range(4):
            powers = tuple(int(i==j) for i in range(4))
            self.assertAlmostEqual(float(WEIGHTS@np.prod(NODES**powers,axis=1)),barycentric_integral(powers),places=15)
        for face in FACES:
            powers = tuple(int(i in face) for i in range(4))
            self.assertAlmostEqual(float(WEIGHTS@np.prod(NODES**powers,axis=1)),barycentric_integral(powers),places=15)
        self.assertAlmostEqual(float(WEIGHTS@np.prod(NODES,axis=1)),barycentric_integral((1,1,1,1)),places=15)
        # This is an intentional enriched rule, not exact integration of every P4 monomial.
        self.assertGreater(abs(float(WEIGHTS@(NODES[:,0]**4))/barycentric_integral((4,0,0,0))-1),.1)
        self.assertLess(normalized_quadratic_mass().sum(axis=1).min(),0.)

    def test_derivatives_match_finite_differences(self):
        point = np.array([.17,.23,.29,.31])
        for axis in range(3):
            shift = np.zeros(4)
            shift[axis],shift[3] = 1e-6,-1e-6
            measured = (basis(point+shift)-basis(point-shift))/(2e-6)
            np.testing.assert_allclose(measured,basis(point,gradients=True)[:,axis],rtol=2e-8,atol=2e-9)

    def test_exact_integrals_match_independent_duffy_quadrature_and_strain_matrix(self):
        vertices = np.array([[.04,0.,0.],[.002,.03,.001],[.001,.001,.004],[0.,0.,0.]])
        material = Material()
        k,m,d = element_matrices(vertices,material)
        q,w = leggauss(6)
        q,w = (q+1)/2,w/2
        bary,weights = [],[]
        for i,r in enumerate(q):
            for j,s in enumerate(q):
                for l,t in enumerate(q):
                    xyz = np.array([r,(1-r)*s,(1-r)*(1-s)*t])
                    bary.append(np.r_[xyz,1-xyz.sum()])
                    weights.append(w[i]*w[j]*w[l]*(1-r)**2*(1-s))
        jac = (vertices[:3]-vertices[3]).T
        weights = np.array(weights)*abs(np.linalg.det(jac))
        shapes = basis(bary)
        gradients = basis(bary,gradients=True)@np.linalg.inv(jac)
        mass = material.density_kg_m3*np.einsum('q,qi,qj->ij',weights,shapes,shapes)
        strain = np.zeros((len(weights),6,45))
        strain[:,0,:15] = gradients[:,:,0]
        strain[:,1,15:30] = gradients[:,:,1]
        strain[:,2,30:] = gradients[:,:,2]
        strain[:,3,:15],strain[:,3,15:30] = gradients[:,:,1],gradients[:,:,0]
        strain[:,4,15:30],strain[:,4,30:] = gradients[:,:,2],gradients[:,:,1]
        strain[:,5,:15],strain[:,5,30:] = gradients[:,:,2],gradients[:,:,0]
        stiffness = np.einsum('q,qai,ab,qbj->ij',weights,strain,constitutive(material),strain,optimize=True)
        self.assertLess(np.linalg.norm(k-stiffness)/np.linalg.norm(stiffness),1e-11)
        self.assertLess(np.linalg.norm(m-mass)/np.linalg.norm(mass),1e-11)
        self.assertGreater(np.linalg.eigvalsh(m).min(),0.)
        self.assertGreater(d.min(),0.)

    def test_all_vertex_permutations_preserve_the_physical_operators(self):
        vertices = np.array([[.04,0.,0.],[.002,.03,.001],[.001,.001,.004],[0.,0.,0.]])
        k,m,d = element_matrices(vertices,Material())
        points = NODES@vertices
        for permutation in permutations(range(4)):
            permuted = vertices[list(permutation)]
            other = NODES@permuted
            matches = np.linalg.norm(other[:,None]-points[None],axis=-1).argmin(axis=1)
            self.assertEqual(len(np.unique(matches)),15)
            ek,em,ed = element_matrices(permuted,Material())
            ids = np.concatenate([matches+15*c for c in range(3)])
            self.assertLess(np.linalg.norm(ek-k[np.ix_(ids,ids)])/np.linalg.norm(k),1e-11)
            self.assertLess(np.linalg.norm(em-m[np.ix_(matches,matches)])/np.linalg.norm(m),1e-11)
            np.testing.assert_allclose(ed,d[matches],rtol=1e-13,atol=0.)

    def test_shared_face_traces_and_resource_screens(self):
        mesh = structured_mesh(Plate(.04,.03,.003),(1,1,1))
        topology = build_topology(mesh)
        self.assertEqual(len(topology.points_m),51)
        owners = {}
        for i,tet in enumerate(mesh.tetrahedra):
            for face in FACES:
                key = tuple(sorted(tet[list(face)]))
                owners.setdefault(key,[]).append(i)
        for face,cells in owners.items():
            if len(cells)!=2:
                continue
            point = np.array([.2,.3,.5])@mesh.points_m[list(face)]
            traces = []
            for cell in cells:
                vertices = mesh.points_m[mesh.tetrahedra[cell]]
                xyz = np.linalg.solve((vertices[:3]-vertices[3]).T,point-vertices[3])
                trace = np.zeros(len(topology.points_m))
                trace[topology.element_nodes[cell]] = basis(np.r_[xyz,1-xyz.sum()])
                traces.append(trace)
            np.testing.assert_allclose(*traces,rtol=0,atol=1e-13)
        with self.assertRaises(MemoryError):
            build_topology(mesh,max_dofs=45)
        with self.assertRaises(MemoryError):
            build_topology(mesh,max_workspace_mib=1e-6)
        with self.assertRaises(ValueError):
            basis([1.,1.,0.,0.])
        with self.assertRaises(ValueError):
            element_matrices(np.zeros((4,3)),Material())
        with self.assertRaises(ValueError):
            assemble_enriched(mesh,Material(),max_dofs=2001)


@unittest.skipUnless(os.environ.get('TRAXIS_HIGHORDER_CHECKS')=='1' and importlib.util.find_spec('ngsolve'),
    'enable optional native enriched-element reference checks')
class NativeEnrichedTests(unittest.TestCase):
    def test_native_enriched_and_p4_embeddings_and_explicit_dynamics(self):
        from traxis.enriched_verification import verify_enriched
        with tempfile.TemporaryDirectory() as directory:
            report = verify_enriched(Path(directory)/'reference',
                ResourceGuard(ResourceLimits(max_seconds=90,max_rss_mib=1024)))
            self.assertTrue(report['completed'])
            self.assertTrue(report['passed'])
            self.assertFalse(report['ready_for_training_data'])
            self.assertEqual(len(report['cases']),2)
            for row in report['cases']:
                self.assertEqual(row['all_coefficient_columns_checked'],row['dofs'])
                self.assertEqual(row['native_api_local_scalar_dofs'],15)
                self.assertEqual(row['native_api_local_vector_dofs'],45)
                self.assertEqual(row['native_embedding_rank'],row['dofs'])
                self.assertEqual(row['p4_embedding_rank'],row['dofs'])
                for key in ('native_stiffness_relative_frobenius','native_consistent_mass_relative_frobenius',
                    'p4_stiffness_relative_frobenius','p4_consistent_mass_relative_frobenius','native_nodal_mass_relative_frobenius'):
                    self.assertLess(row[key],1e-10)
                self.assertTrue(row['six_rigid_modes'])
                self.assertTrue(row['stability_bound_valid'])
                self.assertTrue(all(3.8<x<4.2 for x in row['temporal_error_ratios']))
