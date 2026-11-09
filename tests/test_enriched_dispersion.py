import unittest

import numpy as np

from examples.enriched_dispersion import periodic_projection
from traxis.config import Plate
from traxis.enriched_tet import build_topology
from traxis.geometry import structured_mesh


class EnrichedPeriodicProjectionTests(unittest.TestCase):
    def setUp(self):
        self.plate = Plate(.004,.004,.003)
        self.points = build_topology(structured_mesh(self.plate,(1,1,1))).points_m

    def test_only_periodic_faces_are_tied_and_interior_coefficients_remain_free(self):
        points,n,k = self.points,len(self.points),231.
        projection = periodic_projection(points,self.plate,1,k).toarray()
        self.assertFalse(np.any(projection[n:2*n]))
        np.testing.assert_equal(np.count_nonzero(projection[:n],axis=1),1)
        self.assertEqual(np.linalg.matrix_rank(projection),projection.shape[1])
        for axis in (0,1):
            extent = self.plate.width_m
            for i,point in enumerate(points):
                if point[axis]!=0.:
                    continue
                target = point.copy()
                target[axis] = extent
                j = np.flatnonzero(np.all(np.isclose(points,target,rtol=0,atol=1e-14),axis=1))[0]
                phase = np.exp(1j*k*extent) if axis==0 else 1.
                np.testing.assert_allclose(projection[j],phase*projection[i],rtol=1e-14,atol=1e-14)
        inside = np.flatnonzero(np.all((points[:,:2]>1e-14)&(points[:,:2]<.004-1e-14),axis=1))
        self.assertGreater(len(inside),1)
        self.assertEqual(len(np.unique(projection[inside].argmax(axis=1))),len(inside))

    def test_unmatched_faces_and_nonlattice_nodes_are_rejected(self):
        damaged = self.points.copy()
        damaged[0,0] += 1e-6
        with self.assertRaises(ValueError):
            periodic_projection(damaged,self.plate,1,231.)
        with self.assertRaisesRegex(ValueError,'matching node'):
            periodic_projection(self.points[1:],self.plate,1,231.)
        with self.assertRaises(ValueError):
            periodic_projection(self.points,self.plate,0,231.)
