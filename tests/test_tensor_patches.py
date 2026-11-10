"""Analytic circle moments, physical footprint invariance and signed work."""
import unittest
import numpy as np
from traxis.config import Plate
from traxis.gpu_tensor import make_layout
from traxis.tensor_patches import circular_patch,disk_average


class TensorPatchTests(unittest.TestCase):
    def test_offset_and_cross_cell_disks_reproduce_represented_fields(self):
        plate=Plate(width_m=.04,height_m=.03,thickness_m=.003)
        for order in (2,3,4,5):
            for cells,center in (((2,2,1),(.02,.015)),((3,2,1),(.0173,.0121)),((1,1,1),(.02,.015))):
                layout=make_layout(plate,cells,order);patch=circular_patch(layout,center,.003)
                self.assertLess(patch.report['relative_area_error'],1e-10)
                self.assertLess(patch.report['bounded_coordinate_moment_maximum_absolute_error'],1e-10)
                scalar=layout.screen['scalar_dofs'];points=layout.points()
                vector=patch.vector(scalar)
                for degree in range(order+1):
                    field=np.zeros((3,scalar));field[2]=((points[:,0]-center[0])/.04)**degree
                    expected=(.003/.04)**degree*disk_average(degree,0)
                    self.assertLess(abs(vector@field.ravel()-expected),1e-10)
                force=2.7*vector;velocity=np.random.default_rng(order).normal(size=3*scalar)
                self.assertAlmostEqual(float(force@velocity),2.7*float(vector@velocity),places=12)

    def test_physical_disk_moments_match_across_meshes_and_order(self):
        for order,cells in ((3,(100,100,1)),(4,(150,150,1)),(5,(150,150,1))):
            layout=make_layout(Plate(),cells,order);patch=circular_patch(layout,(.02,.02),.003)
            self.assertAlmostEqual(patch.weights.sum(),1.,places=11)
            self.assertEqual(patch.report['analytic_area_m2'],np.pi*.003**2)
            self.assertGreater(patch.report['surface_cells'],1)
            self.assertLess(patch.report['radius_scaled_quadratic_maximum_absolute_error'],1e-10)

    def test_invalid_physical_disk_and_skewed_surface_are_rejected(self):
        layout=make_layout(Plate(),(10,10,1),4)
        with self.assertRaises(ValueError):circular_patch(layout,(.001,.001),.003)
        with self.assertRaises(ValueError):circular_patch(layout,(.1,.1),0.)
        with self.assertRaises(MemoryError):circular_patch(layout,(.25,.25),.2,max_faces=4)
        skew=make_layout(Plate(),(10,10,1),4,np.array([[1.,.1,0],[0,1,0],[0,0,1]]))
        with self.assertRaises(ValueError):circular_patch(skew,(.1,.1),.003)


if __name__=='__main__':unittest.main()
