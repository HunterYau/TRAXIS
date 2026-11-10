"""Fifth-degree extension for a full-plate polynomial-refinement control."""
import unittest
import numpy as np
from traxis.config import Material,Plate
from traxis.tensor_hex import line_rule,make_grid,assemble_tensor,element_matrices,local_gll_action
from traxis.gpu_tensor import make_layout,streamed_action,mass_diagonal


class TensorFifthDegreeTests(unittest.TestCase):
    def test_quadrature_and_independent_tensor_action(self):
        nodes,w,d,_,k,c=line_rule(5)
        for power in range(10):
            self.assertAlmostEqual(float(w@nodes**power),0. if power%2 else 2/(power+1),places=12)
        np.testing.assert_allclose(d.T@np.diag(w)@d,k,atol=3e-12)
        np.testing.assert_allclose(d.T@np.diag(w),c,atol=3e-12)
        material=Material();j=np.array([[1,.13,.07],[.04,1,.11],[.02,.03,1.]])@np.diag([.02,.015,.0015])
        matrix,_,_=element_matrices(5,j,material)
        state=np.random.default_rng(319).normal(size=(3,6,6,6));expected=matrix@state.ravel()
        self.assertLess(np.linalg.norm(local_gll_action(state,5,j,material).ravel()-expected)/np.linalg.norm(expected),1e-11)

    def test_shared_grid_and_full_candidate_counts(self):
        material=Material();plate=Plate(width_m=.04,height_m=.03,thickness_m=.003)
        grid=make_grid(plate,(2,1,1),5);system=assemble_tensor(grid,material);layout=make_layout(plate,(2,1,1),5)
        np.testing.assert_array_equal(grid.points_m,layout.points())
        self.assertLess(np.linalg.norm(mass_diagonal(layout,material)-system.mass_kg)/np.linalg.norm(system.mass_kg),1e-12)
        values=np.random.default_rng(37).normal(size=system.stiffness.shape[0]);expected=system.stiffness@values
        self.assertLess(np.linalg.norm(streamed_action(layout,material,values)-expected)/np.linalg.norm(expected),1e-11)
        self.assertEqual(make_layout(Plate(),(150,150,1),5).screen['vector_dofs'],10152018)


if __name__=='__main__':unittest.main()
