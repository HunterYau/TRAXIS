"""CPU-side geometry and independent batched action for CUDA tensor work."""
import unittest
import numpy as np
from traxis.config import Plate,Material
from traxis.tensor_hex import make_grid,assemble_tensor,element_matrices
from traxis.gpu_tensor import make_layout,mass_diagonal,streamed_action


class GPUTensorReferenceTests(unittest.TestCase):
    def test_layout_coordinates_indices_and_mass_match_small_reference(self):
        plate=Plate(width_m=.04,height_m=.03,thickness_m=.003);material=Material()
        for order in range(1,5):
            grid=make_grid(plate,(2,1,2),order)
            layout=make_layout(plate,(2,1,2),order)
            np.testing.assert_array_equal(layout.points(),grid.points_m)
            np.testing.assert_array_equal(layout.element_nodes(0,4),grid.element_nodes)
            expected=assemble_tensor(grid,material)
            actual=mass_diagonal(layout,material)
            self.assertLess(np.linalg.norm(actual-expected.mass_kg)/np.linalg.norm(expected.mass_kg),1e-13)
            self.assertAlmostEqual(actual[:len(actual)//3].sum()/(2700*.04*.03*.003),1.,places=13)

    def test_batched_streamed_action_matches_native_checked_assembly(self):
        plate=Plate(width_m=.04,height_m=.03,thickness_m=.003);material=Material()
        random=np.random.default_rng(823)
        for order in range(1,5):
            for transform in (np.eye(3),np.array([[1,.13,.07],[.04,1,.11],[.02,.03,1.]])):
                grid=make_grid(plate,(2,1,2),order,transform)
                layout=make_layout(plate,(2,1,2),order,transform)
                expected=assemble_tensor(grid,material)
                state=random.normal(size=expected.stiffness.shape[0]);reference=expected.stiffness@state
                for chunk in (1,4):
                    result=streamed_action(layout,material,state,chunk_elements=chunk)
                    self.assertLess(np.linalg.norm(result-reference)/np.linalg.norm(reference),1e-12)

    def test_local_stability_bound_covers_assembled_global_spectrum(self):
        plate=Plate(width_m=.04,height_m=.03,thickness_m=.003);material=Material()
        grid=make_grid(plate,(2,1,1),3);layout=make_layout(plate,(2,1,1),3)
        k,_,m=element_matrices(3,layout.jacobian,material,'gll')
        inverse=1/np.sqrt(np.tile(m,3));bound=np.max(inverse*(abs(k)@inverse))
        global_system=assemble_tensor(grid,material)
        inverse_global=1/np.sqrt(global_system.mass_kg)
        eigenvalues=np.linalg.eigvalsh(global_system.stiffness.toarray()*inverse_global[:,None]*inverse_global[None,:])
        self.assertLessEqual(eigenvalues[-1],bound*(1+1e-12))

    def test_full_size_counts_and_rejection_before_vector_allocation(self):
        layout=make_layout(Plate(),(150,150,1),4)
        self.assertEqual(layout.screen['scalar_dofs'],1806005)
        self.assertEqual(layout.screen['vector_dofs'],5418015)
        self.assertFalse(layout.screen['element_index_table_stored'])
        self.assertLess(layout.screen['device_storage_screen_mib'],1024)
        with self.assertRaises(MemoryError):
            make_layout(Plate(),(512,512,512),4)
        with self.assertRaises(MemoryError):
            make_layout(Plate(),(150,150,1),4,max_device_mib=100)


if __name__=='__main__':
    unittest.main()
