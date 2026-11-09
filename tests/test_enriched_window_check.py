import unittest

import numpy as np

from examples.verify_enriched_window import cpu_state_reference
from traxis.config import Material,Plate
from traxis.enriched_tet import assemble_enriched
from traxis.geometry import structured_mesh
from traxis.matrixfree import MatrixFreeSettings


class IndependentSavedStateTests(unittest.TestCase):
    def test_arbitrary_native_permutation_matches_direct_assembled_energies(self):
        mesh = structured_mesh(Plate(.04,.03,.003),(1,1,1))
        material = Material()
        model = assemble_enriched(mesh,material)
        n = model.ndof//3
        rng = np.random.default_rng(2026)
        mapping = rng.permutation(n)
        u,v = rng.normal(size=model.ndof)*1e-9,rng.normal(size=model.ndof)*1e-6
        permutation = np.concatenate([mapping+c*n for c in range(3)])
        native_u,native_v = u[permutation],v[permutation]
        dt = model.stable_dt_s/4
        reference = cpu_state_reference(mesh,material,MatrixFreeSettings(),mapping,native_u,native_v,dt)
        ku = model.stiffness@u
        physical = float((v@(model.mass_kg*v)+u@ku)/2)
        modified = float(physical-dt**2/8*(ku@(ku/model.mass_kg)))
        self.assertAlmostEqual(reference['physical_energy_j']/physical,1.,places=12)
        self.assertAlmostEqual(reference['modified_energy_j']/modified,1.,places=12)
        np.testing.assert_allclose(reference['scalar_mass_kg'],model.mass_kg[:n][mapping],rtol=1e-14,atol=0)

    def test_invalid_map_and_states_fail(self):
        mesh = structured_mesh(Plate(.04,.03,.003),(1,1,1))
        model = assemble_enriched(mesh,Material())
        n = model.ndof//3
        for mapping,u in ((np.zeros(n,dtype=int),np.zeros(3*n)),(np.arange(n),np.full(3*n,np.nan)),
                          (np.arange(n),np.zeros(3*n-1))):
            with self.assertRaises(ValueError):
                cpu_state_reference(mesh,Material(),MatrixFreeSettings(),mapping,u,np.zeros(3*n),1e-9)
