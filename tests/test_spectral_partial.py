"""Small independent assembled references for tensor contractions/partial setup."""
import unittest
import numpy as np
from traxis.pzt_receiver import DiskGeometry
from traxis.spectral_piezoelectric import disk_mesh,assemble
from traxis.spectral_partial import TensorReference,component
from traxis.spectral_pzt_sparse import component as reference_component
from tests.test_quadratic_piezoelectric import material


class PartialTests(unittest.TestCase):
    def test_assembled_operators_and_reciprocity_on_curved_annular_cells(self):
        mat=material();rng=np.random.default_rng(93)
        for rings,layers in ((1,1),(2,2)):
            mesh=disk_mesh(DiskGeometry((.02,.02),.003,.0005,'top',1),side_cells=2,radial_cells=rings,layers=layers)
            t=TensorReference(mesh,mat);a=assemble(mesh,mat,quadrature_order=6);n=t.n
            u=rng.normal(size=3*n);phi=rng.normal(size=n);x=rng.normal(size=n)
            k,bt=t.mechanical(u);b,d=t.electrical(phi)
            for actual,expected in ((k,a.stiffness@u),(bt,a.coupling.T@u),
                    (b,a.coupling@phi),(d,a.dielectric@phi),(t.mass(x),a.mass[:n,:n]@x)):
                self.assertLess(np.linalg.norm(actual-expected)/np.linalg.norm(expected),1e-12)
            self.assertLess(abs(u@b-phi@bt)/max(abs(u@b),abs(phi@bt)),1e-12)

    def test_only_scalar_matrices_assembled_and_electrodes_preserved(self):
        geo=DiskGeometry((.02,.02),.003,.0005,'top',1);mat=material()
        old=reference_component(geo,mat,rings=1,sectors=8,layers=1)
        new=component(geo,mat,rings=1,sectors=8,layers=1)
        self.assertFalse(hasattr(new,'stiffness'));self.assertFalse(hasattr(new,'bi'))
        np.testing.assert_array_equal(new.mesh.hexahedra,old.mesh.hexahedra)
        np.testing.assert_array_equal(new.mesh.points_m,old.mesh.points_m)
        for x,y in ((new.scalar_mass.toarray(),old.scalar_mass.toarray()),(new.di.toarray(),old.di.toarray()),(new.g,old.g)):
            self.assertLess(np.linalg.norm(x-y)/np.linalg.norm(y),1e-12)
        self.assertLess(abs(new.capacitance_f/old.capacitance_f-1),1e-12)
        self.assertEqual(new.body_eigenvalue_bound,old.body_eigenvalue_bound)


if __name__=='__main__':unittest.main()
