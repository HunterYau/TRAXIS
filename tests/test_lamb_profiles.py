"""Independent field equations, free tractions and profile-only selection."""
import unittest
import copy
import json
from pathlib import Path
import numpy as np
from scipy.linalg import eigh
from traxis.lamb_profiles import fundamental_profile,spectral_support
from examples.tensor_profile_study import burst_transform,burst_band,checked_configuration
from traxis.fem import tone_burst
from scipy.integrate import quad


class LambProfileTests(unittest.TestCase):
    def test_profiles_satisfy_navier_free_tractions_and_physical_parity(self):
        for thickness in (.001,.003):
            for frequency in (5000.,25000.,100000.,200000.,250000.):
                for mode,sign in (('A0',-1),('S0',1)):
                    profile = fundamental_profile(frequency,thickness,mode)
                    points = np.linspace(-thickness/2,thickness/2,37)
                    u,_,_ = profile.fields(points)
                    np.testing.assert_allclose(u[0,::-1],sign*u[0],atol=1e-12)
                    np.testing.assert_allclose(u[2,::-1],-sign*u[2],atol=1e-12)
                    self.assertLess(profile.navier_relative_residual(points),1e-9)
                    mu = profile.material.lame[1]
                    traction = profile.traction(np.array([-thickness/2,thickness/2]))
                    self.assertLess(np.linalg.norm(traction)/(mu*profile.wavenumber_rad_m*np.linalg.norm(u)),1e-8)

    def test_analytic_derivatives_agree_with_independent_finite_differences(self):
        for mode in ('A0','S0'):
            profile = fundamental_profile(150000.,.003,mode)
            points = np.linspace(-.001,.001,11)
            step = 3e-6
            u,du,ddu = profile.fields(points)
            left = profile.fields(points-step)[0];right = profile.fields(points+step)[0]
            left2 = profile.fields(points-2*step)[0];right2 = profile.fields(points+2*step)[0]
            first = (left2-8*left+8*right-right2)/(12*step)
            second = (-right2+16*right-30*u+16*left-left2)/(12*step**2)
            self.assertLess(np.linalg.norm(first-du)/np.linalg.norm(du),1e-7)
            self.assertLess(np.linalg.norm(second-ddu)/np.linalg.norm(ddu),1e-6)

    def test_selection_uses_mass_profile_and_handles_split_or_permuted_modes(self):
        mass = np.diag([2.,3.,5.])
        values,vectors = eigh(np.diag([2.,6.,100.]),mass)
        reference = vectors@np.sqrt([.48,.515,.005])
        selected,weights = spectral_support(vectors,mass,reference,.99)
        self.assertEqual(set(selected),{0,1})
        np.testing.assert_allclose(weights,[.48,.515,.005],atol=1e-14)
        permutation = [2,0,1]
        selected2,weights2 = spectral_support(vectors[:,permutation],mass,reference,.99)
        self.assertEqual(set(np.array(permutation)[selected2]),{0,1})
        np.testing.assert_allclose(weights2,weights[permutation],atol=1e-14)
        # One close-frequency or strongest-overlap mode alone is insufficient.
        self.assertGreaterEqual(weights[selected].sum(),.99)

    def test_invalid_profiles_and_incomplete_modal_norm_are_rejected(self):
        with self.assertRaises(ValueError):
            fundamental_profile(100000.,.003,'SH0')
        with self.assertRaises(ValueError):
            fundamental_profile(100000.,.003,'A0').fields(np.array([.002]))
        with self.assertRaises(ArithmeticError):
            spectral_support(np.eye(2)*.9,np.eye(2),np.ones(2))

    def test_burst_transform_matches_existing_force_and_parseval_energy(self):
        duration = 2/100000.
        for frequency in (0.,5000.,100000.,250000.):
            direct = quad(lambda t:tone_burst(t,100000.,2)*np.cos(2*np.pi*frequency*t),0,duration,epsabs=1e-15)[0]
            direct -= 1j*quad(lambda t:tone_burst(t,100000.,2)*np.sin(2*np.pi*frequency*t),0,duration,epsabs=1e-15)[0]
            self.assertLess(abs(burst_transform(frequency)-direct),1e-14)
        whole = quad(lambda f:float(abs(burst_transform(f))**2),0,2000000,epsabs=1e-15,limit=300)[0]
        time_energy = quad(lambda t:tone_burst(t,100000.,2)**2,0,duration,epsabs=1e-15)[0]
        self.assertLess(abs(whole/(time_energy/2)-1),1e-6)
        band = burst_band(dict(carrier_hz=100000.,cycles=2,band_hz=[5000,250000],minimum_force_spectral_energy_fraction=.999))
        self.assertGreater(band['retained_force_energy_fraction'],.999)

    def test_profile_protocol_cannot_omit_cases_or_relax_coverage(self):
        config=json.loads(Path('configs/tensor-profile-study.json').read_text())
        self.assertEqual(checked_configuration(config),config)
        for key,value in (('frequencies_hz',[]),('profile_mass_coverage',.9),
                          ('maximum_support_relative_frequency_error',.05)):
            changed=copy.deepcopy(config);changed[key]=value
            with self.assertRaises(ValueError):
                checked_configuration(changed)


if __name__=='__main__':
    unittest.main()
