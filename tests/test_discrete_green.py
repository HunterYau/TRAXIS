"""Independent central-difference checks of the force indexing and convolution."""
import unittest
from unittest.mock import patch
from copy import deepcopy
from pathlib import Path
import tempfile

import numpy as np

from examples.discrete_green import synthesize
from examples.verify_green_synthesis import reference_window
from examples.tensor_wave import sealed
from traxis.storage import sha256_file, write_json


def direct(force):
    dt=.01;mass=np.array([1.,1.4,.8])
    stiffness=np.array([[5.,-1.,.3],[-1.,4.,-.5],[.3,-.5,3.]])
    load=np.array([1.,-.2,.4]);probes=np.array([[1.,0.,.5],[.2,-.3,.7]])
    u=np.zeros(3);previous=np.zeros(3);answer=[probes@u,probes@u]
    for step in range(1,len(force)-1):
        following=2*u-previous+dt**2*(load*force[step]-stiffness@u)/mass
        previous,u=u,following;answer.append(probes@u)
    return np.asarray(answer)


class DiscreteGreenTests(unittest.TestCase):
    def test_impulse_index_and_first_forced_sample(self):
        force=np.zeros(257);force[1]=1.;green=direct(force)
        restored,stats=synthesize(green,force)
        np.testing.assert_allclose(restored,green,rtol=1e-12,atol=1e-16)
        doubled=force*2.;result,_=synthesize(green,doubled)
        np.testing.assert_allclose(result[2],2*green[2],rtol=1e-12,atol=1e-16)
        self.assertEqual(stats['impulse_force_index'],1)
        np.testing.assert_array_equal(result[:2],0.)

    def test_general_force_matches_an_independent_matrix_recurrence(self):
        impulse=np.zeros(257);impulse[1]=1.;green=direct(impulse)
        rng=np.random.default_rng(20261115)
        forces=[np.sin(np.arange(257)*.17),rng.normal(size=257)]
        for force in forces:
            force[0]=0.;answer,_=synthesize(green,force);reference=direct(force)
            self.assertLess(np.linalg.norm(answer-reference)/np.linalg.norm(reference),1e-12)

    def test_last_force_cannot_change_displacement_at_the_same_sample(self):
        impulse=np.zeros(65);impulse[1]=1.;green=direct(impulse)
        force=np.zeros(65);force[-1]=2.;answer,_=synthesize(green,force)
        np.testing.assert_allclose(answer,0.,atol=1e-16)

    def test_invalid_initial_conditions_shapes_and_values_reject(self):
        impulse=np.zeros(65);impulse[1]=1.;green=direct(impulse)
        for g,f in [(green,impulse+1),(green+1,impulse),(green[:-1],impulse),
                    (green.astype(complex),impulse),(green,np.full(65,np.nan)),
                    (green,impulse.astype(bool))]:
            with self.assertRaises(ValueError):synthesize(g,f)

    def test_workspace_rejects_before_starting_an_fft(self):
        impulse=np.zeros(65);impulse[1]=1.;green=direct(impulse)
        with patch('examples.discrete_green.rfft',side_effect=AssertionError('FFT allocation forbidden')):
            with self.assertRaises(MemoryError):synthesize(green,impulse,max_workspace_mib=1e-6)


class GreenReferenceReceiptTests(unittest.TestCase):
    """Synthetic metadata checks; forward physics is covered by the plate driver."""
    def test_only_the_cpu_receipt_bound_to_the_retained_wave_is_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            project=Path(directory);(project/'examples').mkdir()
            checker=project/'examples/verify_embedded_window.py';checker.write_bytes(b'fixture checker')
            root=project/'outputs/wave';cpu_root=project/'outputs/cpu';cpu_root.mkdir(parents=True)
            level=dict(plan=dict(fingerprint='plan',source_sha256={'solver.py':'source'}),
                parent=dict(fingerprint='forward'),input_identities={'raw.npz':'raw-hash'})
            cpu=dict(completed=True,passed=True,ready_for_training_data=False,training_updates=0,
                plan_fingerprint='plan',forward_result_fingerprint='forward',input_identities=level['input_identities'],
                source_sha256=level['plan']['source_sha256'],checker_sha256=sha256_file(checker))
            path=cpu_root/'cpu-window-check.json';write_json(path,sealed(deepcopy(cpu)))
            with patch('examples.verify_green_synthesis.PROJECT',project), \
                    patch('examples.verify_green_synthesis.read_wave',return_value=level), \
                    patch('examples.verify_green_synthesis.validate_saved_plan',return_value={}):
                _,binding=reference_window(root,cpu_root,None)
                self.assertEqual(binding['cpu_sha256'],sha256_file(path))
                for key,value in [('completed',False),('passed',False),('ready_for_training_data',True),
                                  ('training_updates',1),('plan_fingerprint','other'),
                                  ('forward_result_fingerprint','other'),('input_identities',{}),
                                  ('source_sha256',{}),('checker_sha256','other')]:
                    changed=deepcopy(cpu);changed[key]=value;write_json(path,sealed(changed),overwrite=True)
                    with self.assertRaises(ValueError):reference_window(root,cpu_root,None)

    def test_reference_paths_must_be_inside_the_project_output_tree(self):
        with tempfile.TemporaryDirectory() as directory:
            project=Path(directory)
            with patch('examples.verify_green_synthesis.PROJECT',project):
                with self.assertRaises(ValueError):reference_window(project/'elsewhere',project/'outputs/cpu',None)


if __name__=='__main__':unittest.main()
