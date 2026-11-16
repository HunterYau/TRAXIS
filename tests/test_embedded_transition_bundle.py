"""Synthetic multi-version receipts test proof binding, not physical parity."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

from examples.embedded_wave_evidence import COMPATIBILITY,check_integrator_transition
from examples.tensor_wave import PROJECT,sealed
from traxis.storage import sha256_file,write_json


class TransitionBundleTests(unittest.TestCase):
    def fixture(self,root,mutate=None):
        current=sha256_file(PROJECT/'traxis/gpu_enriched_verlet.py')
        changes=[dict(reviewed_changes=[dict(path='traxis/gpu_enriched_verlet.py',
            run_sha256=old,current_sha256=current) for old in ('old-a','old-b')])]
        errors={label+'_'+name:0. for label in ('dense','sparse') for name in ('time_s','displacement_m',
            'energy_time_s','physical_energy_j','modified_energy_j','final_displacement_m','final_velocity_m_s')}
        proofs=[dict(completed=True,passed=True,ready_for_training_data=False,training_updates=0,
            checker_sha256=sha256_file(PROJECT/'examples/check_embedded_source_transition.py'),
            source_validation=dict(review_sha256=sha256_file(COMPATIBILITY),
                validator_sha256=sha256_file(PROJECT/'examples/embedded_wave_evidence.py')),
            archived_integrator_sha256=old,current_integrator_sha256=current,
            cases=[dict(nonzero_initial_state=b,passed=True,maximum_absolute_errors=errors.copy()) for b in (False,True)])
            for old in ('old-a','old-b')]
        if mutate:mutate(proofs)
        versions=[]
        for i,proof in enumerate(proofs):
            path=root/('version-'+str(i)+'.json');write_json(path,sealed(proof),overwrite=True)
            versions.append(dict(archived_integrator_sha256=proof['archived_integrator_sha256'],
                path=path.name,sha256=sha256_file(path),fingerprint=proof['fingerprint']))
        bundle=dict(kind='embedded_cuda_source_transitions',completed=True,passed=True,
            ready_for_training_data=False,training_updates=0,current_integrator_sha256=current,
            checker_sha256=sha256_file(PROJECT/'examples/check_embedded_source_transitions.py'),versions=versions)
        path=root/'bundle.json';write_json(path,sealed(bundle),overwrite=True)
        return changes,path,bundle

    def test_every_archived_version_is_verified_and_a_subset_can_reuse_the_bundle(self):
        with tempfile.TemporaryDirectory() as directory:
            changes,path,_=self.fixture(Path(directory))
            self.assertEqual(check_integrator_transition(changes,path)['sha256'],sha256_file(path))
            subset=deepcopy(changes);subset[0]['reviewed_changes'].pop()
            self.assertEqual(check_integrator_transition(subset,path)['sha256'],sha256_file(path))

    def test_missing_duplicate_stale_and_unbound_versions_reject(self):
        with tempfile.TemporaryDirectory() as directory:
            for mutate in (lambda b:b['versions'].pop(),
                lambda b:b['versions'].append(deepcopy(b['versions'][0])),
                lambda b:b['versions'][1].update(sha256='changed'),
                lambda b:b['versions'][1].update(fingerprint='changed'),
                lambda b:b.update(checker_sha256='changed'),
                lambda b:b.update(current_integrator_sha256='changed')):
                changes,path,bundle=self.fixture(Path(directory));mutate(bundle)
                bundle.pop('fingerprint',None);write_json(path,sealed(bundle),overwrite=True)
                with self.assertRaises(ValueError):check_integrator_transition(changes,path)

    def test_every_child_keeps_all_original_dense_sparse_and_initial_state_checks(self):
        with tempfile.TemporaryDirectory() as directory:
            for mutate in (lambda p:p[1]['cases'][1]['maximum_absolute_errors'].update(sparse_displacement_m=1e-4),
                lambda p:p[1]['cases'][0]['maximum_absolute_errors'].pop('dense_physical_energy_j'),
                lambda p:p[1]['cases'].pop(),lambda p:p[1].update(passed=False),
                lambda p:p[1]['source_validation'].update(review_sha256='changed')):
                changes,path,_=self.fixture(Path(directory),mutate)
                with self.assertRaises(ValueError):check_integrator_transition(changes,path)

    def test_child_paths_must_stay_inside_the_bound_bundle(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);nested=root/'nested';nested.mkdir()
            changes,path,bundle=self.fixture(nested)
            outside=root/'outside.json';outside.write_bytes((nested/bundle['versions'][0]['path']).read_bytes())
            bundle['versions'][0]['path']='../outside.json'
            bundle.pop('fingerprint',None);write_json(path,sealed(bundle),overwrite=True)
            with self.assertRaises(ValueError):check_integrator_transition(changes,path)

    def test_a_bad_extra_version_cannot_hide_when_only_a_subset_is_needed(self):
        with tempfile.TemporaryDirectory() as directory:
            changes,path,_=self.fixture(Path(directory),lambda p:p[1].update(passed=False))
            changes[0]['reviewed_changes'].pop()
            with self.assertRaises(ValueError):check_integrator_transition(changes,path)


if __name__=='__main__':
    unittest.main()
