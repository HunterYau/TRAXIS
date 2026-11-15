"""Historical source identity checks use synthetic archives, not physics results."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import warnings
import zipfile

from examples.embedded_wave_evidence import COMPATIBILITY, check_integrator_transition, check_snapshot, check_source_review
from examples.tensor_wave import PROJECT, sealed
from traxis.storage import sha256_file, write_json


class SourceSnapshotTests(unittest.TestCase):
    def fixture(self, root, change=None):
        source={'traxis/a.py':b'original source','examples/driver.py':b'driver',
                'examples/check.py':b'checker','case.json':b'{"steps": 100}'}
        plan=dict(source_sha256={'a.py':hashlib.sha256(source['traxis/a.py']).hexdigest()},
            driver_sha256={'driver.py':hashlib.sha256(source['examples/driver.py']).hexdigest()},
            analysis_sources={'examples/check.py':hashlib.sha256(source['examples/check.py']).hexdigest()},
            case=dict(steps=100))
        if change:change(source)
        path=root/'snapshot.zip'
        with zipfile.ZipFile(path,'w') as archive:
            for name,data in source.items():archive.writestr(name,data)
        plan['source_snapshot_sha256']=sha256_file(path)
        return plan,path

    def test_original_bytes_and_case_are_required_even_when_outer_hash_is_rebound(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);plan,path=self.fixture(root);check_snapshot(plan,path)
            for name,data in [('traxis/a.py',b'changed'),('case.json',b'{"steps": 99}')]:
                plan,path=self.fixture(root,lambda files:files.__setitem__(name,data))
                with self.assertRaises(ValueError):check_snapshot(plan,path)

    def test_duplicate_extra_missing_and_large_members_reject(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for change in (lambda f:f.update(extra=b'x'),lambda f:f.pop('traxis/a.py'),
                           lambda f:f.update({'traxis/a.py':b'x'*(4*1024**2+1)})):
                plan,path=self.fixture(root,change)
                with self.assertRaises(ValueError):check_snapshot(plan,path)
            plan,path=self.fixture(root)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore',UserWarning)
                with zipfile.ZipFile(path,'a') as archive:archive.writestr('traxis/a.py',b'original source')
            plan['source_snapshot_sha256']=sha256_file(path)
            with self.assertRaises(ValueError):check_snapshot(plan,path)


class SourceReviewTests(unittest.TestCase):
    def test_only_reviewed_old_and_current_pairs_pass(self):
        plan=dict(source_sha256={'solver.py':'old'},driver_sha256={'run.py':'same'},analysis_sources={'check.py':'same'})
        current=dict(source_sha256={'solver.py':'new'},driver_sha256={'run.py':'same'})
        review=dict(kind='embedded_wave_source_review_v1',files={'traxis/solver.py':
            dict(current_sha256='new',historical_sha256=['old'],reason='explicit parity check')})
        rows=check_source_review(plan,current,{'check.py':'same'},review)
        self.assertEqual(len(rows),1);self.assertEqual(rows[0]['run_sha256'],'old')
        for old,new in [('unreviewed','new'),('old','unreviewed')]:
            changed=deepcopy(plan);changed['source_sha256']['solver.py']=old
            updated=deepcopy(current);updated['source_sha256']['solver.py']=new
            with self.assertRaises(ValueError):check_source_review(changed,updated,{'check.py':'same'},review)
        changed=deepcopy(plan);changed['source_sha256']['extra.py']='other'
        with self.assertRaises(ValueError):check_source_review(changed,current,{'check.py':'same'},review)

    def test_integrator_changes_require_a_complete_matching_parity_receipt(self):
        current=sha256_file(PROJECT/'traxis/gpu_enriched_verlet.py')
        changes=[dict(reviewed_changes=[dict(path='traxis/gpu_enriched_verlet.py',run_sha256='old',current_sha256=current)])]
        self.assertIsNone(check_integrator_transition([],None))
        with self.assertRaises(ValueError):check_integrator_transition(changes,None)
        errors={label+'_'+name:0. for label in ('dense','sparse') for name in ('time_s','displacement_m','energy_time_s',
            'physical_energy_j','modified_energy_j','final_displacement_m','final_velocity_m_s')}
        report=dict(completed=True,passed=True,ready_for_training_data=False,training_updates=0,
            checker_sha256=sha256_file(PROJECT/'examples/check_embedded_source_transition.py'),
            source_validation=dict(review_sha256=sha256_file(COMPATIBILITY),
                validator_sha256=sha256_file(PROJECT/'examples/embedded_wave_evidence.py')),
            archived_integrator_sha256='old',current_integrator_sha256=current,
            cases=[dict(nonzero_initial_state=initialized,passed=True,maximum_absolute_errors=errors.copy()) for initialized in (False,True)])
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'check.json';write_json(path,sealed(deepcopy(report)))
            self.assertEqual(check_integrator_transition(changes,path)['sha256'],sha256_file(path))
            for change in (lambda r:r.update(checker_sha256='changed'),
                           lambda r:r.update(current_integrator_sha256='changed'),
                           lambda r:r['cases'][0]['maximum_absolute_errors'].update(dense_displacement_m=1e-4),
                           lambda r:r['cases'][1]['maximum_absolute_errors'].pop('sparse_displacement_m')):
                changed=deepcopy(report);change(changed);write_json(path,sealed(changed),overwrite=True)
                with self.assertRaises(ValueError):check_integrator_transition(changes,path)


if __name__=='__main__':unittest.main()
