"""Grouping preserves historical files and resolves their old paths."""
import json
import argparse
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from examples import organize_outputs as organizer
from examples import output_locations
from examples.plate_outputs import process_owner
from traxis.storage import sha256_file


class OutputOrganizationTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.project=Path(self.temp.name)
        self.root=self.project/'outputs';self.root.mkdir()

    def test_grouping_keeps_payloads_and_resolves_old_input_paths(self):
        old=self.root/'experimental-validation-test'
        (old/'sources').mkdir(parents=True)
        (old/'sources/recording.mat').write_bytes(b'experiment recording')
        solver=self.root/'embedded-study';solver.mkdir()
        (solver/'result.json').write_text('{"completed":true}')
        simulation=self.root/'simulation-runs/run-001';simulation.mkdir(parents=True)
        (simulation/'result.json').write_text('{"keep":"this run"}')
        with patch.object(organizer,'PROJECT',self.project), \
                patch.object(output_locations,'PROJECT',self.project):
            manifest=organizer.organize(self.root,apply=True)
            resolved=output_locations.organized_output(old/'sources/recording.mat')
            self.assertEqual(resolved.read_bytes(),b'experiment recording')
            self.assertTrue(str(resolved).startswith(str(self.root/'experimental-references')))
            self.assertTrue((self.root/'solver-development/embedded/embedded-study/result.json').is_file())
            self.assertTrue((simulation/'result.json').is_file())
            self.assertTrue(all(e['verified'] for e in manifest['moves'].values()))
            self.assertEqual(organizer.organize(self.root,apply=True),manifest)

    def test_active_queue_refused_before_any_move(self):
        source=self.root/'experimental-validation-test';source.mkdir()
        (source/'queue-owner.json').write_text(json.dumps(process_owner()))
        with patch.object(organizer,'PROJECT',self.project):
            with self.assertRaisesRegex(RuntimeError,'live or unverified'):
                organizer.organize(self.root,apply=True)
        self.assertTrue(source.exists())
        self.assertFalse((self.root/'organization.json').exists())

    def test_existing_destination_is_preserved(self):
        source=self.root/'gpu-test';source.mkdir()
        target=self.root/'gpu-and-performance/gpu-test';target.mkdir(parents=True)
        (target/'keep.txt').write_text('existing output')
        with patch.object(organizer,'PROJECT',self.project):
            with self.assertRaises(FileExistsError):
                organizer.organize(self.root,apply=True)
        self.assertTrue(source.exists())
        self.assertEqual((target/'keep.txt').read_text(),'existing output')

    def test_verification_recovers_after_rename_before_manifest_completion(self):
        source=self.root/'pzt-test';source.mkdir()
        (source/'result.json').write_text('{"saved":true}')
        before=organizer.snapshot(source)
        destination=self.root/'pzt-development/pzt-test';destination.parent.mkdir()
        source.rename(destination)
        (self.root/'organization.json').write_text(json.dumps({'schema':1,'moves':{
            'pzt-test':{'destination':'pzt-development/pzt-test','verified':False,'inventory':before}}}))
        with patch.object(organizer,'PROJECT',self.project):
            result=organizer.organize(self.root,apply=True)
        self.assertTrue(result['moves']['pzt-test']['verified'])


class OutputDefaultTests(unittest.TestCase):
    def parse_without_running(self,argv):
        from traxis.cli import main
        class Parsed(Exception):
            pass
        original=argparse.ArgumentParser.parse_args
        captured=[]
        def intercept(parser,*args,**kwargs):
            captured.append(original(parser,*args,**kwargs))
            raise Parsed()
        with patch('sys.argv',['traxis',*argv]), \
                patch.object(argparse.ArgumentParser,'parse_args',intercept):
            with self.assertRaises(Parsed):
                main()
        return captured[0]

    def test_automatic_defaults_are_grouped_and_explicit_path_is_preserved(self):
        args=self.parse_without_running(['verify'])
        self.assertEqual(args.output,Path('outputs/test-infrastructure/verification'))
        args=self.parse_without_running(['verify-models'])
        self.assertEqual(args.output,Path('outputs/test-infrastructure/model-verification'))
        args=self.parse_without_running(['verify','--output','my-chosen-location'])
        self.assertEqual(args.output,Path('my-chosen-location'))

    def test_path_only_receipt_and_inherited_readiness_bind_current_sources(self):
        root=Path(__file__).resolve().parents[1]
        receipt=root/'docs/evidence/output-default-routing-20261119.json'
        report=json.loads(receipt.read_text())
        self.assertTrue(report['passed'])
        for name,change in report['source_changes'].items():
            self.assertEqual(sha256_file(root/name),change['after_sha256'])
            self.assertTrue(change['only_declared_output_paths_changed'])
        proof=json.loads((root/'docs/evidence/resolved-pzt-organized-output-readiness.json').read_text())
        inherited=proof['output_routing_only_update']
        self.assertEqual(sha256_file(root/inherited['parent_evidence']),inherited['parent_sha256'])
        self.assertEqual(proof['evidence_sha256'][receipt.relative_to(root).as_posix()],sha256_file(receipt))


if __name__=='__main__':
    unittest.main()
