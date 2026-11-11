"""Bounded Windows replacement recovery preserves atomic publication rules."""
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from traxis.storage import atomic_write


def permission_error(code):
    error=PermissionError('simulated native file replacement denial');error.winerror=code
    return error


class StorageRecoveryTests(unittest.TestCase):
    def test_transient_windows_sharing_error_recovers_without_rewriting_temp(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'progress.json';path.write_bytes(b'old');replace=os.replace
            attempts=[];writes=[]
            def interrupted(source,destination):
                attempts.append(Path(source))
                if len(attempts)<3:raise permission_error(32)
                return replace(source,destination)
            def writer(stream):writes.append(True);stream.write(b'complete new value')
            with patch('traxis.storage.os.replace',side_effect=interrupted):
                if os.name=='nt':atomic_write(path,writer,overwrite=True)
                else:
                    with self.assertRaises(PermissionError):atomic_write(path,writer,overwrite=True)
            self.assertEqual(len(writes),1)
            self.assertEqual(path.read_bytes(),b'complete new value' if os.name=='nt' else b'old')
            self.assertEqual(len(set(attempts)),1)
            self.assertEqual([p.name for p in Path(directory).iterdir()],['progress.json'])

    def test_permanent_denial_is_bounded_preserves_old_target_and_cleans_temp(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'progress.json';path.write_bytes(b'old')
            with patch('traxis.storage.os.replace',side_effect=permission_error(5)) as replace,patch('traxis.storage.time.sleep') as sleep:
                with self.assertRaises(PermissionError):atomic_write(path,lambda s:s.write(b'new'),overwrite=True)
            self.assertLessEqual(replace.call_count,10)
            self.assertLessEqual(sum(c.args[0] for c in sleep.call_args_list),.5)
            self.assertEqual(path.read_bytes(),b'old')
            self.assertEqual([p.name for p in Path(directory).iterdir()],['progress.json'])

    def test_unrelated_error_is_not_retried(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'progress.json';path.write_bytes(b'old')
            with patch('traxis.storage.os.replace',side_effect=permission_error(87)) as replace,patch('traxis.storage.time.sleep') as sleep:
                with self.assertRaises(PermissionError):atomic_write(path,lambda s:s.write(b'new'),overwrite=True)
            self.assertEqual(replace.call_count,1);sleep.assert_not_called();self.assertEqual(path.read_bytes(),b'old')

    def test_create_only_publication_never_replaces_an_existing_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'result.json';path.write_bytes(b'old')
            with patch('traxis.storage.os.replace') as replace:
                with self.assertRaises(FileExistsError):atomic_write(path,lambda s:s.write(b'new'))
            replace.assert_not_called();self.assertEqual(path.read_bytes(),b'old')


if __name__=='__main__':unittest.main()
