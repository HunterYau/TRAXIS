import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch
import os

from examples.read_live_progress import open_shared,read_shared_json
from traxis.storage import atomic_write


class LiveProgressTests(unittest.TestCase):
    def test_brief_reader_overlap_does_not_abort_the_atomic_writer(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'progress.json';path.write_bytes(b'{"step": 1}')
            attempted=threading.Event();failures=[];replace=os.replace
            def observed_replace(source,destination):
                try:return replace(source,destination)
                finally:attempted.set()
            def writer():
                try:atomic_write(path,lambda stream:stream.write(b'{"step": 2}'),overwrite=True)
                except BaseException as error:failures.append(error)
            thread=threading.Thread(target=writer)
            with patch('traxis.storage.os.replace',side_effect=observed_replace):
                try:
                    with open_shared(path) as reader:
                        thread.start();self.assertTrue(attempted.wait(2))
                        self.assertEqual(json.loads(reader.read()),{'step':1})
                finally:
                    if thread.ident is not None:thread.join(2)
            self.assertFalse(thread.is_alive());self.assertEqual(failures,[])
            self.assertEqual(read_shared_json(path),{'step':2})

    def test_bounded_reader_rejects_oversized_and_nonobject_json(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'progress.json';path.write_bytes(b'{"step": 1}')
            with self.assertRaises(MemoryError):read_shared_json(path,max_bytes=3)
            path.write_bytes(b'[]')
            with self.assertRaises(ValueError):read_shared_json(path)
            with self.assertRaises(ValueError):read_shared_json(path,max_bytes=True)

    def test_missing_observation_is_an_error_instead_of_a_completed_worker(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(OSError):read_shared_json(Path(directory)/'absent.json')


if __name__=='__main__':unittest.main()
