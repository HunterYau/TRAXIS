import unittest
from pathlib import Path
from unittest.mock import Mock,patch

import psutil

from examples.pace_gpu_worker import CoolingPacer,find_worker


class CoolingPacerTests(unittest.TestCase):
    def test_selection_matches_module_plan_and_live_leaf(self):
        plan=Path('outputs/pacing-selection/plan.json').resolve()
        parent,worker,other=(Mock(pid=i) for i in (101,102,103))
        for process,module,selected in ((parent,'examples.tensor_wave',plan),
                (worker,'examples.tensor_wave',plan),(other,'examples.enriched_gpu_wave',plan)):
            process.info={'cmdline':['python','-m',module,'--worker','--plan',str(selected)]}
            process.children.return_value=[]
        parent.children.return_value=[worker]
        with patch('examples.pace_gpu_worker.psutil.process_iter',return_value=[parent,worker,other]):
            self.assertIs(find_worker(plan,'examples.tensor_wave'),worker)
            self.assertIs(find_worker(plan),other)
        with self.assertRaises(ValueError):find_worker(plan,'unrelated.module')

    def test_selection_rejects_wrong_plan_and_ambiguous_workers(self):
        plan=Path('outputs/pacing-selection/plan.json').resolve()
        processes=[Mock(pid=i) for i in (101,102)]
        for process in processes:
            process.info={'cmdline':['python','-m','examples.tensor_wave','--worker','--plan',str(plan)]}
            process.children.return_value=[]
        with patch('examples.pace_gpu_worker.psutil.process_iter',return_value=processes):
            with self.assertRaises(ValueError):find_worker(plan,'examples.tensor_wave')
            with self.assertRaises(ValueError):find_worker(plan.with_name('other.json'),'examples.tensor_wave')

    def test_hysteresis_and_cleanup_resume_the_owned_worker(self):
        process = Mock()
        pacer = CoolingPacer(process)
        self.assertIsNone(pacer.update(77.))
        self.assertEqual(pacer.update(79.),'suspended')
        self.assertIsNone(pacer.update(76.))
        self.assertEqual(pacer.update(72.),'resumed')
        self.assertEqual(process.suspend.call_count,1)
        self.assertEqual(process.resume.call_count,1)
        pacer.update(78.)
        pacer.close()
        self.assertFalse(pacer.paused)
        self.assertEqual(process.resume.call_count,2)

    def test_failed_suspend_never_creates_a_false_paused_state(self):
        process = Mock()
        process.suspend.side_effect = psutil.AccessDenied(1)
        pacer = CoolingPacer(process)
        with self.assertRaises(psutil.AccessDenied):
            pacer.update(79.)
        self.assertFalse(pacer.paused)
        pacer.close()
        process.resume.assert_not_called()

    def test_cleanup_tolerates_a_worker_that_already_exited(self):
        process = Mock()
        pacer = CoolingPacer(process)
        pacer.update(79.)
        process.resume.side_effect = psutil.NoSuchProcess(1)
        pacer.close()
        self.assertFalse(pacer.paused)
