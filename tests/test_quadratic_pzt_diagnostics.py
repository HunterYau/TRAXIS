import json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace
from examples import quadratic_pzt_basewaves as study

class DiagnosticBudgetTests(unittest.TestCase):
    def test_explicit_node_limit_records_skip_before_assembly(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(study,'assemble') as assembly,patch.object(study.psutil,'virtual_memory',return_value=SimpleNamespace(available=10*2**30)):
            result=study.benchmark(1,8,1,tmp,max_nodes=1)
            self.assertEqual(result['status'],'not_run_resource_forecast')
            self.assertTrue((Path(tmp)/'r1-a8-t1.json').exists())
            assembly.assert_not_called()

    def test_invalid_measured_memory_reference_cannot_enable_assembly(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(study,'assemble') as assembly:
            p=Path(tmp)/'reference.json';p.write_text(json.dumps(dict(status='failed',nodes=100,sampled_peak_rss_mib=10)))
            with self.assertRaises(ValueError):study.benchmark(1,8,1,Path(tmp)/'out',memory_reference=p)
            assembly.assert_not_called()

if __name__=='__main__':unittest.main()
