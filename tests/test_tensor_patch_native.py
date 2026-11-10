"""Native shape-by-shape verification of piecewise tensor disk covectors."""
import os
from pathlib import Path
import unittest
import numpy as np
from examples.tensor_patch_reference import native_circular_covector
from traxis.config import Plate
from traxis.tensor_hex import make_grid
from traxis.gpu_tensor import make_layout
from traxis.tensor_patches import circular_patch
from traxis.elmer_plate_refinement import reference_source_hashes
from traxis.resources import ResourceGuard,ResourceLimits
from traxis.storage import canonical_hash,sha256_file,write_json


@unittest.skipUnless(os.environ.get('TRAXIS_HIGHORDER_CHECKS')=='1','explicit native footprint checks not enabled')
class TensorPatchNativeTests(unittest.TestCase):
    def test_all_piecewise_covectors_against_native_shapes(self):
        guard=ResourceGuard(ResourceLimits(max_seconds=90,max_rss_mib=768,max_system_memory_percent=85,min_available_mib=2048))
        plate=Plate(width_m=.04,height_m=.03,thickness_m=.003);records=[];arrays={}
        cases=[(3,(2,2,1),(.02,.015)),(4,(2,2,1),(.0173,.0121)),
               (5,(2,1,1),(.02,.015)),(5,(1,2,1),(.02,.015)),(5,(1,1,1),(.0173,.0121))]
        for index,(order,cells,center) in enumerate(cases):
            grid=make_grid(plate,cells,order);layout=make_layout(plate,cells,order)
            patch=circular_patch(layout,center,.003,guard=guard);weights=np.zeros(len(grid.points_m));weights[patch.indices]=patch.weights
            native,details=native_circular_covector(grid,center,.003,guard)
            error=float(np.linalg.norm(weights-native)/np.linalg.norm(native));self.assertLess(error,1e-9)
            records.append(dict(order=order,cells=list(cells),center_xy_m=list(center),relative_l2=error,
                native_evaluations=details['evaluations'],patch_report=patch.report))
            arrays[f'case{index}_native']=native;arrays[f'case{index}_tensor']=weights
            arrays[f'case{index}_transform']=details['transform'];arrays[f'case{index}_native_coefficients']=details['native_coefficients']
        output=Path(os.environ['TMP']);np.savez_compressed(output/'native-circle-proofs.npz',**arrays)
        report=dict(cases=records,source_sha256=reference_source_hashes(),test_sha256=sha256_file(__file__),
            native_driver_sha256=sha256_file('examples/tensor_patch_reference.py'),
            archive_sha256=sha256_file(output/'native-circle-proofs.npz'))
        report['fingerprint']=canonical_hash(report);write_json(output/'native-circle-report.json',report)


if __name__=='__main__':unittest.main()
