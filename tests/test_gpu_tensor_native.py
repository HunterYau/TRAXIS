"""Explicitly enabled native/CUDA checks for sharing across all mesh axes."""
import json
import os
from pathlib import Path
import unittest
import numpy as np
from examples.tensor_reference import native_operators
from traxis.config import Plate,Material
from traxis.tensor_hex import make_grid
from traxis.gpu_tensor import make_layout,build_gpu_tensor
from traxis.elmer_plate_refinement import reference_source_hashes
from traxis.resources import ResourceGuard,ResourceLimits
from traxis.storage import canonical_hash,sha256_file,write_json


@unittest.skipUnless(os.environ.get('TRAXIS_GPU_CHECKS')=='1' and os.environ.get('TRAXIS_HIGHORDER_CHECKS')=='1',
                     'explicit native/CUDA checks not enabled')
class GPUTensorSharedNativeTests(unittest.TestCase):
    def check_grid(self,cells,order,transform,name):
        guard=ResourceGuard(ResourceLimits(max_seconds=90,max_rss_mib=1024,max_system_memory_percent=85,min_available_mib=2048))
        material=Material();plate=Plate(width_m=.04,height_m=.03,thickness_m=.003)
        grid=make_grid(plate,cells,order,transform);layout=make_layout(plate,cells,order,transform)
        system=build_gpu_tensor(layout,material,guard=guard)
        native,_,overlap=native_operators(grid,material,guard)
        k=native[2];basis=np.eye(len(k));gpu=np.column_stack([system.stiffness@basis[:,i] for i in range(len(k))])
        error=float(np.linalg.norm(gpu-k)/np.linalg.norm(k))
        mass_error=float(np.linalg.norm(system.mass_diagonal-np.diag(native[3]))/np.linalg.norm(system.mass_diagonal))
        self.assertLess(error,1e-10);self.assertLess(mass_error,1e-10)
        root=Path(os.environ['TMP']);path=root/(name+'.npz')
        np.savez_compressed(path,native_stiffness=k,gpu_stiffness=gpu,mass_diagonal=system.mass_diagonal)
        report=dict(cells=list(cells),order=order,transform=np.asarray(transform).tolist(),vector_dofs=len(k),
            all_column_native_relative_l2=error,mass_relative_l2=mass_error,shared_transform_difference=overlap,
            source_sha256=reference_source_hashes(),test_sha256=sha256_file(__file__),
            native_driver_sha256=sha256_file('examples/tensor_reference.py'),archive_sha256=sha256_file(path))
        report['fingerprint']=canonical_hash(report);write_json(root/(name+'.json'),report)

    def test_q3_sharing_across_x_y_z(self):
        self.check_grid((2,2,2),3,np.eye(3),'shared-xyz-q3')

    def test_q4_skewed_sharing_across_y_z(self):
        self.check_grid((1,2,2),4,np.array([[1,.13,.07],[.04,1,.11],[.02,.03,1.]]),'shared-yz-q4')


if __name__=='__main__':unittest.main()
