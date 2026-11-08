"""Isolated native GPU CG contract check for the guarded regression suite."""

import argparse
from pathlib import Path
import unittest

import numpy as np
from scipy.sparse.linalg import spsolve

from traxis.config import Material,Plate
from traxis.geometry import structured_mesh
from traxis.gpu_cg import gpu_checked_cg
from traxis.gpu_elements import build_gpu_elements
from traxis.highorder import assemble_highorder
from traxis.iterative_fem import IterativeSettings
from traxis.matrixfree import MatrixFreeSettings
from traxis.resources import ResourceGuard,ResourceLimits
from traxis.storage import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    guard = ResourceGuard(ResourceLimits(max_seconds=45,max_rss_mib=1024))
    check = unittest.TestCase()
    mesh = structured_mesh(Plate(.04,.03,.003),(2,2,1))
    gpu = build_gpu_elements(mesh,Material(),MatrixFreeSettings(order=2),guard=guard)
    native = assemble_highorder(mesh,Material(),2,guard)
    rhs = np.random.default_rng(2026).normal(size=gpu.space.ndof)
    guess = np.full(gpu.space.ndof,.1)
    saved = guess.copy()
    result = gpu_checked_cg(gpu.mass,rhs,guess,gpu.mass_diagonal,
        IterativeSettings(max_iterations=500),guard=guard)
    direct = spsolve(native.mass,rhs)
    relative = float(np.linalg.norm(result.solution-direct)/np.linalg.norm(direct))
    check.assertLess(relative,1e-7)
    np.testing.assert_array_equal(guess,saved)
    cpu_residual = float(np.linalg.norm(native.mass@result.solution-rhs)/np.linalg.norm(rhs))
    check.assertLess(cpu_residual,1e-9)
    check.assertEqual(result.statistics['host_vector_uploads'],3)
    check.assertEqual(result.statistics['host_vector_downloads'],1)
    check.assertGreater(result.iterations,3)
    zero = gpu_checked_cg(gpu.mass,np.zeros_like(rhs),guess,gpu.mass_diagonal,guard=guard)
    np.testing.assert_array_equal(zero.solution,np.zeros_like(rhs))
    check.assertEqual(zero.iterations,0)
    events = []
    with check.assertRaisesRegex(ArithmeticError,'device CG solve failed'):
        gpu_checked_cg(gpu.mass,rhs,np.zeros_like(rhs),gpu.mass_diagonal,
            IterativeSettings(max_iterations=1),guard=guard,progress=events.append)
    check.assertEqual(events[-1]['status'],1)
    check.assertGreater(events[-1]['true_relative_residual'],1e-9)
    write_json(args.output,dict(passed=True,direct_relative_l2=relative,cpu_relative_residual=cpu_residual,
        preserved_guess=True,zero_rhs_exact=True,nonconvergence_rejected=True,
        solver_statistics=result.statistics,resources=guard.report()))


if __name__ == '__main__':
    main()
