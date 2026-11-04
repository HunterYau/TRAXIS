"""Independent infinite-plate dispersion check for the 3D tetrahedral FEM.

Bloch-periodic x and periodic y remove finite edges solely for this analytic
benchmark. These conditions are not used in finite physical-plate simulations.
"""

import math

import numpy as np
from scipy import sparse
from scipy.linalg import eigh

from traxis.config import Material, Plate, positive
from traxis.fem import assemble
from traxis.geometry import structured_mesh
from traxis.resources import ResourceGuard


def lamb_frequencies_hz(wavenumber_rad_m: float, thickness_m: float,
                        elements_per_wavelength: int = 16, thickness_cells: int = 8,
                        material: Material = Material(), guard: ResourceGuard | None = None) -> tuple[float, float]:
    """Return lowest two sagittal modes at a prescribed real wavenumber.

    A one-cell periodic unit is condensed with complex phase exp(i*k*x).
    Suppressing y displacement gives the Lamb (x,z) sector; it excludes SH0.
    Free top/bottom surfaces are unchanged. Refining x and z independently is
    still required for a full solver convergence study.
    """
    positive("wavenumber", wavenumber_rad_m)
    positive("thickness", thickness_m)
    if type(elements_per_wavelength) is not int or not 4 <= elements_per_wavelength <= 256:
        raise ValueError("elements per wavelength must be an integer in [4,256]")
    if type(thickness_cells) is not int or not 2 <= thickness_cells <= 128:
        raise ValueError("thickness cells must be an integer in [2,128]")
    dx = 2 * math.pi / wavenumber_rad_m / elements_per_wavelength
    system = assemble(structured_mesh(Plate(dx, dx, thickness_m), (1, 1, thickness_cells)), material, guard)
    xyz = system.mesh.points_m
    nz = thickness_cells + 1
    rows, cols, phases = [], [], []
    for node, (x, _y, z) in enumerate(xyz):
        layer = int(round(z / thickness_m * thickness_cells))
        for component, axis in enumerate((0, 2)):
            rows.append(3 * node + axis)
            cols.append(2 * layer + component)
            phases.append(np.exp(1j * wavenumber_rad_m * x))
    p = sparse.coo_matrix((phases, (rows, cols)), shape=(system.ndof, 2 * nz)).tocsr()
    kh = (p.conj().T @ system.stiffness @ p).toarray()
    mh = (p.conj().T @ sparse.diags(system.mass_kg) @ p).toarray()
    eigenvalues = eigh(kh, mh, eigvals_only=True)
    if eigenvalues[0] <= 0 or not np.isfinite(eigenvalues).all():
        raise ArithmeticError("invalid Bloch guided-mode eigenvalues")
    frequencies = np.sqrt(eigenvalues[:2]) / (2 * math.pi)
    if guard:
        guard.check(force=True)
    return float(frequencies[0]), float(frequencies[1])
