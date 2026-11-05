"""Exact small P2 mass reference from simplex barycentric monomial integrals."""

import math
import numpy as np

from traxis.config import positive
from traxis.quadratic_patches import TETRAHEDRON_EDGES


def normalized_quadratic_mass():
    """Scalar ten-node tetrahedron mass divided by density and volume."""
    shapes = []
    for i in range(4):
        one, two = [0]*4, [0]*4
        one[i], two[i] = 1, 2
        shapes.append(((2., two), (-1., one)))
    for i, j in TETRAHEDRON_EDGES:
        powers = [0]*4
        powers[i], powers[j] = 1, 1
        shapes.append(((4., powers),))
    matrix = np.empty((10, 10))
    for i, a in enumerate(shapes):
        for j, b in enumerate(shapes):
            value = 0.
            for ca, pa in a:
                for cb, pb in b:
                    powers = [x+y for x, y in zip(pa, pb)]
                    value += ca*cb * 6 * math.prod(math.factorial(p) for p in powers) / math.factorial(sum(powers)+3)
            matrix[i,j] = value
    return matrix


def exact_quadratic_mass(points, tetrahedra, density):
    """Dense node-major reference, bounded to 128 nodes and straight P2 tets.

    Midpoints may appear in any order. Each is matched to its geometric edge
    before the exact local mass is assembled; no numerical quadrature is used.
    """
    positive('density', density)
    points, tets = np.asarray(points), np.asarray(tetrahedra)
    if (np.iscomplexobj(points) or points.ndim != 2 or points.shape[1] != 3 or not 10 <= len(points) <= 128
            or not np.isfinite(points).all() or tets.ndim != 2 or tets.shape[1] != 10 or tets.dtype.kind not in 'iu'
            or not 1 <= len(tets) <= 512 or tets.min() < 0 or tets.max() >= len(points)
            or len(np.unique(tets)) != len(points) or len(np.unique(np.sort(tets[:,:4], axis=1), axis=0)) != len(tets)):
        raise ValueError('exact P2 mass needs a bounded complete quadratic tetrahedron mesh')
    scalar = np.zeros((len(points), len(points)))
    normalized = normalized_quadratic_mass()
    for tet in tets:
        corners = points[tet[:4]]
        edges = corners[1:] - corners[:1]
        determinant = abs(float(np.linalg.det(edges)))
        scale = float(np.linalg.norm(edges, axis=1).max())
        if determinant <= 1e-12 * scale**3:
            raise ValueError('degenerate quadratic mass element')
        targets = corners[TETRAHEDRON_EDGES].mean(axis=1)
        distances = np.linalg.norm(targets[:,None] - points[tet[4:]][None], axis=2)
        matches = distances <= 1e-10 * scale
        if not ((matches.sum(axis=1) == 1).all() and (matches.sum(axis=0) == 1).all()):
            raise ValueError('quadratic mass reference requires straight unique edge midpoints')
        ids = np.r_[tet[:4], tet[4:][matches.argmax(axis=1)]]
        scalar[np.ix_(ids, ids)] += density * determinant / 6 * normalized
    return np.kron(scalar, np.eye(3))
