"""Analytic P1 triangle integrals over a circular mechanical footprint.

Green's theorem reduces area and first moments to straight-edge/circle-arc
integrals. Linear shape functions then need only those three moments.
This removes triangle-centroid inclusion error without changing the FEM mesh.
"""

import math

import numpy as np

from traxis.config import positive


def _cross(a, b):
    return float(a[0] * b[1] - a[1] * b[0])


def _unit_disk_moments(vertices):
    """Integrate (1,x,y) over triangle intersection with the unit disk."""
    result = np.zeros(3)
    orientation = _cross(vertices[1] - vertices[0], vertices[2] - vertices[0])
    scale = max(float(np.linalg.norm(vertices[(i + 1) % 3] - vertices[i])) for i in range(3))
    if abs(orientation) <= 1e-14 * scale**2:
        raise ValueError("degenerate surface triangle")
    for a, b in zip(vertices, np.roll(vertices, -1, axis=0)):
        delta = b - a
        aa, bb, cc = float(delta @ delta), float(a @ delta), float(a @ a - 1)
        discriminant = bb**2 - aa * cc
        tolerance = 64 * np.finfo(float).eps * max(bb**2, abs(aa * cc), 1.)
        cuts = [0., 1.]
        if discriminant >= -tolerance:
            root = math.sqrt(max(0., discriminant))
            cuts += [t for t in ((-bb - root) / aa, (-bb + root) / aa) if 0 < t < 1]
        cuts = sorted(set(cuts))
        for lo, hi in zip(cuts, cuts[1:]):
            p, q = a + lo * delta, a + hi * delta
            middle = (p + q) / 2
            if float(middle @ middle) <= 1 + 64 * np.finfo(float).eps:
                cross = _cross(p, q)
                result[0] += cross / 2
                result[1:] += cross * (p + q) / 6
            else:
                p, q = p / np.linalg.norm(p), q / np.linalg.norm(q)
                angle = math.atan2(_cross(p, q), float(p @ q))
                result += [angle / 2, (q[1] - p[1]) / 3, (p[0] - q[0]) / 3]
    return result if orientation > 0 else -result


def disk_triangle_shape_integrals(vertices_xy_m, center_xy_m, radius_m):
    """Return three nonnegative integrals of nodal P1 shape functions, in m².

    The footprint can cut any triangle edge, lie entirely inside a triangle,
    cover the triangle, or have zero intersection. Vertex order is arbitrary.
    Coordinates are normalized about the disk to avoid small SI-area scales.
    """
    vertices, center = np.asarray(vertices_xy_m, dtype=float), np.asarray(center_xy_m, dtype=float)
    positive("patch radius", radius_m)
    if vertices.shape != (3, 2) or center.shape != (2,) or not np.isfinite(vertices).all() or not np.isfinite(center).all():
        raise ValueError("expected finite triangle (3,2) and center (2,)")
    normalized = (vertices - center) / radius_m
    moments = _unit_disk_moments(normalized)
    if moments[0] < -1e-12:
        raise ArithmeticError("negative circle/triangle intersection area")
    if moments[0] <= 1e-14:
        return np.zeros(3)
    integrals = np.linalg.solve(np.column_stack([np.ones(3), normalized]).T, moments)
    if integrals.min() < -1e-10 * moments[0] - 1e-13:
        raise ArithmeticError("negative circle/triangle shape integral")
    integrals = np.maximum(integrals, 0)
    integrals *= moments[0] / integrals.sum()
    return integrals * radius_m**2
