"""Work-conjugate circular mechanical patches on flat straight-sided P2 faces."""

from dataclasses import dataclass
import math
from pathlib import Path

import numpy as np

from traxis.config import Plate, positive
from traxis.patches import disk_triangle_quadratic_integrals


TETRAHEDRON_EDGES = np.array([[0, 1], [1, 2], [2, 0], [0, 3], [1, 3], [2, 3]])


@dataclass(frozen=True)
class QuadraticTetMesh:
    points_m: np.ndarray
    tetrahedra: np.ndarray
    plate: Plate

    def __post_init__(self):
        points, tets = np.asarray(self.points_m), np.asarray(self.tetrahedra)
        if (points.ndim != 2 or points.shape[1] != 3 or not 10 <= len(points) <= 20000
                or not np.isfinite(points).all() or tets.ndim != 2 or tets.shape[1] != 10
                or tets.dtype.kind not in 'iu' or not 1 <= len(tets) <= 40000
                or tets.min() < 0 or tets.max() >= len(points) or len(np.unique(tets)) != len(points)):
            raise ValueError('need a bounded, finite, complete 10-node tetrahedron mesh')
        if len(np.unique(np.sort(tets[:, :4], axis=1), axis=0)) != len(tets):
            raise ValueError('duplicate quadratic tetrahedron')
        corners = points[tets[:, :4]]
        edges = corners[:, 1:] - corners[:, :1]
        if np.any(abs(np.linalg.det(edges)) <= 1e-12 * np.linalg.norm(edges, axis=2).max(axis=1)**3):
            raise ValueError('degenerate quadratic tetrahedron corners')
        dimensions = np.array([self.plate.width_m, self.plate.height_m, self.plate.thickness_m])
        tolerance = dimensions.max() * 1e-10
        if (points < -tolerance).any() or (points > dimensions + tolerance).any():
            raise ValueError('quadratic mesh nodes escape the declared plate bounds')
        expected = corners[:, TETRAHEDRON_EDGES].mean(axis=2)
        middle = points[tets[:, 4:]]
        lengths = np.linalg.norm(corners[:, TETRAHEDRON_EDGES[:, 0]] - corners[:, TETRAHEDRON_EDGES[:, 1]], axis=2).max(axis=1)
        tolerance = 1e-10 * lengths + 32 * np.finfo(float).eps * np.maximum(abs(corners).max(axis=(1, 2)), lengths)
        matches = np.linalg.norm(expected[:, :, None] - middle[:, None], axis=3) <= tolerance[:, None, None]
        if not ((matches.sum(axis=2) == 1).all() and (matches.sum(axis=1) == 1).all()):
            raise ValueError('quadratic tetrahedra need unique straight-edge midpoints')
        mids = np.take_along_axis(tets[:, 4:], matches.argmax(axis=2), axis=1).ravel()
        edge_keys = np.sort(tets[:, :4][:, TETRAHEDRON_EDGES], axis=2).reshape(-1, 2)
        keys, inverse = np.unique(edge_keys, axis=0, return_inverse=True)
        low, high = np.full(len(keys), len(points)), np.full(len(keys), -1)
        np.minimum.at(low, inverse, mids)
        np.maximum.at(high, inverse, mids)
        if ((low != high).any() or len(np.unique(low)) != len(low)
                or len(np.intersect1d(low, tets[:, :4]))):
            raise ValueError('quadratic edges must share one conforming midpoint node')
        points, tets = points.astype(np.float64, copy=True), tets.astype(np.int64, copy=True)
        points.setflags(write=False)
        tets.setflags(write=False)
        object.__setattr__(self, 'points_m', points)
        object.__setattr__(self, 'tetrahedra', tets)


def load_quadratic_elmer_mesh(directory: Path, plate: Plate) -> QuadraticTetMesh:
    """Read serial native Elmer 510 elements, with SI coordinates declared by the caller."""
    directory = Path(directory)
    for name in ('mesh.header', 'mesh.nodes', 'mesh.elements'):
        if (directory / name).stat().st_size > 16 * 1024**2:
            raise MemoryError('quadratic mesh input exceeds 16 MiB')
    header = (directory / 'mesh.header').read_text(encoding='ascii').splitlines()[0].split()
    if len(header) != 3:
        raise ValueError('invalid quadratic mesh header')
    count, elements, boundary = map(int, header)
    if not 10 <= count <= 20000 or not 1 <= elements <= 40000 or boundary < 0:
        raise ValueError('quadratic mesh counts exceed local bounds')
    nodes = np.loadtxt(directory / 'mesh.nodes', ndmin=2)
    tets = np.loadtxt(directory / 'mesh.elements', ndmin=2)
    if (nodes.shape != (count, 5) or not np.isfinite(nodes).all()
            or not np.array_equal(nodes[:, 0], np.arange(1, count + 1))
            or tets.shape != (elements, 13) or not np.isfinite(tets).all()
            or not np.array_equal(tets, tets.astype(np.int64))
            or not np.array_equal(tets[:, 0], np.arange(1, elements + 1))
            or not (tets[:, 1] == 1).all() or not (tets[:, 2] == 510).all()):
        raise ValueError('expected sequential single-body Elmer quadratic tetrahedra')
    mesh = QuadraticTetMesh(nodes[:, 2:], tets[:, 3:].astype(np.int64) - 1, plate)
    expected = mesh.points_m[mesh.tetrahedra[:, :4][:, TETRAHEDRON_EDGES]].mean(axis=2)
    scale = np.linalg.norm(expected, axis=2).max(axis=1)
    if not np.all(np.linalg.norm(mesh.points_m[mesh.tetrahedra[:, 4:]] - expected, axis=2)
                  <= (1e-10 * max(plate.width_m, plate.height_m, plate.thickness_m) + 32 * np.finfo(float).eps * scale[:, None])):
        raise ValueError('Elmer 510 edge ordering disagrees with straight-sided geometry')
    return mesh


def quadratic_surface_patch(mesh: QuadraticTetMesh, center_xy_m, radius_m, direction=(0., 0., 1.)):
    """Unit-total-force load / same area-averaged displacement, in node-major order.

    Each top face must have three corner nodes and actual straight-edge
    midpoints. Midpoint IDs are found geometrically, allowing different
    tetrahedral edge orderings without guessing a solver-specific convention.
    """
    positive('patch radius', radius_m)
    center, direction = np.asarray(center_xy_m, float), np.asarray(direction, float)
    if center.shape != (2,) or not np.isfinite(center).all():
        raise ValueError('patch center must contain two finite coordinates')
    if direction.shape != (3,) or not np.isfinite(direction).all() or np.linalg.norm(direction) == 0:
        raise ValueError('patch direction must be finite and nonzero')
    direction = direction / np.linalg.norm(direction)
    corners = mesh.tetrahedra[:, :4]
    faces = corners[:, np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])].reshape(-1, 3)
    owners = np.repeat(np.arange(len(corners)), 4)
    top = np.isclose(mesh.points_m[:, 2], mesh.plate.thickness_m, rtol=0, atol=mesh.plate.thickness_m * 1e-10)
    selected = top[faces].all(axis=1)
    faces, owners = faces[selected], owners[selected]
    unique, first, counts = np.unique(np.sort(faces, axis=1), axis=0, return_index=True, return_counts=True)
    if (counts > 2).any():
        raise ValueError('nonmanifold top-surface topology')
    faces, owners = unique[counts == 1], owners[first[counts == 1]]
    if not len(faces):
        raise ValueError('quadratic mesh has no free flat top faces')
    xy = mesh.points_m[faces, :2]
    lower, upper = xy.min(axis=1) - center, xy.max(axis=1) - center
    nearest = np.maximum(np.maximum(lower, -upper), 0)
    selected = (nearest**2).sum(axis=1) <= radius_m**2
    weights = np.zeros(len(mesh.points_m))
    covered = 0
    for face, owner in zip(faces[selected], owners[selected]):
        vertices = mesh.points_m[face]
        mid_ids = mesh.tetrahedra[owner, 4:]
        length = max(np.linalg.norm(vertices[i] - vertices[j]) for i, j in ((0, 1), (1, 2), (2, 0)))
        tolerance = 1e-10 * length + 32 * np.finfo(float).eps * max(abs(vertices).max(), length)
        ids = list(face)
        for i, j in ((0, 1), (1, 2), (2, 0)):
            expected = (vertices[i] + vertices[j]) / 2
            matches = np.flatnonzero(np.linalg.norm(mesh.points_m[mid_ids] - expected, axis=1) <= tolerance)
            if len(matches) != 1:
                raise ValueError('top face needs unique straight-edge quadratic midpoints')
            ids.append(int(mid_ids[matches[0]]))
        integrals = disk_triangle_quadratic_integrals(vertices[:, :2], center, radius_m)
        weights[ids] += integrals
        covered += int(integrals.sum() > 0)
    area = float(weights.sum())
    if area <= 0:
        raise ValueError('patch has no positive-area intersection with the top surface')
    weights /= area
    relative = mesh.points_m[:, :2] - center
    centroid_offset = weights @ relative
    moments = weights @ np.column_stack([relative[:, 0]**2, relative[:, 0] * relative[:, 1], relative[:, 1]**2])
    clipped = bool(np.any(center < radius_m) or np.any(center > [mesh.plate.width_m - radius_m, mesh.plate.height_m - radius_m]))
    report = {'integration': 'analytic_disk_P2_shape_integrals', 'requested_center_xy_m': center.tolist(),
              'radius_m': radius_m, 'direction': direction.tolist(), 'area_m2': area,
              'centroid_xy_m': (center + centroid_offset).tolist(), 'centroid_error_m': float(np.linalg.norm(centroid_offset)),
              'centered_second_moments_m2': moments.tolist(),
              'full_disk_second_moment_error_m2': float(np.max(abs(moments - [radius_m**2 / 4, 0., radius_m**2 / 4]))) if not clipped else None,
              'full_disk_area_relative_error': abs(area / (math.pi * radius_m**2) - 1),
              'support_nodes': int(np.count_nonzero(weights)), 'negative_shape_coefficients': int((weights < 0).sum()),
              'minimum_normalized_shape_coefficient': float(weights.min()),
              'covered_surface_triangles': covered, 'edge_clipped': clipped}
    return (weights[:, None] * direction).ravel(), report
