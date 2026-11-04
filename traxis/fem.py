"""Small-strain 3D elasticity with linear tetrahedra and lumped mass.

All exposed mechanical vectors use node-major [ux, uy, uz] ordering. Free
surfaces follow from the weak form; there are no clamped nodes by default.
"""

from dataclasses import dataclass
from typing import Callable
import math

import numpy as np
from scipy import sparse

from traxis.config import Material, positive
from traxis.geometry import TetMesh
from traxis.patches import disk_triangle_shape_integrals
from traxis.resources import ResourceGuard


def constitutive(material: Material) -> np.ndarray:
    lam, mu = material.lame
    d = np.zeros((6, 6))
    d[:3, :3] = lam
    d[np.arange(3), np.arange(3)] += 2 * mu
    d[np.arange(3, 6), np.arange(3, 6)] = mu
    return d


def element_matrices(points: np.ndarray, material: Material) -> tuple[np.ndarray, np.ndarray]:
    """Return stiffness and volume for batches of (4,3) vertex coordinates."""
    points = np.asarray(points, dtype=float)
    if points.ndim != 3 or points.shape[1:] != (4, 3) or not np.isfinite(points).all():
        raise ValueError("expected finite tetrahedron coordinates with shape (n,4,3)")
    affine = np.concatenate([np.ones((*points.shape[:2], 1)), points], axis=2)
    edges = points[:, 1:] - points[:, :1]
    determinant = np.linalg.det(edges)
    scale = np.linalg.norm(edges, axis=2).max(axis=1)
    if np.any(np.abs(determinant) <= 1e-12 * scale**3):
        raise ValueError("degenerate or severely ill-conditioned tetrahedron")
    volume = np.abs(determinant) / 6
    gradient = np.linalg.inv(affine)[:, 1:, :].transpose(0, 2, 1)
    b = np.zeros((len(points), 6, 12))
    for node in range(4):
        x, y, z = gradient[:, node, :].T
        j = 3 * node
        b[:, 0, j] = x
        b[:, 1, j + 1] = y
        b[:, 2, j + 2] = z
        b[:, 3, j], b[:, 3, j + 1] = y, x
        b[:, 4, j + 1], b[:, 4, j + 2] = z, y
        b[:, 5, j], b[:, 5, j + 2] = z, x
    stiffness = (b.transpose(0, 2, 1) @ constitutive(material) @ b) * volume[:, None, None]
    return stiffness, volume


@dataclass
class ElasticSystem:
    stiffness: sparse.csr_matrix
    mass_kg: np.ndarray
    mesh: TetMesh
    stable_dt_s: float

    @property
    def ndof(self) -> int:
        return len(self.mass_kg)


def assemble(mesh: TetMesh, material: Material = Material(),
             guard: ResourceGuard | None = None, batch_size: int = 256) -> ElasticSystem:
    if type(batch_size) is not int or not 1 <= batch_size <= 4096:
        raise ValueError("assembly batch size must be between 1 and 4096")
    points, tets = mesh.points_m, mesh.tetrahedra
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("invalid mesh points")
    if tets.ndim != 2 or tets.shape[1] != 4 or len(tets) == 0 or tets.dtype.kind not in "iu":
        raise ValueError("invalid tetrahedron connectivity")
    if tets.min() < 0 or tets.max() >= len(points):
        raise ValueError("tetrahedron node index out of range")
    ndof = len(points) * 3
    stiffness = sparse.csr_matrix((ndof, ndof))
    nodal_mass = np.zeros(len(points))
    for start in range(0, len(tets), batch_size):
        if guard:
            guard.check()
        nodes = tets[start:start + batch_size]
        ke, volume = element_matrices(points[nodes], material)
        np.add.at(nodal_mass, nodes.ravel(), np.repeat(volume * material.density_kg_m3 / 4, 4))
        dofs = (nodes[:, :, None] * 3 + np.arange(3)).reshape(-1, 12)
        rows = np.broadcast_to(dofs[:, :, None], ke.shape).ravel()
        cols = np.broadcast_to(dofs[:, None, :], ke.shape).ravel()
        stiffness += sparse.coo_matrix((ke.ravel(), (rows, cols)), shape=(ndof, ndof)).tocsr()
    if np.any(nodal_mass <= 0):
        raise ValueError("mesh has unused or zero-mass nodes")
    mass = np.repeat(nodal_mass, 3)
    invsqrt = 1 / np.sqrt(mass)
    bound = np.max(invsqrt * (abs(stiffness) @ invsqrt))
    if not np.isfinite(bound) or bound <= 0:
        raise ValueError("invalid stiffness eigenvalue bound")
    if guard:
        guard.check(force=True)
    return ElasticSystem(stiffness, mass, mesh, 2 / math.sqrt(bound))


def surface_patch(system: ElasticSystem, center_xy_m: tuple[float, float], radius_m: float,
                  direction=(0.0, 0.0, 1.0)) -> np.ndarray:
    """Unit-total-force circular load / matching area-averaged displacement."""
    return surface_patch_with_report(system, center_xy_m, radius_m, direction)[0]


def surface_patch_with_report(system: ElasticSystem, center_xy_m: tuple[float, float], radius_m: float,
                              direction=(0.0, 0.0, 1.0)) -> tuple[np.ndarray, dict]:
    """Integrate P1 functions on the disk/top-face intersections analytically.

    Transmit/receive weights are work-conjugate, preserving mechanical
    reciprocity. This is displacement, not bonded-PZT voltage or strain.
    """
    positive("patch radius", radius_m)
    center = np.asarray(center_xy_m, dtype=float)
    direction = np.asarray(direction, dtype=float)
    if center.shape != (2,) or not np.isfinite(center).all():
        raise ValueError("patch center must contain two finite coordinates")
    if direction.shape != (3,) or not np.isfinite(direction).all() or np.linalg.norm(direction) == 0:
        raise ValueError("patch direction must be a finite nonzero 3-vector")
    direction = direction / np.linalg.norm(direction)
    mesh = system.mesh
    top_z = mesh.plate.thickness_m
    top_nodes = np.isclose(mesh.points_m[:, 2], top_z, rtol=0, atol=top_z * 1e-10)
    all_faces = mesh.tetrahedra[:, np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])].reshape(-1, 3)
    faces = np.sort(all_faces[top_nodes[all_faces].all(axis=1)], axis=1)
    faces, multiplicities = np.unique(faces, axis=0, return_counts=True)
    faces = faces[multiplicities == 1]
    xy = mesh.points_m[faces, :2]
    lower, upper = xy.min(axis=1) - center, xy.max(axis=1) - center
    nearest = np.maximum(np.maximum(lower, -upper), 0)
    faces = faces[(nearest**2).sum(axis=1) <= radius_m**2]
    weights = np.zeros(len(mesh.points_m))
    covered = 0
    for ids in faces:
        integrals = disk_triangle_shape_integrals(mesh.points_m[ids, :2], center, radius_m)
        weights[ids] += integrals
        covered += int(integrals.sum() > 0)
    area = float(weights.sum())
    if area <= 0:
        raise ValueError("patch has no positive-area intersection with the top surface")
    centroid = weights @ mesh.points_m[:, :2] / area
    weights /= area
    report = {"integration": "analytic_disk_P1_shape_integrals", "requested_center_xy_m": center.tolist(),
              "radius_m": radius_m, "direction": direction.tolist(), "area_m2": area,
              "centroid_xy_m": centroid.tolist(), "centroid_error_m": float(np.linalg.norm(centroid - center)),
              "full_disk_area_relative_error": abs(area / (math.pi * radius_m**2) - 1),
              "support_nodes": int(np.count_nonzero(weights)), "covered_surface_triangles": covered,
              "edge_clipped": bool(np.any(center < radius_m) or np.any(center > [mesh.plate.width_m - radius_m, mesh.plate.height_m - radius_m]))}
    return (weights[:, None] * direction).ravel(), report


def tone_burst(t_s: float, frequency_hz: float, cycles: int = 5) -> float:
    positive("frequency", frequency_hz)
    if type(cycles) is not int or cycles < 1:
        raise ValueError("cycles must be a positive integer")
    if not math.isfinite(t_s):
        raise ValueError("time must be finite")
    duration = cycles / frequency_hz
    if t_s < 0 or t_s >= duration:
        return 0.0
    return math.sin(2 * math.pi * frequency_hz * t_s) * 0.5 * (1 - math.cos(2 * math.pi * t_s / duration))


@dataclass
class SimulationResult:
    time_s: np.ndarray
    displacement_m: np.ndarray
    physical_energy_j: np.ndarray
    modified_energy_j: np.ndarray
    final_displacement_m: np.ndarray
    final_velocity_m_s: np.ndarray


def simulate(system: ElasticSystem, steps: int, dt_s: float, probes: np.ndarray,
             force: Callable[[float], np.ndarray] | None = None,
             initial_displacement: np.ndarray | None = None,
             initial_velocity: np.ndarray | None = None,
             guard: ResourceGuard | None = None) -> SimulationResult:
    """Velocity Verlet without damping, storing probes and energies only."""
    positive("time step", dt_s)
    if dt_s > 0.95 * system.stable_dt_s:
        raise ValueError("time step exceeds conservative explicit stability limit")
    if type(steps) is not int or not 1 <= steps <= 50000:
        raise ValueError("steps must be between 1 and 50000")
    probes = np.asarray(probes, dtype=float)
    if probes.ndim != 2 or probes.shape[1] != system.ndof or not np.isfinite(probes).all():
        raise ValueError("probes must have finite shape (n_probes, ndof)")
    if (steps + 1) * (probes.shape[0] + 3) * 8 > 128 * 1024**2:
        raise MemoryError("probe storage would exceed 128 MiB")

    def vector(value):
        result = np.zeros(system.ndof) if value is None else np.array(value, dtype=float, copy=True)
        if result.shape != (system.ndof,) or not np.isfinite(result).all():
            raise ValueError("state/force must have finite shape (ndof,)")
        return result

    u, v = vector(initial_displacement), vector(initial_velocity)
    mass, k = system.mass_kg, system.stiffness
    times = np.arange(steps + 1) * dt_s
    traces = np.empty((steps + 1, len(probes)))
    physical, modified = np.empty(steps + 1), np.empty(steps + 1)
    ku = k @ u
    a = (vector(force(0)) - ku) / mass if force else -ku / mass
    for step in range(steps + 1):
        if guard and step % 16 == 0:
            guard.check()
        if not np.isfinite(u).all() or not np.isfinite(v).all():
            raise FloatingPointError("nonfinite mechanical state")
        traces[step] = probes @ u
        physical[step] = 0.5 * (np.dot(mass * v, v) + np.dot(u, ku))
        # Exact invariant of the unforced, undamped Verlet recurrence (roundoff aside).
        modified[step] = physical[step] - dt_s**2 / 8 * np.dot(ku / mass, ku)
        if step == steps:
            break
        u += dt_s * v + 0.5 * dt_s**2 * a
        ku = k @ u
        next_a = (vector(force(times[step + 1])) - ku) / mass if force else -ku / mass
        v += 0.5 * dt_s * (a + next_a)
        a = next_a
    if guard:
        guard.check(force=True)
    return SimulationResult(times, traces, physical, modified, u, v)
