from dataclasses import dataclass
import math

import numpy as np

from traxis.config import CircularRecess, Plate


def depth_map_mm(plate: Plate, recesses: tuple[CircularRecess, ...], shape=(512, 512),
                 supersample: int = 1) -> np.ndarray:
    """Pixel area averages; overlapping recesses use maximum removed depth.

    Axis order is (y, x), with pixel centers at half-cell offsets. Supersampling
    is quadrature of the analytic geometry, not extra physical resolution.
    """
    if len(shape) != 2 or any(type(n) is not int or n <= 0 for n in shape):
        raise ValueError("shape must contain two positive integers")
    if type(supersample) is not int or not 1 <= supersample <= 16:
        raise ValueError("supersample must be between 1 and 16")
    for recess in recesses:
        recess.validate(plate)
    ny, nx = shape
    result = np.zeros(shape, dtype=np.float64)
    for sy in range(supersample):
        y = (np.arange(ny) + (sy + 0.5) / supersample) * plate.height_m / ny
        for sx in range(supersample):
            x = (np.arange(nx) + (sx + 0.5) / supersample) * plate.width_m / nx
            sample = np.zeros(shape)
            for r in recesses:
                inside = (x[None, :] - r.x_m)**2 + (y[:, None] - r.y_m)**2 <= r.radius_m**2
                sample[inside] = np.maximum(sample[inside], r.depth_m * 1000)
            result += sample
    return (result / supersample**2).astype(np.float32)


@dataclass(frozen=True)
class MeshEstimate:
    nodes_upper: int
    tetrahedra_upper: int
    assembly_bytes_upper: int


def estimate_mesh(cells: tuple[int, int, int]) -> MeshEstimate:
    if len(cells) != 3 or any(type(n) is not int or n < 1 for n in cells):
        raise ValueError("cells must be three positive integers")
    nodes = math.prod(n + 1 for n in cells)
    tets = 6 * math.prod(cells)
    # Includes duplicate COO entries, CSR storage/workspace, mesh, vectors.
    # Conservative screening estimate, not a measured peak-memory guarantee.
    return MeshEstimate(nodes, tets, tets * 144 * 40 + nodes * 256)


@dataclass
class TetMesh:
    points_m: np.ndarray
    tetrahedra: np.ndarray
    plate: Plate
    removed_volume_m3: float
    requested_removed_volume_m3: float


def structured_mesh(plate: Plate, cells: tuple[int, int, int],
                    recesses: tuple[CircularRecess, ...] = (),
                    max_assembly_bytes: int = 256 * 1024**2) -> TetMesh:
    """Conforming six-tet cells with centroid-voxelized backside removal.

    Every cell shares the same body-diagonal orientation. Unused nodes are
    compacted so removed material cannot leave zero-mass degrees of freedom.
    """
    estimate = estimate_mesh(cells)
    if estimate.assembly_bytes_upper > max_assembly_bytes:
        raise MemoryError(f"estimated assembly {estimate.assembly_bytes_upper / 1024**2:.1f} MiB "
                          f"exceeds {max_assembly_bytes / 1024**2:.1f} MiB limit")
    for r in recesses:
        r.validate(plate)
    nx, ny, nz = cells
    axes = [np.linspace(0, size, n + 1) for size, n in
            zip((plate.width_m, plate.height_m, plate.thickness_m), cells)]
    grid = np.meshgrid(*axes, indexing="ij")
    points = np.column_stack([a.ravel() for a in grid])
    node_ids = np.arange(len(points)).reshape(nx + 1, ny + 1, nz + 1)
    offsets = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0],
                        [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]])
    pattern = np.array([[0, 1, 3, 7], [0, 3, 2, 7], [0, 2, 6, 7],
                        [0, 6, 4, 7], [0, 4, 5, 7], [0, 5, 1, 7]])
    tets = []
    removed = 0
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                center = np.array([(axes[0][i] + axes[0][i + 1]) / 2,
                                   (axes[1][j] + axes[1][j + 1]) / 2,
                                   (axes[2][k] + axes[2][k + 1]) / 2])
                if any((center[0] - r.x_m)**2 + (center[1] - r.y_m)**2 <= r.radius_m**2
                       and center[2] < r.depth_m for r in recesses):
                    removed += 1
                    continue
                idx = offsets + [i, j, k]
                vertices = node_ids[idx[:, 0], idx[:, 1], idx[:, 2]]
                tets.extend(vertices[pattern])
    connectivity = np.asarray(tets, dtype=np.int64)
    used, inverse = np.unique(connectivity, return_inverse=True)
    volume = plate.width_m * plate.height_m * plate.thickness_m
    separated = all(math.hypot(a.x_m - b.x_m, a.y_m - b.y_m) >= a.radius_m + b.radius_m
                    for i, a in enumerate(recesses) for b in recesses[i + 1:])
    # Analytic sum for disjoint circles; area quadrature for overlapping unions.
    if separated:
        requested = sum(r.volume_m3 for r in recesses)
    else:
        requested = float(depth_map_mm(plate, recesses, (512, 512), 2).mean()) / 1000
        requested *= plate.width_m * plate.height_m
    return TetMesh(points[used], inverse.reshape(-1, 4), plate,
                   volume * removed / (nx * ny * nz), requested)
