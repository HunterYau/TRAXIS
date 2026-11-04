"""Headless OpenCASCADE/Gmsh meshes for backside circular recesses.

Gmsh API reference: https://gmsh.info/doc/texinfo/ . The CAD surface is exact;
linear surface facets/tetrahedra approximate it and their volume error is
reported. A resolved CAD shape does not establish resolved guided waves.
"""

import math

import numpy as np

from traxis.config import CircularRecess, Plate, positive
from traxis.geometry import TetMesh
from traxis.resources import ResourceGuard


def cad_mesh(plate: Plate, recesses: tuple[CircularRecess, ...], size_m: float,
              *, max_assembly_mib: float = 256, max_tetrahedra: int = 40000,
              seed: int = 2026, guard: ResourceGuard | None = None) -> tuple[TetMesh, dict]:
    positive("CAD mesh size", size_m)
    positive("assembly limit", max_assembly_mib)
    if type(max_tetrahedra) is not int or max_tetrahedra < 1:
        raise ValueError("tetrahedron limit must be positive")
    if type(seed) is not int or seed < 0:
        raise ValueError("mesh seed must be nonnegative")
    for r in recesses:
        r.validate(plate)
    volume = plate.width_m * plate.height_m * plate.thickness_m
    area = 2 * (plate.width_m * plate.height_m + (plate.width_m + plate.height_m) * plate.thickness_m)
    # Screening estimate only: actual mesh size depends on geometry/quality.
    estimated_tets = math.ceil(12 * volume / size_m**3 + 6 * area / size_m**2)
    estimated_bytes = estimated_tets * 144 * 40
    if estimated_tets > max_tetrahedra or estimated_bytes > max_assembly_mib * 1024**2:
        raise MemoryError("CAD mesh screening estimate exceeds the local element/assembly limit")
    if guard:
        guard.check(force=True)
    import gmsh
    if gmsh.isInitialized():
        raise RuntimeError("Gmsh is already initialized; use an isolated mesh session")
    gmsh.initialize([], readConfigFiles=False, run=False)
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.option.setNumber("General.NumThreads", 1)
        for dim in (1, 2, 3):
            gmsh.option.setNumber(f"Mesh.MaxNumThreads{dim}D", 1)
        gmsh.option.setNumber("Mesh.ElementOrder", 1)
        gmsh.option.setNumber("Mesh.Algorithm3D", 1)
        gmsh.option.setNumber("Mesh.MeshSizeMax", size_m)
        gmsh.option.setNumber("Mesh.MeshSizeMin", size_m / 2)
        gmsh.option.setNumber("Mesh.RandomSeed", seed % (2**31 - 1))
        gmsh.model.add("backside-recess")
        box = gmsh.model.occ.addBox(0, 0, 0, plate.width_m, plate.height_m, plate.thickness_m)
        if recesses:
            tools = []
            for r in recesses:
                # Extend below z=0 to avoid coincident bottom cutter caps.
                tag = gmsh.model.occ.addCylinder(r.x_m, r.y_m, -plate.thickness_m,
                                                 0, 0, plate.thickness_m + r.depth_m, r.radius_m)
                tools.append((3, tag))
            volumes, _ = gmsh.model.occ.cut([(3, box)], tools)
        else:
            volumes = [(3, box)]
        gmsh.model.occ.synchronize()
        if not volumes or any(dim != 3 for dim, _ in volumes):
            raise ValueError("CAD subtraction did not produce a solid plate")
        cad_volume = sum(gmsh.model.occ.getMass(dim, tag) for dim, tag in volumes)
        gmsh.model.mesh.setSize(gmsh.model.getEntities(0), size_m)
        if guard:
            guard.check()
        gmsh.model.mesh.generate(3)
        types, _element_ids, connectivity = gmsh.model.mesh.getElements(3)
        if len(types) != 1 or int(types[0]) != 4:
            raise ValueError("local elasticity backend requires linear four-node tetrahedra")
        tags, coordinates, _ = gmsh.model.mesh.getNodes()
        count = len(connectivity[0]) // 4
        if count > max_tetrahedra or count * 144 * 40 > max_assembly_mib * 1024**2:
            raise MemoryError("actual CAD mesh exceeds the local element/assembly limit")
        order = np.argsort(tags)
        tags = tags[order]
        xyz = np.asarray(coordinates).reshape(-1, 3)[order]
        element_tags = np.asarray(connectivity[0]).reshape(-1, 4)
        tets = np.searchsorted(tags, element_tags)
        if np.any(tets >= len(tags)) or not np.array_equal(tags[tets], element_tags):
            raise ValueError("Gmsh connectivity references unknown nodes")
        used, inverse = np.unique(tets, return_inverse=True)
        xyz, tets = xyz[used], inverse.reshape(-1, 4)
        tet_volumes = abs(np.linalg.det(xyz[tets[:, 1:]] - xyz[tets[:, :1]])) / 6
        if np.any(tet_volumes <= 0):
            raise ValueError("CAD mesh contains degenerate tetrahedra")
        mesh_volume = float(tet_volumes.sum())
        requested = max(0., volume - cad_volume)
        removed = max(0., volume - mesh_volume)
        mesh = TetMesh(xyz, tets, plate, removed, requested)
        report = {"backend": "gmsh_opencascade", "gmsh_version": gmsh.__version__, "size_m": size_m,
                  "mesh_seed": seed, "threads": 1, "nodes": len(xyz), "tetrahedra": len(tets),
                  "cad_retained_volume_m3": cad_volume, "tet_retained_volume_m3": mesh_volume,
                  "retained_volume_relative_error": abs(mesh_volume / cad_volume - 1),
                  "requested_removed_volume_m3": requested, "meshed_removed_volume_m3": removed,
                  "removed_volume_relative_error": abs(removed / requested - 1) if requested else None,
                  "screened_tetrahedra": estimated_tets, "screened_assembly_mib": estimated_bytes / 1024**2,
                  "ready_for_training_data": False,
                  "note": "CAD/mesh geometry check only; no PZT or guided-wave convergence validation."}
        if guard:
            guard.check(force=True)
        return mesh, report
    finally:
        gmsh.finalize()
