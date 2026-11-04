"""Bounded numerical checks, separate from physical validation."""

from dataclasses import asdict
import json
import hashlib
from pathlib import Path
import platform
import time

import numpy as np
import scipy
from scipy.linalg import eigh

from traxis import __version__
from traxis.config import CircularRecess, Material, Plate
from traxis.fem import assemble, constitutive, simulate, surface_patch, tone_burst
from traxis.geometry import structured_mesh
from traxis.resources import ResourceGuard
from traxis.dispersion import wavenumber_rad_m
from traxis.waveguide import lamb_frequencies_hz


def run_verification(output: Path, guard: ResourceGuard) -> dict:
    """Write provenance, numeric gates, waveforms and complete telemetry."""
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    report = {"kind": "numerical_verification_not_physical_validation", "passed": False,
              "ready_for_training_data": False,
              "traxis_version": __version__, "python": platform.python_version(),
              "numpy": np.__version__, "scipy": scipy.__version__, "gates": {},
              "limitations": ["Small smoke plate, not full research geometry",
                              "No bonded PZT/electrical/adhesive model",
                              "No experimental traces or licensed solver comparison",
                              "No learned models or training in this verification"]}
    report["source_sha256"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in sorted(Path(__file__).parent.glob("*.py"))}
    try:
        guard.check(force=True)
        material = Material()
        plate = Plate(0.04, 0.04, 0.003)
        cells = (8, 8, 4)
        recess = CircularRecess(0.02, 0.02, 0.008, 0.0015)
        systems = [assemble(structured_mesh(plate, cells, defects), material, guard)
                   for defects in ((), (recess,))]
        intact, damaged = systems
        report["settings"] = {"plate": asdict(plate), "material": asdict(material),
                              "cells": cells, "recess": asdict(recess),
                              "observable": "area-averaged top-surface normal displacement [m]",
                              "source": "Hann tone burst, 150 kHz, 2 cycles, 1 N peak scale",
                              "boundaries": "all natural traction-free", "damping": "none"}
        report["meshes"] = [{"nodes": len(s.mesh.points_m), "tetrahedra": len(s.mesh.tetrahedra),
                             "ndof": s.ndof, "stable_dt_s": s.stable_dt_s,
                             "removed_volume_m3": s.mesh.removed_volume_m3,
                             "requested_removed_volume_m3": s.mesh.requested_removed_volume_m3,
                             "relative_removed_volume_error": abs(s.mesh.removed_volume_m3 / s.mesh.requested_removed_volume_m3 - 1)
                             if s.mesh.requested_removed_volume_m3 else 0.0}
                            for s in systems]
        gates = report["gates"]

        def gate(name, value, threshold, comparison="less"):
            passed = value < threshold if comparison == "less" else value > threshold
            gates[name] = {"value": float(value), "threshold": threshold,
                           "comparison": comparison, "passed": bool(passed)}

        xyz = intact.mesh.points_m
        rigid = (np.cross([0.1, 0.2, -0.1], xyz) + [0.3, 0.2, 0.1]).ravel()
        k, mass = intact.stiffness, intact.mass_kg
        gate("rigid_motion_relative_residual", np.linalg.norm(k @ rigid) /
             (np.linalg.norm(k.data) * np.linalg.norm(rigid)), 1e-12)
        g = np.array([[0.001, 0.002, 0.003], [0.004, -0.001, 0.002], [-0.002, 0.003, 0.001]])
        strain = np.array([g[0, 0], g[1, 1], g[2, 2], g[0, 1] + g[1, 0],
                           g[1, 2] + g[2, 1], g[0, 2] + g[2, 0]])
        u = (xyz @ g.T).ravel()
        expected = 0.5 * plate.width_m * plate.height_m * plate.thickness_m * strain @ constitutive(material) @ strain
        gate("affine_energy_relative_error", abs(0.5 * u @ k @ u / expected - 1), 1e-11)
        gate("intact_mass_relative_error", abs(mass[0::3].sum() /
             (plate.width_m * plate.height_m * plate.thickness_m * material.density_kg_m3) - 1), 1e-12)

        dt = min(s.stable_dt_s for s in systems) * 0.8
        # Covers drive and first finite-plate responses, at most 50,000 steps.
        steps = int(np.ceil(40e-6 / dt))
        traces = []
        for system in systems:
            a = surface_patch(system, (0.007, 0.02), 0.005)
            b = surface_patch(system, (0.033, 0.02), 0.005)
            run = simulate(system, steps, dt, b[None, :],
                           force=lambda t: a * tone_burst(t, 150000, 2), guard=guard)
            traces.append(run.displacement_m[:, 0])
            post_drive = run.time_s > 2 / 150000 + dt
            invariant = run.modified_energy_j[post_drive]
            gate(f"{'intact' if system is intact else 'damaged'}_post_drive_energy_drift",
                 np.ptp(invariant) / abs(invariant.mean()), 1e-8)
        a = surface_patch(intact, (0.007, 0.02), 0.005)
        b = surface_patch(intact, (0.033, 0.02), 0.005)
        reverse = simulate(intact, steps, dt, a[None, :],
                           force=lambda t: b * tone_burst(t, 150000, 2), guard=guard)
        gate("mechanical_reciprocity_relative_error", np.linalg.norm(traces[0] - reverse.displacement_m[:, 0]) /
             np.linalg.norm(traces[0]), 1e-9)
        # Sensitivity is a regression check, not evidence of correct defect scattering.
        gate("defect_response_relative_change", np.linalg.norm(traces[0] - traces[1]) /
             np.linalg.norm(traces[0]), 0.01, "greater")
        report["time_integration"] = {"dt_s": dt, "steps": steps, "duration_s": steps * dt}
        np.savez_compressed(output / "mechanical-traces.npz", time_s=run.time_s,
                            intact_displacement_m=traces[0], damaged_displacement_m=traces[1],
                            reverse_displacement_m=reverse.displacement_m[:, 0])

        # A continuum cosine mode provides an independent spatial check.
        spatial = []
        mat = Material(poisson=0)
        cube = Plate(0.01, 0.01, 0.01)
        exact_lambda = mat.young_pa / mat.density_kg_m3 * (np.pi / cube.width_m)**2
        for n in (2, 4, 8):
            s = assemble(structured_mesh(cube, (n, n, n)), mat, guard)
            u = np.zeros((len(s.mesh.points_m), 3))
            u[:, 0] = np.cos(np.pi * s.mesh.points_m[:, 0] / cube.width_m)
            u = u.ravel()
            quotient = (u @ s.stiffness @ u) / np.dot(s.mass_kg * u, u)
            spatial.append({"cells_per_axis": n, "rayleigh_quotient_s_2": float(quotient),
                            "relative_error": float(abs(quotient / exact_lambda - 1))})
        report["spatial_refinement"] = {"reference_eigenvalue_s_2": exact_lambda, "levels": spatial,
                                        "scope": "free nu=0 cube cosine-mode energy; not Lamb-wave convergence"}
        for i in (0, 1):
            gate(f"spatial_error_reduction_{i}", spatial[i]["relative_error"] / spatial[i + 1]["relative_error"], 3.5, "greater")

        # Small generalized eigensystem isolates time-integration error.
        tiny = assemble(structured_mesh(Plate(0.02, 0.02, 0.003), (2, 2, 2)), material, guard)
        values, modes = eigh(tiny.stiffness.toarray(), np.diag(tiny.mass_kg))
        mode = modes[:, 10] * 1e-8
        omega = np.sqrt(values[10])
        dt = tiny.stable_dt_s * 0.8
        exact = mode * np.cos(omega * 80 * dt)
        errors = []
        for refinement in (1, 2, 4):
            r = simulate(tiny, 80 * refinement, dt / refinement, mode[None, :],
                         initial_displacement=mode, guard=guard)
            errors.append(float(np.linalg.norm(r.final_displacement_m - exact)))
        report["temporal_refinement"] = {"error_norms_m": errors, "refinement_factors": [1, 2, 4]}
        for i in (0, 1):
            gate(f"temporal_error_reduction_{i}", errors[i] / errors[i + 1], 3.8, "greater")

        # Periodic unit cells compare the actual tetrahedral wave operator to
        # an independently solved traction equation, at two plate thicknesses.
        report["guided_mode_dispersion"] = []
        for thickness in (0.001, 0.003):
            for mode, mode_index in (("A0", 0), ("S0", 1)):
                f = 100000.
                wavenumber = wavenumber_rad_m(f, thickness, mode, material)
                levels = []
                for n in (16, 32, 64, 128):
                    fem_f = lamb_frequencies_hz(wavenumber, thickness, n, n // 2, material, guard)[mode_index]
                    levels.append({"elements_per_wavelength": n, "thickness_cells": n // 2,
                                   "fem_frequency_hz": fem_f, "relative_error": abs(fem_f / f - 1)})
                entry = {"mode": mode, "thickness_m": thickness, "reference_frequency_hz": f,
                         "reference_wavenumber_rad_m": wavenumber, "levels": levels}
                report["guided_mode_dispersion"].append(entry)
                prefix = f"{mode}_{thickness * 1000:g}mm"
                gate(f"{prefix}_finest_dispersion_error", levels[-1]["relative_error"], 0.005)
                reduction = min(a["relative_error"] / b["relative_error"] for a, b in zip(levels, levels[1:]))
                gate(f"{prefix}_dispersion_error_reduction", reduction, 3, "greater")
        report["passed"] = all(g["passed"] for g in gates.values())
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        report["wall_seconds"] = time.monotonic() - started
        report["resources"] = guard.report()
        (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    return report
