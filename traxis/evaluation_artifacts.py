"""Bounded, authenticated reads of completed replay reports and saved maps."""

import json
from pathlib import Path
import zipfile

import numpy as np

from traxis.config import Plate
from traxis.metrics import depth_metrics
from traxis.storage import canonical_hash, sha256_file


def load_evaluation(directory: Path) -> dict:
    path = Path(directory) / "evaluation.json"
    if path.stat().st_size > 64 * 1024**2:
        raise MemoryError("evaluation report exceeds 64 MiB")
    report = json.loads(path.read_text(encoding="utf-8-sig"))
    if report.get("schema_version") != 2 or not report.get("passed"):
        raise ValueError("requires a completed schema-2 evaluation; rerun older evaluations")
    body = {k: v for k, v in report.items() if k != "fingerprint"}
    if canonical_hash(body) != report.get("fingerprint"):
        raise ValueError("evaluation report fingerprint mismatch")
    plan = report["plan"]
    if canonical_hash({k: v for k, v in plan.items() if k != "fingerprint"}) != plan["fingerprint"]:
        raise ValueError("evaluation plan fingerprint mismatch")
    if report["plan_fingerprint"] != plan["fingerprint"] or report["dataset_fingerprint"] != plan["dataset_fingerprint"]:
        raise ValueError("report and frozen plan disagree")
    if not report["episodes"]:
        raise ValueError("evaluation has no episodes")
    return report


def load_case(directory: Path, episode: dict, index: int, records: int) -> tuple[Plate, dict]:
    """Authenticate one map archive, then check reported errors on its grid."""
    case = Path(directory) / f"case-{index:05d}"
    metrics_path = case / "metrics.json"
    if metrics_path.stat().st_size > 4 * 1024**2:
        raise MemoryError("case metrics exceed 4 MiB")
    saved = json.loads(metrics_path.read_text(encoding="utf-8-sig"))
    if canonical_hash(saved) != canonical_hash(episode):
        raise ValueError("case metrics disagree with evaluation report")
    maps_path = case / "maps.npz"
    if sha256_file(maps_path) != episode["maps_sha256"]:
        raise ValueError("evaluation maps checksum mismatch")
    with zipfile.ZipFile(maps_path) as archive:
        if sum(item.file_size for item in archive.infolist()) > 256 * 1024**2:
            raise MemoryError("evaluation maps exceed 256 MiB uncompressed")
    plate = Plate(**episode["plate"])
    row = next((r for r in episode["rows"] if r["records"] == records), None)
    if row is None:
        raise ValueError("requested case budget was not evaluated")
    with np.load(maps_path, allow_pickle=False) as archive:
        maps = {"truth": archive["truth_mm"].copy(), "prediction": archive[f"depth_{records}_mm"].copy()}
        name = f"std_{records}_mm"
        if name in archive.files:
            maps["std"] = archive[name].copy()
    if any(a.ndim != 2 or max(a.shape) > 1024 or not np.isfinite(a).all() for a in maps.values()):
        raise ValueError("maps must be finite 2D arrays at most 1024 cells per side")
    if "reference_shape" in episode and tuple(episode["reference_shape"]) != maps["truth"].shape:
        raise ValueError("saved truth grid disagrees with the recorded reference shape")
    calculated = depth_metrics(maps["prediction"], maps["truth"], plate, episode["threshold_mm"])
    for metric in ("mae_mm", "rmse_mm", "dice", "defect_region_mae_mm"):
        a, b = calculated[metric], row["metrics"][metric]
        if (a is None) != (b is None) or (a is not None and not np.isclose(a, b, rtol=1e-6, atol=1e-8)):
            raise ValueError("saved maps do not reproduce reported depth metrics")
    if "std" in maps and (maps["std"].shape != maps["prediction"].shape or np.any(maps["std"] < 0)):
        raise ValueError("ensemble spread must match the prediction grid and be nonnegative")
    return plate, maps
