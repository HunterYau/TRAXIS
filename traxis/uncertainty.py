"""Split-conformal intervals for one frozen, finite inspection protocol.

One independent parent contributes one worst-case score over its map cells,
saved budgets and policies. Coverage requires exchangeable new parents and
an unchanged acquisition/reconstruction protocol; code cannot verify those
scientific assumptions. See docs/uncertainty-calibration.md.
"""

from decimal import Decimal, ROUND_CEILING
import json
import math
from pathlib import Path
import time

import numpy as np

from traxis.evaluation_artifacts import load_case, load_evaluation
from traxis.metrics import resample_cell_map
from traxis.resources import ResourceGuard
from traxis.storage import atomic_write, canonical_hash, sha256_file, write_json


def _source_hash():
    root = Path(__file__).parent
    return canonical_hash({name: sha256_file(root / name) for name in
                           ("uncertainty.py", "evaluation_artifacts.py", "metrics.py", "storage.py")})


def corrected_quantile(scores, alpha: float) -> tuple[float | None, int]:
    """Return the ceil((n+1)(1-alpha))-th score; None denotes +infinity.

    Select the order statistic directly: interpolating empirical quantiles
    does not implement this finite-sample rank. Decimal avoids a spurious
    extra rank when a decimal alpha makes the product an integer.
    """
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 1 or not len(scores) or not np.isfinite(scores).all() or np.any(scores < 0):
        raise ValueError("calibration requires nonempty, finite, nonnegative parent scores")
    if not math.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("alpha must lie strictly between zero and one")
    rank = int((Decimal(len(scores) + 1) * (Decimal(1) - Decimal(str(alpha)))).to_integral_value(rounding=ROUND_CEILING))
    return (float(np.partition(scores, rank - 1)[rank - 1]) if rank <= len(scores) else None), rank


def _read_fingerprinted(path: Path, kind: str) -> dict:
    path = Path(path)
    if path.stat().st_size > 8 * 1024**2:
        raise MemoryError("calibration artifact/plan exceeds 8 MiB")
    result = json.loads(path.read_text(encoding="utf-8-sig"))
    if result.get("schema_version") != 1 or result.get("kind") != kind:
        raise ValueError("unsupported calibration artifact/plan")
    if canonical_hash({k: v for k, v in result.items() if k != "fingerprint"}) != result.get("fingerprint"):
        raise ValueError("calibration artifact/plan fingerprint mismatch")
    if result.get("calibration_source_hash") != _source_hash():
        raise ValueError("calibration source changed; use the recorded revision or freeze/refit explicitly")
    return result


def _policy_identity(spec: dict) -> dict:
    if spec["kind"] == "rl":
        return {"kind": "rl", "sha256": spec["sha256"]}
    return spec


def _inspection_protocol(plan: dict) -> dict:
    return {"models": [{"sha256": m["sha256"], "kind": m["kind"]} for m in plan["models"]],
            "ensemble": plan["ensemble"], "initial_actions": plan["initial_actions"],
            "budgets": plan["budgets"], "threshold_mm": plan["threshold_mm"],
            "policies": [_policy_identity(p) for p in plan["policy_specs"]],
            "inference_source_hash": plan["inference_source_hash"]}


def freeze_calibration_plan(evaluation_plan: Path, output: Path, *, alpha: float = .1,
                             std_floor_mm: float = .05, mode: str = "debug") -> dict:
    """Freeze scalar settings against a replay plan before calibration replay."""
    path = Path(evaluation_plan)
    if path.stat().st_size > 8 * 1024**2:
        raise MemoryError("replay plan exceeds 8 MiB")
    plan = json.loads(path.read_text(encoding="utf-8-sig"))
    if canonical_hash({k: v for k, v in plan.items() if k != "fingerprint"}) != plan["fingerprint"]:
        raise ValueError("replay plan fingerprint mismatch")
    if plan["split"] != "uncertainty_calibration" or not plan["ensemble"]:
        raise ValueError("calibration needs an ensemble replay plan for the dedicated uncertainty_calibration split")
    corrected_quantile([0.], alpha)
    if not math.isfinite(std_floor_mm) or std_floor_mm <= 0 or mode not in ("debug", "research"):
        raise ValueError("record a positive spread floor in mm and debug/research mode")
    result = {"schema_version": 1, "kind": "frozen_uncertainty_calibration_plan",
              "evaluation_plan_fingerprint": plan["fingerprint"], "protocol": _inspection_protocol(plan),
              "alpha": alpha, "std_floor_mm": std_floor_mm, "mode": mode,
              "calibration_source_hash": _source_hash(),
              "scope": "all reference cells, recorded budgets and listed policies for one new parent",
              "settings_requirement": "Select alpha, floor, models and protocol without calibration labels or final-test outcomes."}
    result["fingerprint"] = canonical_hash(result)
    write_json(output, result)
    return result


def _cohort_structure(report: dict) -> tuple[dict, list[str]]:
    """Restrict the first implementation to one scan per independent parent."""
    descriptors = list(report["predictors"].values())
    if len(descriptors) != 1 or descriptors[0]["kind"] != "ensemble":
        raise ValueError("calibration requires one frozen ensemble predictor")
    episodes = report["episodes"]
    parents, signatures = {}, set()
    expected = {canonical_hash(_policy_identity(p)) for p in report["plan"]["policy_specs"]}
    for episode in episodes:
        required = ("acquisition_signature", "observable", "reference_shape")
        if any(name not in episode for name in required):
            raise ValueError("replay lacks calibration context; freeze and replay with the current schema")
        if episode["split"] != report["plan"]["split"]:
            raise ValueError("episode split disagrees with its frozen plan")
        signature = {name: episode[name] for name in ("acquisition_signature", "observable", "reference_shape", "domain", "catalog_size")}
        signatures.add(canonical_hash(signature))
        parent = parents.setdefault(episode["parent_specimen_id"], {"specimens": set(), "policies": set()})
        parent["specimens"].add(episode["specimen_id"])
        identity = canonical_hash(_policy_identity(episode["policy"]))
        if identity in parent["policies"]:
            raise ValueError("repeated policy episodes/scans are not independent calibration parents")
        parent["policies"].add(identity)
        budgets = sorted(set(report["plan"]["budgets"]) | {episode["catalog_size"]})
        if [r["records"] for r in episode["rows"]] != budgets:
            raise ValueError("episodes must contain exactly the frozen budgets and dense reference")
    if len(signatures) != 1 or any(len(p["specimens"]) != 1 or p["policies"] != expected for p in parents.values()):
        raise ValueError("calibration needs one scan per parent, a common acquisition/grid/domain and every frozen policy")
    return signature, sorted(parents)


def _aligned_maps(plate, maps, floor):
    if "std" not in maps:
        raise ValueError("ensemble spread is missing from the saved replay maps")
    truth = maps["truth"].astype(float)
    mean = resample_cell_map(maps["prediction"], truth.shape)
    spread = resample_cell_map(maps["std"], truth.shape)
    nominal = plate.thickness_m * 1000
    if np.any(mean < 0) or np.any(mean > nominal) or floor > nominal:
        raise ValueError("mean/floor is inconsistent with the physical depth range")
    return truth, mean, np.maximum(spread, floor)


def fit_calibration(plan_path: Path, evaluation: Path, output: Path,
                     guard: ResourceGuard | None = None) -> dict:
    plan = _read_fingerprinted(plan_path, "frozen_uncertainty_calibration_plan")
    report = load_evaluation(evaluation)
    if report["plan_fingerprint"] != plan["evaluation_plan_fingerprint"] or report["plan"]["split"] != "uncertainty_calibration":
        raise ValueError("use the dedicated replay bound to the frozen calibration plan")
    if report.get("calibration_plan_fingerprint") != plan["fingerprint"]:
        raise ValueError("interval settings were not bound before calibration replay")
    if plan["mode"] == "research" and report["debug_only"]:
        raise ValueError("research calibration rejects debug models or plumbing-only data")
    signature, parents = _cohort_structure(report)
    scores = {parent: 0. for parent in parents}
    for index, episode in enumerate(report["episodes"]):
        for row in episode["rows"]:
            if guard:
                guard.check()
            plate, maps = load_case(evaluation, episode, index, row["records"])
            truth, mean, scale = _aligned_maps(plate, maps, plan["std_floor_mm"])
            score = float(np.max(abs(truth - mean) / scale))
            parent = episode["parent_specimen_id"]
            scores[parent] = max(scores[parent], score)
    quantile, rank = corrected_quantile(list(scores.values()), plan["alpha"])
    result = {"schema_version": 1, "kind": "parent_conformal_calibration", "passed": True,
              "alpha": plan["alpha"], "nominal_coverage": 1 - plan["alpha"], "std_floor_mm": plan["std_floor_mm"],
              "mode": plan["mode"], "debug_only": report["debug_only"] or plan["mode"] == "debug",
              "quantile": quantile, "finite_quantile": quantile is not None, "order_statistic_rank": rank,
              "independent_parents": len(parents), "parent_scores": scores,
              "calibration_parents": parents, "cohort_signature": signature,
              "calibration_plan_fingerprint": plan["fingerprint"], "protocol": plan["protocol"],
              "calibration_evaluation_sha256": sha256_file(Path(evaluation) / "evaluation.json"),
              "calibration_source_hash": _source_hash(), "scope": plan["scope"],
              "assumptions": ["exchangeable independent calibration/new parents from the same distribution",
                              "fixed model/selector/settings chosen without calibration labels",
                              "one scan per parent and unchanged finite inspection protocol"],
              "note": "Mathematical coverage is conditional on these assumptions; they and physical applicability are not certified. A null quantile returns the entire physical depth range.",
              "resources": guard.report() if guard else None}
    result["fingerprint"] = canonical_hash(result)
    write_json(output, result)
    return result


def prediction_interval(mean_mm, std_mm, nominal_mm: float, calibration: dict):
    """Label-free bounds after the caller has validated the bound protocol."""
    mean, spread = np.asarray(mean_mm, dtype=float), np.asarray(std_mm, dtype=float)
    if mean.ndim != 2 or mean.shape != spread.shape or not all(np.isfinite(a).all() for a in (mean, spread)):
        raise ValueError("mean/spread must be matching finite 2D maps")
    if not math.isfinite(nominal_mm) or nominal_mm <= 0 or np.any(mean < 0) or np.any(mean > nominal_mm) or np.any(spread < 0):
        raise ValueError("interval inputs are outside the physical range")
    floor, quantile = calibration["std_floor_mm"], calibration["quantile"]
    if not math.isfinite(floor) or not 0 < floor <= nominal_mm:
        raise ValueError("spread floor must be positive and no larger than nominal depth")
    if quantile is not None and (not math.isfinite(quantile) or quantile < 0):
        raise ValueError("quantile must be finite/nonnegative or null for the full physical range")
    if quantile is None:
        return np.zeros_like(mean), np.full_like(mean, nominal_mm)
    width = quantile * np.maximum(spread, floor)
    # Keep exact score ties covered despite division/multiplication rounding.
    width = width + 8 * np.finfo(float).eps * nominal_mm
    return np.maximum(0, mean - width), np.minimum(nominal_mm, mean + width)


def evaluate_calibration(calibration_path: Path, evaluation: Path, output: Path,
                          guard: ResourceGuard | None = None) -> dict:
    calibration = _read_fingerprinted(calibration_path, "parent_conformal_calibration")
    report = load_evaluation(evaluation)
    if report["plan"]["split"] not in ("test", "physical_holdout"):
        raise ValueError("coverage assessment requires final test/physical_holdout replay, not calibration or model-selection parents")
    signature, parents = _cohort_structure(report)
    if (canonical_hash(_inspection_protocol(report["plan"])) != canonical_hash(calibration["protocol"])
            or signature != calibration["cohort_signature"]):
        raise ValueError("calibration and assessment model, policy, budget, domain, acquisition or grid changed")
    if set(parents) & set(calibration["calibration_parents"]):
        raise ValueError("calibration parents cannot be reused for held-out coverage assessment")
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("interval assessment output must be new or empty")
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    result = {"schema_version": 1, "kind": "heldout_conformal_assessment", "passed": False,
              "debug_only": calibration["debug_only"] or report["debug_only"],
              "calibration_sha256": sha256_file(calibration_path),
              "evaluation_sha256": sha256_file(Path(evaluation) / "evaluation.json"),
              "nominal_coverage": calibration["nominal_coverage"], "finite_quantile": calibration["finite_quantile"],
              "scope": calibration["scope"], "rows": [], "parent_simultaneous_coverage": dict.fromkeys(parents, True),
              "interval_files": {}, "note": "Empirical coverage on held-out parents, not proof of exchangeability or physical validity; no settings may be tuned on these outcomes."}
    try:
        for index, episode in enumerate(report["episodes"]):
            intervals = {}
            for row in episode["rows"]:
                if guard:
                    guard.check()
                records = row["records"]
                plate, maps = load_case(evaluation, episode, index, records)
                truth, mean, scale = _aligned_maps(plate, maps, calibration["std_floor_mm"])
                lower, upper = prediction_interval(mean, scale, plate.thickness_m * 1000, calibration)
                covered = (truth >= lower) & (truth <= upper)
                parent = episode["parent_specimen_id"]
                result["parent_simultaneous_coverage"][parent] &= bool(covered.all())
                result["rows"].append({"case": index, "parent_specimen_id": parent, "specimen_id": episode["specimen_id"],
                                       "policy": _policy_identity(episode["policy"]), "records": records,
                                       "pixel_coverage": float(covered.mean()), "whole_map_covered": bool(covered.all()),
                                       "defect_pixel_coverage": float(covered[truth > 0].mean()) if np.any(truth > 0) else None,
                                       "mean_interval_width_mm": float((upper - lower).mean()),
                                       "full_span_fraction": float(((lower == 0) & (upper == plate.thickness_m * 1000)).mean())})
                intervals[f"lower_{records}_mm"], intervals[f"upper_{records}_mm"] = lower, upper
                if sum(a.nbytes for a in intervals.values()) > 128 * 1024**2:
                    raise MemoryError("one case exceeds the 128 MiB interval-map limit")
            path = output / f"case-{index:05d}-intervals.npz"
            atomic_write(path, lambda stream: np.savez_compressed(stream, **intervals))
            result["interval_files"][path.name] = sha256_file(path)
        result["independent_parents"] = len(parents)
        result["empirical_parent_simultaneous_coverage"] = float(np.mean(list(result["parent_simultaneous_coverage"].values())))
        result["empirical_coverage_meets_nominal"] = result["empirical_parent_simultaneous_coverage"] >= result["nominal_coverage"]
        result["passed"] = True  # Software execution status, not statistical acceptance.
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        result["wall_seconds"] = time.perf_counter() - started
        result["resources"] = guard.report() if guard else None
        result["fingerprint"] = canonical_hash(result)
        write_json(output / "coverage.json", result)
    return result
