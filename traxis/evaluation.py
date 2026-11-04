"""Equal-record-budget replay; labels remain on the evaluator side."""

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import platform
import time
from typing import TYPE_CHECKING

import numpy as np

from traxis.bank import ReplayEnvironment, WaveformBank
from traxis.config import positive
from traxis.dataset import BankDataset
from traxis.metrics import depth_metrics, resample_cell_map, uncertainty_diagnostics
from traxis.resources import ResourceGuard
from traxis.storage import atomic_write, canonical_hash, sha256_file, write_json

if TYPE_CHECKING:
    from traxis.reconstruction import Reconstructor


def _inference_source_hash():
    root = Path(__file__).parent
    return canonical_hash({name: sha256_file(root / name) for name in
                           ("evaluation.py", "reconstruction.py", "models.py", "adaptive.py", "policies.py",
                            "imaging.py", "das_baseline.py", "metrics.py", "bank.py", "dataset.py", "config.py", "acquisition.py")})


@dataclass(frozen=True)
class ReplayResult:
    report: dict
    maps: dict[str, np.ndarray]

    def save(self, directory: Path):
        directory = Path(directory)
        if directory.exists() and any(directory.iterdir()):
            raise FileExistsError("evaluation output directory must be new or empty")
        directory.mkdir(parents=True, exist_ok=True)
        atomic_write(directory / "maps.npz", lambda stream: np.savez_compressed(stream, **self.maps))
        self.report["maps_sha256"] = sha256_file(directory / "maps.npz")
        write_json(directory / "metrics.json", self.report)


def replay_episode(bank: WaveformBank, reconstructor: "Reconstructor", policy, budgets: tuple[int, ...],
                    *, initial_actions: tuple[int, ...] = (), threshold_mm: float = 0.1,
                    guard: ResourceGuard | None = None) -> ReplayResult:
    """Dense reconstruction is a reference prediction, never defect ground truth.

    All policies begin with the same explicitly supplied initial records and
    exclude acquired actions. No rewards, labels, errors or unselected signals
    are passed to `choose(observation, reconstruction)`.
    """
    positive("defect threshold", threshold_mm)
    total = len(bank.actions)
    if not budgets or any(type(n) is not int or not len(initial_actions) <= n <= total for n in budgets):
        raise ValueError("budgets must lie between initial observation count and full catalog size")
    if len(set(initial_actions)) != len(initial_actions):
        raise ValueError("initial actions must be unique")
    requested = sorted(set(budgets) | {total})
    env = ReplayEnvironment(bank)
    selected = []
    for i in initial_actions:
        env.reveal(i)
        selected.append(i)
    rows, maps = [], {"truth_mm": bank.depth_mm}
    reconstruction_seconds, selection_seconds = 0., 0.
    started = time.perf_counter()
    while True:
        if guard:
            guard.check()
        observation = env.state()
        tick = time.perf_counter()
        reconstruction = reconstructor.predict(observation)
        inference_seconds = time.perf_counter() - tick
        reconstruction_seconds += inference_seconds
        if observation.count in requested:
            metrics = depth_metrics(reconstruction.depth_mm, bank.depth_mm, bank.plate, threshold_mm)
            row = {"records": observation.count, "action_indices": selected.copy(), "metrics": metrics,
                   "reconstruction_seconds_cumulative": reconstruction_seconds,
                   "inference_seconds": inference_seconds,
                   "inference_calls_cumulative": observation.count - len(initial_actions) + 1,
                   "selector_seconds_cumulative": selection_seconds,
                   "unique_excitation_requests": len({(bank.actions[i].transmitter, bank.actions[i].frequency_hz) for i in selected})}
            maps[f"depth_{observation.count}_mm"] = reconstruction.depth_mm.copy()
            if reconstruction.std_mm is not None:
                spread = resample_cell_map(reconstruction.std_mm, bank.depth_mm.shape)
                mean = resample_cell_map(reconstruction.depth_mm, bank.depth_mm.shape)
                row["uncertainty"] = uncertainty_diagnostics(mean, spread, bank.depth_mm)
                maps[f"std_{observation.count}_mm"] = reconstruction.std_mm.copy()
            rows.append(row)
        if observation.count == total:
            break
        tick = time.perf_counter()
        action = policy.choose(observation, reconstruction)
        selection_seconds += time.perf_counter() - tick
        env.reveal(action)
        selected.append(action)
    dense_error = rows[-1]["metrics"]["mae_mm"]
    for row in rows:
        error = row["metrics"]["mae_mm"]
        row["dense_reference_mae_mm"] = dense_error
        row["absolute_excess_mae_mm"] = error - dense_error
        row["relative_mae_to_dense"] = error / dense_error if dense_error > 1e-8 else None
        row["within_10_percent_of_dense"] = error <= 1.1 * dense_error if dense_error > 1e-8 else None
    return ReplayResult({"kind": "offline_waveform_replay", "specimen_id": bank.metadata.specimen_id,
                         "parent_specimen_id": bank.metadata.parent_specimen_id, "split": bank.metadata.split,
                         "domain": bank.metadata.domain, "plate": asdict(bank.plate),
                         "sensors_xy_m": bank.sensors_xy_m.tolist(), "catalog_size": total,
                         "observable": bank.metadata.observable, "reference_shape": list(bank.depth_mm.shape),
                         "acquisition_signature": canonical_hash({"plate": asdict(bank.plate), "sensors": bank.sensors_xy_m.tolist(),
                                                                   "actions": [asdict(a) for a in bank.actions], "time_s": bank.time_s.tolist()}),
                         "data_validation_status": bank.metadata.validation_status,
                         "initial_actions": list(initial_actions), "threshold_mm": threshold_mm,
                         "rows": rows, "wall_seconds": time.perf_counter() - started,
                         "hardware_acquisition_seconds": None,
                         "note": "Record and excitation counts are replay accounting, not measured hardware time savings."}, maps)


def freeze_evaluation_plan(path: Path, dataset: BankDataset, model_paths: list[Path], *, split: str,
                            budgets: tuple[int, ...], initial_actions: tuple[int, ...] = (),
                            threshold_mm: float = 0.1, policy_specs: tuple[dict, ...] = ({"kind": "fixed"},),
                            ensemble: bool = False) -> dict:
    if split not in ("validation", "test", "physical_holdout", "uncertainty_calibration") or not dataset.select(split):
        raise ValueError("choose a nonempty evaluation split")
    positive("defect threshold", threshold_mm)
    if not model_paths or not policy_specs or not budgets or any(type(n) is not int or n < 0 for n in budgets):
        raise ValueError("evaluation plan needs models, policies and nonnegative budgets")
    normalized_policies = []
    for policy in policy_specs:
        if policy.get("kind") not in ("fixed", "random", "uncertainty_path", "rl"):
            raise ValueError("unknown policy in evaluation plan")
        if policy["kind"] == "random" and (type(policy.get("seed")) is not int or policy["seed"] < 0):
            raise ValueError("random policies require an explicit nonnegative seed")
        if policy["kind"] == "rl":
            artifact = Path(policy["path"]).resolve()
            normalized_policies.append({"kind": "rl", "path": str(artifact), "sha256": sha256_file(artifact)})
        elif policy["kind"] == "uncertainty_path":
            if not ensemble:
                raise ValueError("uncertainty-path evaluation requires reconstruction ensemble spread")
            samples = policy.get("path_samples", 16)
            if type(samples) is not int or not 4 <= samples <= 128:
                raise ValueError("path samples must be an integer in [4,128]")
            normalized_policies.append({"kind": "uncertainty_path", "path_samples": samples})
        else:
            normalized_policies.append(dict(policy))
    models = [{"path": str(Path(p).resolve()), "sha256": sha256_file(p),
               "kind": "das" if Path(p).suffix.lower() == ".json" else "neural"} for p in model_paths]
    if len({m["sha256"] for m in models}) != len(models):
        raise ValueError("model artifacts must be distinct")
    plan = {"schema_version": 1, "dataset_fingerprint": dataset.fingerprint, "split": split,
            "models": models, "policy_specs": normalized_policies, "budgets": list(budgets),
            "initial_actions": list(initial_actions), "threshold_mm": threshold_mm,
            "selection_frozen": True, "ensemble": ensemble, "inference_source_hash": _inference_source_hash()}
    for spec in normalized_policies:
        if spec["kind"] == "rl":
            from traxis.adaptive import ActorCriticPolicy
            validate_selector_provenance(plan, dataset, ActorCriticPolicy.load(Path(spec["path"])).provenance)
    plan["fingerprint"] = canonical_hash(plan)
    write_json(path, plan)
    return plan


def validate_evaluation_plan(plan: dict, dataset: BankDataset, provenances: list[dict]):
    body = {k: v for k, v in plan.items() if k != "fingerprint"}
    if canonical_hash(body) != plan["fingerprint"] or plan["dataset_fingerprint"] != dataset.fingerprint:
        raise ValueError("frozen evaluation plan or dataset changed")
    if plan.get("inference_source_hash") != _inference_source_hash():
        raise ValueError("inference source changed; use the recorded revision or freeze a new evaluation plan")
    if not plan["selection_frozen"]:
        raise ValueError("evaluation settings must be frozen")
    for artifact in plan["models"]:
        if sha256_file(Path(artifact["path"])) != artifact["sha256"]:
            raise ValueError("model changed after evaluation settings were frozen")
    for spec in plan["policy_specs"]:
        if spec["kind"] == "rl" and sha256_file(Path(spec["path"])) != spec["sha256"]:
            raise ValueError("selector changed after evaluation settings were frozen")
    held_out = {e.parent_specimen_id for e in dataset.select(plan["split"])}
    if len(provenances) != len(plan["models"]):
        raise ValueError("missing model training provenance")
    for provenance in provenances:
        if not {"fit_parents", "validation_parents"} <= provenance.keys():
            raise ValueError("model lacks split provenance")
        used = set(provenance["fit_parents"])
        if plan["split"] != "validation":
            used |= set(provenance["validation_parents"])
        if held_out & used:
            raise ValueError("held-out parent was used for fitting or model selection")


def validate_selector_provenance(plan: dict, dataset: BankDataset, provenance: dict):
    """Selectors inherit reconstruction exposure as well as their own fit roles."""
    if not {"fit_parents", "validation_parents", "reconstruction_artifacts", "training"} <= provenance.keys():
        raise ValueError("selector lacks training or reconstruction provenance")
    used = set(provenance["fit_parents"])
    if plan["split"] != "validation":
        used |= set(provenance["validation_parents"])
    held_out = {e.parent_specimen_id for e in dataset.select(plan["split"])}
    if used & held_out:
        raise ValueError("held-out parent was used to fit or select the sensing policy")
    bound = {a["sha256"] for a in provenance["reconstruction_artifacts"]}
    actual = {a["sha256"] for a in plan["models"]}
    if bound != actual or (len(actual) > 1 and not plan["ensemble"]):
        raise ValueError("RL policy is bound to a different reconstruction model/ensemble")
    if list(provenance["training"]["initial_actions"]) != list(plan["initial_actions"]):
        raise ValueError("evaluation initial observations differ from the RL training protocol")


def policy_group(spec: dict) -> str:
    """Pool random repeats, while preserving distinct learned policy artifacts."""
    if spec["kind"] == "rl":
        return "rl-" + spec["sha256"][:12]
    if spec["kind"] == "uncertainty_path":
        return "uncertainty_path-" + str(spec.get("path_samples", 16))
    return spec["kind"]


def summarize_episodes(episodes: list[dict]) -> list[dict]:
    """Average repeated scans/seeds within parents before across-plate summaries."""
    grouped = {}
    names = ("mae_mm", "rmse_mm", "defect_region_mae_mm", "dice", "iou", "centroid_error_mm",
             "false_positive_area_mm2", "loss_volume_error_mm3", "intact_region_mae_mm",
             "absolute_excess_mae_mm", "relative_mae_to_dense")
    for episode in episodes:
        for row in episode["rows"]:
            key = (episode["predictor"], policy_group(episode["policy"]), row["records"])
            parent = grouped.setdefault(key, {}).setdefault(episode["parent_specimen_id"], {})
            scan = parent.setdefault(episode["specimen_id"], {name: [] for name in names})
            for name in names:
                value = row.get(name) if name in ("absolute_excess_mae_mm", "relative_mae_to_dense") else row["metrics"].get(name)
                if value is not None:
                    scan[name].append(value)
    summaries = []
    for (predictor, policy, records), parents in sorted(grouped.items()):
        parent_scores = {}
        for parent, scans in parents.items():
            parent_scores[parent] = {}
            for name in names:
                scan_means = [float(np.mean(scan[name])) for scan in scans.values() if scan[name]]
                parent_scores[parent][name] = float(np.mean(scan_means)) if scan_means else None
        metrics = {}
        for name in names:
            values = [score[name] for score in parent_scores.values() if score[name] is not None]
            metrics[name] = {"mean": float(np.mean(values)) if values else None,
                             "between_parent_std": float(np.std(values, ddof=1)) if len(values) > 1 else None,
                             "independent_parents": len(values)}
        summaries.append({"predictor": predictor, "policy": policy, "records": records,
                          "metrics": metrics, "per_parent": parent_scores})
    return summaries


def run_frozen_evaluation(plan_path: Path, dataset: BankDataset, output: Path,
                           guard: ResourceGuard | None = None, *, calibration_plan_path: Path | None = None) -> dict:
    import torch
    from traxis.das_baseline import DASDepthCalibration
    from traxis.imaging import DASSettings
    from traxis.models import load_model
    from traxis.policies import FixedPolicy, RandomPolicy
    from traxis.reconstruction import EnsembleReconstructor, NeuralReconstructor
    from traxis.adaptive import ActorCriticPolicy, UncertaintyPathPolicy
    plan = json.loads(Path(plan_path).read_text(encoding="utf-8-sig"))
    calibration_plan_fingerprint = None
    if plan["split"] == "uncertainty_calibration":
        from traxis.uncertainty import _read_fingerprinted
        if calibration_plan_path is None:
            raise ValueError("freeze interval settings before replay and supply the calibration plan")
        declared = _read_fingerprinted(calibration_plan_path, "frozen_uncertainty_calibration_plan")
        if declared["evaluation_plan_fingerprint"] != plan["fingerprint"]:
            raise ValueError("calibration settings were frozen for a different replay plan")
        calibration_plan_fingerprint = declared["fingerprint"]
    elif calibration_plan_path is not None:
        raise ValueError("calibration settings apply only to reserved calibration replay")
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("evaluation run needs a new or empty output directory")
    output.mkdir(parents=True, exist_ok=True)
    predictors, provenances, descriptors = [], [], []
    # Load the actual bound artifacts; callers cannot substitute provenance.
    for artifact in plan["models"]:
        if guard:
            guard.check()
        if sha256_file(Path(artifact["path"])) != artifact["sha256"]:
            raise ValueError("model changed after plan freezing")
        if artifact["kind"] == "das":
            predictor = DASDepthCalibration.load(Path(artifact["path"]))
            provenance = predictor.provenance
            descriptor = {"kind": "das", "neural_parameters": 0, "calibrated_scalar_parameters": 1,
                          "config": json.loads(json.dumps(asdict(predictor.settings))), "device": "cpu"}
        else:
            model, scale, provenance = load_model(Path(artifact["path"]))
            das = DASSettings.from_dict(provenance["das_settings"]) if provenance.get("das_settings") else None
            predictor = NeuralReconstructor(model, scale, das, guard=guard)
            descriptor = {"kind": "neural", "neural_parameters": sum(p.numel() for p in model.parameters()),
                          "config": asdict(model.config), "device": "cpu"}
        descriptor.update({"artifact_sha256": artifact["sha256"], "artifact_bytes": Path(artifact["path"]).stat().st_size,
                           "seed": provenance.get("seed"), "mode": provenance.get("mode")})
        descriptors.append(descriptor)
        predictors.append(predictor)
        provenances.append(provenance)
    validate_evaluation_plan(plan, dataset, provenances)
    if plan["ensemble"]:
        if any(a["kind"] != "neural" for a in plan["models"]) or len({p.get("seed") for p in provenances}) != len(provenances):
            raise ValueError("ensembles require neural models with distinct recorded initialization seeds")
        predictors = [EnsembleReconstructor(predictors)]
        descriptors = [{"kind": "ensemble", "members": descriptors,
                        "neural_parameters": sum(d["neural_parameters"] for d in descriptors),
                        "device": "cpu", "config": descriptors[0]["config"]}]
    learned_policies = {}
    for i, spec in enumerate(plan["policy_specs"]):
        if spec["kind"] == "rl":
            policy = ActorCriticPolicy.load(Path(spec["path"]))
            validate_selector_provenance(plan, dataset, policy.provenance)
            learned_policies[i] = policy
    episodes = []
    report = {"schema_version": 2, "kind": "frozen_offline_replay", "passed": False, "plan_fingerprint": plan["fingerprint"],
              "dataset_fingerprint": dataset.fingerprint,
              "plan": plan, "predictors": {f"model-{i}": d for i, d in enumerate(descriptors)},
              "calibration_plan_fingerprint": calibration_plan_fingerprint,
              "runtime": {"python": platform.python_version(), "platform": platform.platform(),
                          "processor": platform.processor(), "torch": str(torch.__version__),
                          "torch_cpu_threads": torch.get_num_threads(), "numpy": np.__version__},
              "debug_only": any(p.get("mode") != "research" for p in provenances)
              or any(p.provenance.get("mode") != "research" for p in learned_policies.values())
              or any(e.validation_status == "plumbing_only" or e.observable == "arbitrary" for e in dataset.select(plan["split"])),
              "episodes": episodes}
    try:
        if calibration_plan_fingerprint and declared["mode"] == "research" and report["debug_only"]:
            raise ValueError("research calibration rejects debug models or plumbing-only data")
        for entry in dataset.select(plan["split"]):
            bank = dataset.read(entry)
            for model_index, predictor in enumerate(predictors):
                for policy_index, spec in enumerate(plan["policy_specs"]):
                    if spec["kind"] == "fixed":
                        policy = FixedPolicy()
                    elif spec["kind"] == "random":
                        policy = RandomPolicy(spec["seed"])
                    elif spec["kind"] == "uncertainty_path":
                        policy = UncertaintyPathPolicy(spec["path_samples"])
                    else:
                        policy = learned_policies[policy_index]
                    result = replay_episode(bank, predictor, policy, tuple(plan["budgets"]),
                                            initial_actions=tuple(plan["initial_actions"]), threshold_mm=plan["threshold_mm"], guard=guard)
                    result.report.update({"predictor": f"model-{model_index}", "policy": spec})
                    result.save(output / f"case-{len(episodes):05d}")
                    episodes.append(result.report)
        report["summary"] = summarize_episodes(episodes)
        report["passed"] = bool(episodes)
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        report["resources"] = guard.report() if guard else None
        report["fingerprint"] = canonical_hash(report)
        write_json(output / "evaluation.json", report)
    return report
