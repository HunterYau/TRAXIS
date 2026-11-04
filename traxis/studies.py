"""Matched, bounded architecture studies with one active job per process."""

from contextlib import contextmanager
from dataclasses import asdict, replace
import gc
import itertools
import json
import os
from pathlib import Path
import time

import torch

from traxis.dataset import BankDataset
from traxis.imaging import DASSettings
from traxis.models import ModelConfig, SignalScale, build_model, load_model
from traxis.resources import ResourceGuard
from traxis.storage import canonical_hash, sha256_file, write_json
from traxis.training import TrainConfig, TrainingSession, fit_signal_scale


def _source_hash():
    root = Path(__file__).parent
    return canonical_hash({name: sha256_file(root / name) for name in
                           ("studies.py", "training.py", "models.py", "reconstruction.py", "das_baseline.py",
                            "imaging.py", "metrics.py", "dataset.py", "bank.py", "storage.py")})


def _json_value(value):
    return json.loads(json.dumps(value, allow_nan=False))


@contextmanager
def study_lock(root: Path):
    """OS lock is released on exit/crash; the persistent file is not a PID claim."""
    with (root / ".study.lock").open("a+b") as stream:
        if os.name == "nt":
            import msvcrt
            stream.seek(0, 2)
            if stream.tell() == 0:
                stream.write(b"\0")
                stream.flush()
            stream.seek(0)
            lock = lambda: msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            unlock = lambda: (stream.seek(0), msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1))
        else:
            import fcntl
            lock = lambda: fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            unlock = lambda: fcntl.flock(stream, fcntl.LOCK_UN)
        try:
            lock()
        except OSError as exc:
            raise RuntimeError("another process holds this study's execution lock") from exc
        try:
            yield
        finally:
            unlock()


def create_study(dataset_path: Path, specification: dict, output: Path,
                   guard: ResourceGuard | None = None) -> dict:
    """Plan a factorial study and train-only scale; perform no optimizer updates."""
    dataset = BankDataset.load(dataset_path)
    matrix = specification.get("matrix", {})
    architectures = tuple(matrix.get("architectures", ("direct", "das_unet")))
    sizes = tuple(matrix.get("output_sizes", (128, 256)))
    extras = tuple(matrix.get("extra_decoder_blocks", (0, 1)))
    seeds = tuple(matrix.get("seeds", (101, 202)))
    subset_seeds = tuple(matrix.get("subset_seeds", (37, 53)))
    for values in (architectures, sizes, extras, seeds, subset_seeds):
        if not values or len(set(values)) != len(values):
            raise ValueError("matrix axes must be nonempty and unique")
    if len(seeds) != len(subset_seeds) or len(seeds) > 8:
        raise ValueError("provide 1–8 paired initialization/subset seeds")
    if any(type(seed) is not int or seed < 0 for seed in seeds + subset_seeds):
        raise ValueError("initialization/subset seeds must be nonnegative integers")
    training = TrainConfig(**specification.get("training", {}))
    if training.phase != "pretrain" or training.run_id is not None:
        raise ValueError("study matrices pretrain from scratch and assign their own run identities")
    if training.max_steps > 100000:
        raise ValueError("one study job is limited to 100,000 explicit updates")
    # Fixed exposure is required for the primary architecture ablation. Best
    # validation weights may differ, but every job gets the same update budget.
    training = replace(training, patience=max(training.patience, training.max_steps + 1))
    das = DASSettings.from_dict(specification["das"]) if specification.get("das") else None
    if "das_unet" in architectures and das is None:
        raise ValueError("DAS-assisted variants require explicit imaging settings")
    cpu_threads = specification.get("cpu_threads", 1)
    if type(cpu_threads) is not int or not 1 <= cpu_threads <= 8:
        raise ValueError("study CPU threads must lie in [1,8]")
    eligible = dataset.select("train") + dataset.select("validation")
    if not dataset.select("train") or not dataset.select("validation"):
        raise ValueError("a matched study needs distinct nonempty train and validation splits")
    if len({e.observable for e in eligible}) != 1:
        raise ValueError("training/validation observations must use one signal unit")
    if training.mode == "research" and any(e.validation_status == "plumbing_only" or e.observable == "arbitrary" for e in eligible):
        raise ValueError("research studies require verified physical-unit data")
    reference_shapes, thicknesses, frequencies, catalogs = set(), set(), set(), []
    training_controls = set()
    for entry in eligible:
        if guard:
            guard.check()
        bank = dataset.read(entry)
        reference_shapes.add(bank.depth_mm.shape)
        thicknesses.add(bank.plate.thickness_m * 1000)
        frequencies.update(a.frequency_hz for a in bank.actions)
        catalogs.append(len(bank.actions))
        if entry.split == "train":
            training_controls.add("damaged" if bank.depth_mm.any() else "intact")
        del bank
    if training_controls != {"intact", "damaged"}:
        raise ValueError("matched studies require both intact and damaged training parents")
    if len(reference_shapes) != 1 or next(iter(reference_shapes)) != (512, 512) or len(thicknesses) != 1:
        raise ValueError("matched studies require a common 512 x 512 reference and nominal thickness")
    if training.min_records > min(catalogs) or training.validation_records > min(catalogs):
        raise ValueError("training/validation budgets exceed an eligible action catalog")
    if das and not frequencies <= das.group_velocity_m_s.keys():
        raise ValueError("DAS group speeds must cover all eligible excitation frequencies")
    jobs = []
    base = dict(specification.get("model", {}))
    if set(base) & {"architecture", "output_size", "extra_decoder_blocks"}:
        raise ValueError("put architecture, output size and decoder depth in the matrix axes")
    for architecture, size, extra, seed in itertools.product(architectures, sizes, extras, seeds):
        subset_seed = subset_seeds[seeds.index(seed)]
        config = ModelConfig(**base, architecture=architecture, output_size=size, extra_decoder_blocks=extra)
        if config.max_depth_mm != next(iter(thicknesses)):
            raise ValueError("model depth bound must match every training/validation plate")
        tag = "A" if architecture == "direct" else "B"
        jobs.append({"id": f"{tag}-r{size}-extra{extra}-seed{seed}", "kind": "neural", "replicate": seed,
                     "model": asdict(config), "training": asdict(replace(training, seed=seed, subset_seed=subset_seed)),
                     "das": _json_value(asdict(das)) if architecture == "das_unet" else None})
    if len(jobs) > 64 or (training.mode == "debug" and sum(j["training"]["max_steps"] for j in jobs) > 32):
        raise ValueError("studies are limited to 64 neural jobs and debug studies to 32 total updates")
    baseline = specification.get("baseline")
    if baseline is not None:
        if das is None:
            raise ValueError("DAS baseline jobs require imaging settings")
        budgets = tuple(baseline["budgets"])
        if not budgets or len(set(budgets)) != len(budgets) or any(type(b) is not int or not 1 <= b <= min(catalogs) for b in budgets):
            raise ValueError("baseline budgets must be unique positive counts within each eligible catalog")
        if training.mode == "debug" and len(dataset.select("train")) * len(budgets) > 16:
            raise ValueError("debug baseline fitting is limited to sixteen training/subset images per job")
        for size, (seed, subset_seed) in itertools.product(sizes, zip(seeds, subset_seeds)):
            jobs.append({"id": f"C-r{size}-seed{seed}", "kind": "das", "replicate": seed, "seed": subset_seed,
                         "mode": training.mode, "budgets": list(budgets), "damaged_weight": training.damaged_weight,
                         "das": _json_value(asdict(replace(das, shape=(size, size))))})
    pairs = []
    for a, b in itertools.combinations([j for j in jobs if j["kind"] == "neural"], 2):
        changed = [key for key in ("architecture", "output_size", "extra_decoder_blocks") if a["model"][key] != b["model"][key]]
        if a["replicate"] == b["replicate"] and len(changed) == 1:
            pairs.append({"factor": changed[0], "first": a["id"], "second": b["id"]})
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("study planning needs a new or empty directory")
    scale = fit_signal_scale(dataset, guard)
    plan = {"schema_version": 1, "kind": "matched_architecture_study", "source_hash": _source_hash(),
            "study_directory": str(output), "dataset_path": str(Path(dataset_path).resolve()),
            "dataset_fingerprint": dataset.fingerprint, "signal_scale": asdict(scale), "reference_shape": [512, 512],
            "cpu_threads": cpu_threads, "mode": training.mode, "jobs": jobs, "controlled_pairs": pairs,
            "loss_grid": "bank_reference_cell_averages", "fixed_neural_update_budget": training.max_steps,
            "training_parents": sorted({e.parent_specimen_id for e in dataset.select("train")}),
            "validation_parents": sorted({e.parent_specimen_id for e in dataset.select("validation")}),
            "note": "Planning fits only a training-signal RMS. No optimization, policy fitting or held-out label reading. Baselines fit all eligible training/subset images, not neural update draws."}
    plan["fingerprint"] = canonical_hash(plan)
    write_json(output / "study.json", plan)
    return plan


def load_study(path: Path) -> tuple[dict, BankDataset]:
    path = Path(path).resolve()
    if path.stat().st_size > 8 * 1024**2:
        raise MemoryError("study plan exceeds 8 MiB")
    plan = json.loads(path.read_text(encoding="utf-8-sig"))
    if plan.get("schema_version") != 1 or plan.get("kind") != "matched_architecture_study":
        raise ValueError("unsupported study plan")
    if canonical_hash({k: v for k, v in plan.items() if k != "fingerprint"}) != plan["fingerprint"]:
        raise ValueError("study plan fingerprint mismatch")
    if plan["source_hash"] != _source_hash() or Path(plan["study_directory"]) != path.parent:
        raise ValueError("study source or location changed; preserve the recorded revision/location or plan a new study")
    dataset = BankDataset.load(Path(plan["dataset_path"]))
    if dataset.fingerprint != plan["dataset_fingerprint"]:
        raise ValueError("study dataset changed")
    return plan, dataset


def _job_directory(plan, job):
    root = Path(plan["study_directory"]).resolve()
    target = (root / "jobs" / job["id"]).resolve()
    if not target.is_relative_to(root):
        raise ValueError("study job path escapes its directory")
    return target


def _load_job_state(plan, job):
    path = _job_directory(plan, job) / "job.json"
    if not path.exists():
        return None
    if path.stat().st_size > 64 * 1024**2:
        raise MemoryError("job progress exceeds 64 MiB")
    state = json.loads(path.read_text(encoding="utf-8-sig"))
    if state["study_fingerprint"] != plan["fingerprint"] or state["job_id"] != job["id"]:
        raise ValueError("job directory belongs to a different study")
    if canonical_hash({k: v for k, v in state.items() if k != "fingerprint"}) != state["fingerprint"]:
        raise ValueError("job progress fingerprint mismatch")
    return state


def _save_job_state(path, state):
    body = {k: v for k, v in state.items() if k != "fingerprint"}
    body["fingerprint"] = canonical_hash(body)
    write_json(path / "job.json", body, overwrite=(path / "job.json").exists())
    return body


def _exposure_prefixes(history):
    prefix, result = canonical_hash("matched-training-draws"), []
    for row in history:
        draw = {key: row[key] for key in ("step", "specimen_id", "records", "action_indices")}
        prefix = canonical_hash([prefix, draw])
        result.append(prefix)
    return result


def _check_matched_exposure(plan, job, prefixes):
    for other in plan["jobs"]:
        if other["kind"] != "neural" or other["replicate"] != job["replicate"] or other["id"] == job["id"]:
            continue
        state = _load_job_state(plan, other)
        if state:
            count = min(len(prefixes), len(state.get("exposure_prefixes", [])))
            if count and prefixes[count - 1] != state["exposure_prefixes"][count - 1]:
                raise ValueError("paired architecture jobs acquired different training observations")


def _advance_job(plan, dataset, job, updates, guard):
    root = _job_directory(plan, job)
    state = _load_job_state(plan, job)
    if state is None:
        if root.exists() and any(root.iterdir()):
            raise FileExistsError("job directory contains unowned files")
        root.mkdir(parents=True, exist_ok=True)
        state = _save_job_state(root, {"study_fingerprint": plan["fingerprint"], "job_id": job["id"],
                                       "completed": False, "steps": 0, "pending_validation": False})
    if state["completed"]:
        if sha256_file(root / state["artifact"]) != state["artifact_sha256"]:
            raise ValueError("completed study artifact changed")
        return state, 0
    session = None
    started = time.perf_counter()
    prior_wall_seconds = state.get("wall_seconds", 0.)
    try:
        if job["kind"] == "das":
            from traxis.das_baseline import fit_das_baseline
            fitted = fit_das_baseline(dataset, DASSettings.from_dict(job["das"]), tuple(job["budgets"]),
                                       seed=job["seed"], mode=job["mode"], damaged_weight=job["damaged_weight"], guard=guard)
            fitted.provenance.update({"study_fingerprint": plan["fingerprint"], "study_job": job["id"]})
            path = root / "selected.json"
            if path.exists():
                from traxis.das_baseline import DASDepthCalibration
                if asdict(DASDepthCalibration.load(path)) != asdict(fitted):
                    raise ValueError("existing baseline artifact disagrees with the recovered study fit")
            else:
                fitted.save(path)
            state.update({"completed": True, "artifact": path.name, "artifact_sha256": sha256_file(path),
                          "parameters": 1, "validation_history": [], "exposure_prefixes": [],
                          "wall_seconds": prior_wall_seconds + time.perf_counter() - started})
            state.pop("error", None)
            return _save_job_state(root, state), 0
        checkpoint = root / "latest.pt"
        config = replace(TrainConfig(**job["training"]), run_id=canonical_hash([plan["fingerprint"], job["id"]]))
        model_config = ModelConfig(**job["model"])
        das = DASSettings.from_dict(job["das"]) if job["das"] else None
        if checkpoint.exists():
            session = TrainingSession.resume(checkpoint, dataset, guard)
            if session.config != config or session.model_config != model_config or session.das_settings != das:
                raise ValueError("checkpoint belongs to a different study job")
        else:
            if state["steps"]:
                raise ValueError("study checkpoint is missing after completed updates")
            session = TrainingSession(dataset, model_config, config, das, guard=guard,
                                        _resume_scale=SignalScale(**plan["signal_scale"]))
        if session.scale != SignalScale(**plan["signal_scale"]):
            raise ValueError("study signal scale changed")
        before = session.step_count
        session.run(min(updates, config.max_steps - before), checkpoint)
        if session.step_count == config.max_steps and (not session.validation_history or session.validation_history[-1]["step"] != session.step_count):
            session.pending_validation = True
            session.save_checkpoint(checkpoint, overwrite=True)
            session.run(0, checkpoint)
        prefixes = _exposure_prefixes(session.history)
        _check_matched_exposure(plan, job, prefixes)
        completed = session.step_count == config.max_steps and not session.pending_validation
        state.update({"steps": session.step_count, "pending_validation": session.pending_validation,
                      "exposure_prefixes": prefixes, "validation_history": session.validation_history,
                      "loss_history": [{key: row[key] for key in ("step", "specimen_id", "records", "loss_mm2", "gradient_norm_before_clip")}
                                       for row in session.history],
                      "completed": completed, "parameters": sum(p.numel() for p in session.model.parameters())})
        if completed:
            path = root / "selected.pt"
            if path.exists():
                model, scale, provenance = load_model(path)
                expected = session.best_weights
                if (scale != session.scale or model.config != model_config or provenance.get("run_id") != config.run_id
                        or provenance.get("weights_step") != session.best_step
                        or any(not torch.equal(value, expected[name]) for name, value in model.state_dict().items())):
                    raise ValueError("existing artifact disagrees with recovered selected study weights")
                del model
            else:
                session.export_model(path)
            state.update({"artifact": path.name, "artifact_sha256": sha256_file(path), "selected_step": session.best_step})
        state.pop("error", None)
        state["wall_seconds"] = prior_wall_seconds + time.perf_counter() - started
        return _save_job_state(root, state), session.step_count - before
    except Exception as exc:
        state.update({"completed": False, "error": f"{type(exc).__name__}: {exc}"})
        state["wall_seconds"] = prior_wall_seconds + time.perf_counter() - started
        if session is not None:
            state.update({"steps": session.step_count, "pending_validation": session.pending_validation,
                          "exposure_prefixes": _exposure_prefixes(session.history), "validation_history": session.validation_history})
        _save_job_state(root, state)
        raise
    finally:
        del session
        gc.collect()


def advance_study(plan_path: Path, *, job_ids: tuple[str, ...] | None = None, max_new_updates: int = 1,
                    baseline_fits: int = 0, guard: ResourceGuard | None = None) -> dict:
    if type(max_new_updates) is not int or not 0 <= max_new_updates <= 100000 or type(baseline_fits) is not int or not 0 <= baseline_fits <= 16:
        raise ValueError("explicit update budget must lie in [0,100000] and baseline fits in [0,16]")
    plan, dataset = load_study(plan_path)
    requested = set(job_ids) if job_ids is not None else {j["id"] for j in plan["jobs"]}
    if not requested or not requested <= {j["id"] for j in plan["jobs"]}:
        raise ValueError("unknown/empty study job selection")
    torch.set_num_threads(plan["cpu_threads"])
    report = {"kind": "bounded_study_advance", "passed": False, "study_fingerprint": plan["fingerprint"],
              "new_updates": 0, "baseline_fits": 0, "jobs": []}
    with study_lock(Path(plan["study_directory"])):
        try:
            for job in plan["jobs"]:
                if job["id"] not in requested:
                    continue
                if guard:
                    guard.check()
                previous = _load_job_state(plan, job)
                if not (previous and previous["completed"]):
                    if job["kind"] == "neural" and report["new_updates"] >= max_new_updates and not (previous and previous["pending_validation"]):
                        continue
                    if job["kind"] == "das" and report["baseline_fits"] >= baseline_fits:
                        continue
                state, count = _advance_job(plan, dataset, job, max_new_updates - report["new_updates"], guard)
                report["new_updates"] += count
                report["baseline_fits"] += int(job["kind"] == "das" and not (previous and previous["completed"]))
                report["jobs"].append({"id": job["id"], "steps": state["steps"], "completed": state["completed"]})
                if guard:
                    guard.check(force=True)
            report["passed"] = True
        except Exception as exc:
            report["error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            report["resources"] = guard.report() if guard else None
            root = Path(plan["study_directory"])
            index = len(list(root.glob("advance-*.json")))
            write_json(root / f"advance-{index:05d}.json", report)
    return report


def freeze_study_comparisons(plan_path: Path, output: Path, *, split="validation", budgets=(2, 4),
                              random_seeds=(17, 29, 43), initial_actions=(), threshold_mm=.1,
                              guard: ResourceGuard | None = None) -> dict:
    from traxis.das_baseline import DASDepthCalibration
    from traxis.evaluation import freeze_evaluation_plan
    if split not in ("validation", "test"):
        raise ValueError("architecture studies freeze simulation validation/test comparisons; select configurations before physical evaluation")
    plan, dataset = load_study(plan_path)
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("comparison plans need a new or empty output directory")
    paths, ensemble_groups, descriptors = [], {}, []
    with study_lock(Path(plan["study_directory"])):
        for job in plan["jobs"]:
            if guard:
                guard.check()
            state = _load_job_state(plan, job)
            if not state or not state["completed"]:
                raise ValueError("complete every planned study job before freezing matched comparisons")
            path = _job_directory(plan, job) / state["artifact"]
            if sha256_file(path) != state["artifact_sha256"]:
                raise ValueError("study artifact changed")
            if job["kind"] == "neural":
                if state["steps"] != job["training"]["max_steps"]:
                    raise ValueError("study jobs have unequal configured exposure")
                _check_matched_exposure(plan, job, state["exposure_prefixes"])
                model, scale, provenance = load_model(path)
                if (model.config != ModelConfig(**job["model"]) or scale != SignalScale(**plan["signal_scale"])
                        or provenance.get("run_id") != canonical_hash([plan["fingerprint"], job["id"]])
                        or provenance.get("steps") != state["steps"] or provenance.get("loss_grid") != plan["loss_grid"]):
                    raise ValueError("study model provenance disagrees with the planned job")
                key = canonical_hash(job["model"])
                ensemble_groups.setdefault(key, []).append(path)
                del model
            else:
                fitted = DASDepthCalibration.load(path)
                if fitted.provenance.get("study_fingerprint") != plan["fingerprint"] or fitted.provenance.get("study_job") != job["id"]:
                    raise ValueError("baseline provenance disagrees with the planned job")
            paths.append(path)
            descriptors.append({"predictor": f"model-{len(paths)-1}", "job_id": job["id"],
                                "artifact_sha256": state["artifact_sha256"], "parameters": state["parameters"],
                                "compute_wall_seconds": state["wall_seconds"], "validation_history": state.get("validation_history", []),
                                "loss_history": state.get("loss_history", [])})
        specs = ({"kind": "fixed"},) + tuple({"kind": "random", "seed": seed} for seed in random_seeds)
        frozen = freeze_evaluation_plan(output / "individuals.json", dataset, paths, split=split, budgets=tuple(budgets),
                                        initial_actions=tuple(initial_actions), threshold_mm=threshold_mm, policy_specs=specs)
        ensembles = []
        for index, members in enumerate(ensemble_groups.values()):
            if len(members) < 2:
                continue
            path = output / f"ensemble-{index:02d}.json"
            ensemble = freeze_evaluation_plan(path, dataset, members, split=split, budgets=tuple(budgets),
                                               initial_actions=tuple(initial_actions), threshold_mm=threshold_mm,
                                               policy_specs=specs, ensemble=True)
            ensembles.append({"plan": path.name, "fingerprint": ensemble["fingerprint"], "members": [str(p) for p in members]})
        by_id = {job["id"]: (job, path) for job, path in zip(plan["jobs"], paths)}
        controlled = []
        for index, pair in enumerate(plan["controlled_pairs"]):
            first, second = by_id[pair["first"]], by_id[pair["second"]]
            selected = [first, second]
            if pair["factor"] == "architecture":
                selected += [(job, path) for job, path in by_id.values() if job["kind"] == "das"
                             and job["replicate"] == first[0]["replicate"]
                             and job["das"]["shape"] == [first[0]["model"]["output_size"]] * 2]
            path = output / f"controlled-pair-{index:03d}.json"
            comparison = freeze_evaluation_plan(path, dataset, [p for _, p in selected], split=split, budgets=tuple(budgets),
                                                  initial_actions=tuple(initial_actions), threshold_mm=threshold_mm, policy_specs=specs)
            controlled.append({**pair, "plan": path.name, "fingerprint": comparison["fingerprint"],
                               "predictors": {f"model-{i}": job["id"] for i, (job, _) in enumerate(selected)}})
        result = {"kind": "frozen_matched_study_comparisons", "study_fingerprint": plan["fingerprint"],
                  "individual_plan_fingerprint": frozen["fingerprint"], "predictors": descriptors,
                  "ensembles": ensembles, "controlled_pairs": controlled,
                  "note": "No replay has run. Architecture comparisons use fixed/repeated-random observations. Policy selection is a later stage bound to one selected model/ensemble."}
        write_json(output / "comparisons.json", result)
    return result
