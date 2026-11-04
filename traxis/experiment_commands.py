"""CLI operations kept separate from the lightweight physics commands."""

import json
from pathlib import Path

from traxis.dataset import BankDataset
from traxis.imaging import DASSettings
from traxis.storage import write_json


def read_settings(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def train_command(args, guard):
    from traxis.models import ModelConfig
    from traxis.training import TrainConfig, TrainingSession
    dataset = BankDataset.load(args.dataset)
    args.output.mkdir(parents=True, exist_ok=True)
    if args.resume:
        if args.config or args.adapt_from:
            raise ValueError("resume restores recorded configuration; do not supply a new config or adaptation source")
        session = TrainingSession.resume(args.resume, dataset, guard)
    else:
        if args.config is None:
            raise ValueError("new training runs require an explicit configuration file")
        if any(args.output.iterdir()):
            raise FileExistsError("new training runs need a new or empty output directory")
        spec = read_settings(args.config)
        config, model = TrainConfig(**spec["training"]), ModelConfig(**spec["model"])
        das = DASSettings.from_dict(spec["das"]) if spec.get("das") else None
        if args.adapt_from:
            session = TrainingSession.adapt(args.adapt_from, dataset, config, das, guard)
            if session.model_config != model:
                raise ValueError("adaptation configuration must match the existing reconstruction architecture")
        else:
            session = TrainingSession(dataset, model, config, das, guard=guard)
    report = {"kind": "subset_training_chunk", "passed": False}
    try:
        session.run(args.steps, args.output / "latest.pt")
        selected = args.output / f"selected-step-{session.step_count}.pt"
        session.export_model(selected)
        report.update({"passed": True, "selected_model": str(selected)})
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        report.update({"provenance": session.provenance(), "history": session.history,
                       "validation_history": session.validation_history, "resources": guard.report()})
        write_json(args.output / "training.json", report, overwrite=bool(args.resume))
    return {"passed": report["passed"], "updates_total": session.step_count,
            "model": report["selected_model"], "mode": session.config.mode}


def train_policy_command(args, guard):
    from traxis.adaptive import PolicyConfig
    from traxis.policy_training import PolicyTrainingSession, RLTrainConfig
    dataset = BankDataset.load(args.dataset)
    args.output.mkdir(parents=True, exist_ok=True)
    if args.resume:
        if args.config or args.reconstructors:
            raise ValueError("policy resume restores recorded configuration and reconstruction artifacts")
        session = PolicyTrainingSession.resume(args.resume, dataset, guard)
    else:
        if args.config is None or not args.reconstructors:
            raise ValueError("new policy runs require a config and reconstruction artifacts")
        if any(args.output.iterdir()):
            raise FileExistsError("new policy runs need a new or empty output directory")
        settings = read_settings(args.config)
        session = PolicyTrainingSession(dataset, args.reconstructors, PolicyConfig(**settings["policy"]),
                                         RLTrainConfig(**settings["training"]), guard)
    report = {"kind": "bounded_policy_training", "passed": False}
    try:
        session.run(args.episodes, args.output / "latest-policy.pt")
        selected = args.output / f"selected-episode-{session.episode_count}.pt"
        session.export_policy(selected)
        report.update({"passed": True, "selected_policy": str(selected)})
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        report.update({"provenance": session.provenance(), "history": session.history,
                       "validation_history": session.validation_history, "resources": guard.report()})
        write_json(args.output / "policy-training.json", report, overwrite=bool(args.resume))
    return {"passed": report["passed"], "episodes_total": session.episode_count,
            "policy": report["selected_policy"], "mode": session.config.mode}


def add_experiment_commands(sub):
    index = sub.add_parser("index-data", help="index and fingerprint waveform banks without loading all specimens together")
    index.add_argument("--banks-dir", type=Path, required=True)
    index.add_argument("--output", type=Path, required=True)
    train = sub.add_parser("train", help="run an explicit bounded number of subset updates")
    train.add_argument("--dataset", type=Path, required=True)
    train.add_argument("--config", type=Path)
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--steps", type=int, default=2)
    train.add_argument("--resume", type=Path)
    train.add_argument("--adapt-from", type=Path)
    policy = sub.add_parser("train-policy", help="bounded on-policy adaptive sensing with a frozen reconstructor")
    policy.add_argument("--dataset", type=Path, required=True)
    policy.add_argument("--config", type=Path)
    policy.add_argument("--reconstructors", type=Path, nargs="+")
    policy.add_argument("--output", type=Path, required=True)
    policy.add_argument("--episodes", type=int, default=1)
    policy.add_argument("--resume", type=Path)
    calibrate = sub.add_parser("calibrate-das", help="fit scalar DAS-to-depth conversion on training specimens only")
    calibrate.add_argument("--dataset", type=Path, required=True)
    calibrate.add_argument("--settings", type=Path, required=True)
    calibrate.add_argument("--budgets", type=int, nargs="+", required=True)
    calibrate.add_argument("--mode", choices=("debug", "research"), default="debug")
    calibrate.add_argument("--output", type=Path, required=True)
    freeze = sub.add_parser("freeze-evaluation", help="record model/data fingerprints, policies, thresholds and budgets before testing")
    freeze.add_argument("--dataset", type=Path, required=True)
    freeze.add_argument("--models", type=Path, nargs="+", required=True)
    freeze.add_argument("--split", choices=("validation", "test", "physical_holdout", "uncertainty_calibration"), required=True)
    freeze.add_argument("--budgets", type=int, nargs="+", required=True)
    freeze.add_argument("--initial-actions", type=int, nargs="*", default=[])
    freeze.add_argument("--threshold-mm", type=float, default=0.1)
    freeze.add_argument("--random-seeds", type=int, nargs="*", default=[17, 29, 43])
    freeze.add_argument("--ensemble", action="store_true")
    freeze.add_argument("--uncertainty-path", action="store_true")
    freeze.add_argument("--rl-policies", type=Path, nargs="*", default=[])
    freeze.add_argument("--output", type=Path, required=True)
    evaluate = sub.add_parser("evaluate", help="replay a frozen evaluation plan")
    evaluate.add_argument("--dataset", type=Path, required=True)
    evaluate.add_argument("--plan", type=Path, required=True)
    evaluate.add_argument("--calibration-plan", type=Path)
    evaluate.add_argument("--output", type=Path, required=True)
    for command in (index, train, policy, calibrate, freeze, evaluate):
        command.add_argument("--max-seconds", type=float, default=60)
        command.add_argument("--max-rss-mib", type=float, default=1024)


def run_experiment_command(args, guard) -> dict:
    guard.check(force=True)
    if args.command == "index-data":
        dataset = BankDataset.build(sorted(args.banks_dir.glob("*.npz")), args.output, guard)
        return {"specimens": len(dataset.entries), "dataset_fingerprint": dataset.fingerprint}
    if args.command == "train":
        return train_command(args, guard)
    if args.command == "train-policy":
        return train_policy_command(args, guard)
    dataset = BankDataset.load(args.dataset)
    if args.command == "calibrate-das":
        from traxis.das_baseline import fit_das_baseline
        fitted = fit_das_baseline(dataset, DASSettings.from_dict(read_settings(args.settings)),
                                  tuple(args.budgets), mode=args.mode, guard=guard)
        fitted.save(args.output)
        return {"output": str(args.output), "slope": fitted.slope_mm_per_signal_unit, "mode": args.mode}
    if args.command == "freeze-evaluation":
        from traxis.evaluation import freeze_evaluation_plan
        specs = ({"kind": "fixed"},) + tuple({"kind": "random", "seed": seed} for seed in args.random_seeds)
        if args.uncertainty_path:
            specs += ({"kind": "uncertainty_path"},)
        specs += tuple({"kind": "rl", "path": str(path)} for path in args.rl_policies)
        plan = freeze_evaluation_plan(args.output, dataset, args.models, split=args.split, budgets=tuple(args.budgets),
                                      initial_actions=tuple(args.initial_actions), threshold_mm=args.threshold_mm,
                                      policy_specs=specs, ensemble=args.ensemble)
        return {"output": str(args.output), "plan_fingerprint": plan["fingerprint"]}
    if args.command == "evaluate":
        from traxis.evaluation import run_frozen_evaluation
        report = run_frozen_evaluation(args.plan, dataset, args.output, guard, calibration_plan_path=args.calibration_plan)
        return {"passed": report["passed"], "episodes": len(report["episodes"]), "debug_only": report["debug_only"]}
    raise ValueError("unknown experiment command")
