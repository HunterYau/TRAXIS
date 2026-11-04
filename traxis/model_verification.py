"""Eight bounded single-step model checks, never a training cycle."""

from dataclasses import asdict
import gc
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from traxis.bank import ReplayEnvironment
from traxis.diagnostic_data import analytic_bank
from traxis.imaging import DASSettings, delay_and_sum
from traxis.models import ModelConfig, SignalScale, build_model, observation_batch, reconstruction_loss
from traxis.resources import ResourceGuard


def verify_models(output: Path, guard: ResourceGuard) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    report = {"kind": "single_step_debug_not_architecture_performance_comparison", "passed": False,
              "torch_version": str(torch.__version__), "device": "cpu", "seed": 2026,
              "data": "analytic point-scattering fixture; not elastodynamic training data",
              "updates_per_configuration": 1, "configurations": []}
    report["source_sha256"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in sorted(Path(__file__).parent.glob("*.py"))}
    predictions = {}
    try:
        guard.check(force=True)
        bank = analytic_bank()
        scale = SignalScale.fit([bank])
        env = ReplayEnvironment(bank)
        indices = (0, 3, 12, 17)
        for index in indices:
            state = env.reveal(index)
        batch = observation_batch([state], scale)
        report["observed_action_indices"] = list(indices)
        report["signal_scale"] = asdict(scale)
        for architecture in ("direct", "das_unet"):
            for size in (128, 256):
                truth = torch.from_numpy(bank.depth_mm.copy())[None, None]
                image = delay_and_sum(state, DASSettings({100000.: 2500.}, 25e-6, (size, size)), guard)
                das = torch.from_numpy(np.stack([image.intensity / scale.rms, image.coverage]))[None]
                for extra in (0, 1):
                    guard.check(force=True)
                    torch.manual_seed(report["seed"])
                    config = ModelConfig(architecture, size, extra_decoder_blocks=extra)
                    model = build_model(config).cpu()
                    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
                    started = time.perf_counter()
                    prediction = model(batch) if architecture == "direct" else model(batch, das)
                    loss = reconstruction_loss(prediction, truth)
                    loss_before = float(loss.detach())
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    gradients = [p.grad for p in model.parameters() if p.grad is not None]
                    finite_gradients = all(bool(torch.isfinite(g).all()) for g in gradients)
                    if not finite_gradients:
                        raise FloatingPointError("nonfinite model gradients")
                    gradient_norm = float(torch.sqrt(sum(g.square().sum() for g in gradients)))
                    optimizer.step()
                    step_seconds = time.perf_counter() - started
                    guard.check(force=True)
                    model.eval()
                    with torch.inference_mode():
                        started = time.perf_counter()
                        after = model(batch) if architecture == "direct" else model(batch, das)
                        inference_seconds = time.perf_counter() - started
                        loss_after = float(reconstruction_loss(after, truth))
                    entry = {"config": asdict(config), "parameters": sum(p.numel() for p in model.parameters()),
                             "loss_reference_shape": list(bank.depth_mm.shape),
                             "loss_before_mm2": loss_before, "loss_after_mm2": loss_after,
                             "gradient_norm": gradient_norm, "finite_gradients": finite_gradients,
                             "step_seconds": step_seconds, "inference_seconds": inference_seconds,
                             "passed": bool(torch.isfinite(after).all()) and loss_after < loss_before
                             and gradient_norm > 0 and tuple(after.shape) == (1, 1, size, size)}
                    report["configurations"].append(entry)
                    predictions[f"{architecture}_{size}_extra{extra}"] = after[0, 0].numpy().copy()
                    del prediction, after, loss, optimizer, model, gradients
                    gc.collect()
                    guard.check(force=True)
        np.savez_compressed(output / "single-step-debug-maps.npz", truth_mm=bank.depth_mm, **predictions)
        report["passed"] = len(report["configurations"]) == 8 and all(e["passed"] for e in report["configurations"])
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        report["resources"] = guard.report()
        (output / "model-report.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    return report
