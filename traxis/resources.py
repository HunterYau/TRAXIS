"""Bounded CPU runs with best-effort hardware telemetry."""

from dataclasses import asdict, dataclass
import math
import shutil
import subprocess
import time

import psutil


class ResourceLimitError(RuntimeError):
    pass


@dataclass(frozen=True)
class ResourceLimits:
    max_seconds: float = 60
    max_rss_mib: float = 512
    max_system_memory_percent: float = 92
    min_available_mib: float = 768
    max_gpu_memory_percent: float = 90
    max_gpu_temperature_c: float = 80
    max_cpu_temperature_c: float = 90

    def __post_init__(self):
        if any(not math.isfinite(v) or v <= 0 for v in asdict(self).values()):
            raise ValueError("resource limits must be finite and positive")
        if self.max_system_memory_percent > 100 or self.max_gpu_memory_percent > 100:
            raise ValueError("memory percentages cannot exceed 100")


def snapshot(include_gpu: bool = True) -> dict:
    memory = psutil.virtual_memory()
    result = {"system_memory_percent": memory.percent,
              "available_mib": memory.available / 1024**2,
              "process_rss_mib": psutil.Process().memory_info().rss / 1024**2,
              "cpu_temperatures_c": None, "gpus": [], "gpu_status": "disabled"}
    temperatures = getattr(psutil, "sensors_temperatures", lambda: {})()
    values = [t.current for group in temperatures.values() for t in group
              if t.current is not None and math.isfinite(t.current)]
    result["cpu_temperatures_c"] = values or None
    if include_gpu:
        executable = shutil.which("nvidia-smi")
        result["gpu_status"] = "unavailable"
        if executable:
            try:
                p = subprocess.run([executable, "--query-gpu=name,memory.total,memory.used,temperature.gpu",
                                    "--format=csv,noheader,nounits"], capture_output=True, text=True,
                                   timeout=3, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0), check=True)
                for row in p.stdout.strip().splitlines():
                    name, total, used, temp = [part.strip() for part in row.split(",")]
                    result["gpus"].append({"name": name, "total_mib": float(total),
                                           "used_mib": float(used), "temperature_c": float(temp)})
                result["gpu_status"] = "available"
            except (OSError, ValueError, subprocess.SubprocessError):
                result["gpu_status"] = "query_failed"
    return result


class ResourceGuard:
    """Cooperative guard; call at each bounded assembly/time-step batch.

    It cannot preempt a blocked native kernel. Preflight allocation limits are
    also required. GPU reads are throttled; this code never allocates GPU data.
    """

    def __init__(self, limits: ResourceLimits | None = None):
        self.limits = limits or ResourceLimits()
        self.started = time.monotonic()
        self.last_check = -math.inf
        self.last_gpu_check = -math.inf
        self.history = []
        self.last_gpus = []

    def check(self, force: bool = False) -> None:
        elapsed = time.monotonic() - self.started
        if elapsed > self.limits.max_seconds:
            raise ResourceLimitError("wall-time limit reached")
        if not force and elapsed - self.last_check < 0.25:
            return
        include_gpu = elapsed - self.last_gpu_check >= 2
        sample = snapshot(include_gpu)
        if include_gpu:
            self.last_gpu_check = elapsed
            self.last_gpus = sample["gpus"]
        sample["elapsed_seconds"] = elapsed
        self.last_check = elapsed
        if len(self.history) < 10000:
            self.history.append(sample)
        self.enforce(sample, self.last_gpus)

    def enforce(self, sample: dict, gpus: list) -> None:
        limits = self.limits
        checks = [(sample["process_rss_mib"] > limits.max_rss_mib, "process RAM"),
                  (sample["system_memory_percent"] > limits.max_system_memory_percent, "system RAM"),
                  (sample["available_mib"] < limits.min_available_mib, "available RAM")]
        for gpu in gpus:
            checks += [(gpu["used_mib"] / gpu["total_mib"] * 100 > limits.max_gpu_memory_percent, "GPU memory"),
                       (gpu["temperature_c"] > limits.max_gpu_temperature_c, "GPU temperature")]
        checks += [(t > limits.max_cpu_temperature_c, "CPU temperature")
                   for t in sample["cpu_temperatures_c"] or []]
        for exceeded, label in checks:
            if exceeded:
                detail = (f" ({sample['process_rss_mib']:.1f} MiB > {limits.max_rss_mib:.1f} MiB)"
                    if label == 'process RAM' else '')
                raise ResourceLimitError(f"{label} limit reached{detail}")

    def report(self) -> dict:
        return {"limits": asdict(self.limits), "samples": self.history,
                "cpu_temperature_note": "null means no supported sensor; temperature was not verified"}
