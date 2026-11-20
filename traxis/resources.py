"""Bounded CPU runs with best-effort hardware telemetry."""

from dataclasses import asdict, dataclass
import math
import shutil
import subprocess
import time
import warnings

import psutil


class ResourceLimitError(RuntimeError):
    pass


@dataclass(frozen=True)
class ResourceLimits:
    # None disables an individual stop while retaining telemetry.
    max_seconds: float | None = 60
    max_rss_mib: float | None = 512
    max_system_memory_percent: float | None = 89.9
    min_available_mib: float | None = 768
    max_gpu_memory_percent: float | None = 90
    max_gpu_temperature_c: float | None = 80
    max_cpu_temperature_c: float | None = 90

    def __post_init__(self):
        for name, value in asdict(self).items():
            if value is None:
                continue
            if value is None or not math.isfinite(value) or value <= 0:
                raise ValueError("resource limits must be finite and positive, or None to disable")
        if (self.max_system_memory_percent is not None and self.max_system_memory_percent > 100
                or self.max_gpu_memory_percent is not None and self.max_gpu_memory_percent > 100):
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

    def __init__(self, limits: ResourceLimits | None = None, *, telemetry_errors_fatal=True):
        self.limits = limits or ResourceLimits()
        self.telemetry_errors_fatal = telemetry_errors_fatal
        self.telemetry_error = None
        self.started = time.monotonic()
        self.last_check = -math.inf
        self.last_gpu_check = -math.inf
        self.history = []
        self.last_gpus = []
        self.latest_sample = None

    def check(self, force: bool = False) -> None:
        elapsed = time.monotonic() - self.started
        if self.limits.max_seconds is not None and elapsed > self.limits.max_seconds:
            raise ResourceLimitError("wall-time limit reached")
        if not force and elapsed - self.last_check < 0.25:
            return
        include_gpu = elapsed - self.last_gpu_check >= 2
        try:
            sample = snapshot(include_gpu)
        except (OSError, ValueError, RuntimeError, psutil.Error, subprocess.SubprocessError) as error:
            if self.telemetry_errors_fatal:
                raise
            if self.telemetry_error is None:
                warnings.warn(f"Resource reading unavailable; simulation continues: {error}", RuntimeWarning)
            self.telemetry_error = str(error)
            self.latest_sample = None
            self.last_gpus = []
            self.last_check = elapsed
            return
        self.telemetry_error = None
        if include_gpu:
            self.last_gpu_check = elapsed
            self.last_gpus = sample["gpus"]
        sample["elapsed_seconds"] = elapsed
        self.last_check = elapsed
        self.latest_sample = sample
        if len(self.history) < 10000:
            self.history.append(sample)
        self.enforce(sample, self.last_gpus)

    def enforce(self, sample: dict, gpus: list) -> None:
        limits = self.limits
        checks = []
        if limits.max_rss_mib is not None:
            checks.append((sample["process_rss_mib"] > limits.max_rss_mib, "process RAM"))
        if limits.max_system_memory_percent is not None:
            checks.append((sample["system_memory_percent"] > limits.max_system_memory_percent, "system RAM"))
        if limits.min_available_mib is not None:
            checks.append((sample["available_mib"] < limits.min_available_mib, "available RAM"))
        for gpu in gpus:
            if limits.max_gpu_memory_percent is not None:
                checks.append((gpu["used_mib"] / gpu["total_mib"] * 100 > limits.max_gpu_memory_percent, "GPU memory"))
            if limits.max_gpu_temperature_c is not None:
                checks.append((gpu["temperature_c"] > limits.max_gpu_temperature_c, "GPU temperature"))
        if limits.max_cpu_temperature_c is not None:
            checks += [(t > limits.max_cpu_temperature_c, "CPU temperature")
                       for t in sample["cpu_temperatures_c"] or []]
        for exceeded, label in checks:
            if exceeded:
                detail = (f" ({sample['process_rss_mib']:.1f} MiB > {limits.max_rss_mib:.1f} MiB)"
                    if label == 'process RAM' else '')
                raise ResourceLimitError(f"{label} limit reached{detail}")

    def report(self) -> dict:
        return {"limits": asdict(self.limits), "samples": self.history,
                "latest_sample": self.latest_sample,
                "telemetry_error": self.telemetry_error,
                "cpu_temperature_note": "null means no supported sensor; temperature was not verified"}
