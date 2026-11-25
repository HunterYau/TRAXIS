"""Count usable container CPU cores, respecting affinity and CPU quota."""
import math
import os
from pathlib import Path


def available_threads():
    count = len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else (os.cpu_count() or 1)
    # Standard container mounts: cgroup v2 and v1 (including combined CPU mounts).
    v2 = Path('/sys/fs/cgroup/cpu.max')
    if v2.exists():
        quota, period = v2.read_text().split()
        if quota != 'max': count = min(count, max(1, math.floor(int(quota)/int(period))))
    for mount in ('/sys/fs/cgroup/cpu', '/sys/fs/cgroup/cpu,cpuacct'):
        q, p = Path(mount)/'cpu.cfs_quota_us', Path(mount)/'cpu.cfs_period_us'
        if q.exists() and p.exists():
            quota, period = int(q.read_text()), int(p.read_text())
            if quota > 0: count = min(count, max(1, math.floor(quota/period)))
    return max(1, count)


if __name__ == '__main__':
    print(available_threads())
