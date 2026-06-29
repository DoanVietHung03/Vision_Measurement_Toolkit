"""Small latency profiler used by video workflows."""

from __future__ import annotations

import threading
from collections import defaultdict

import numpy as np


class LatencyProfiler:
    """Thread-safe latency recorder with summary statistics."""

    def __init__(self) -> None:
        self._records: dict[str, list[float]] = defaultdict(list)
        self._lock = threading.Lock()

    def update(self, name: str, elapsed_ms: float) -> None:
        with self._lock:
            self._records[name].append(float(elapsed_ms))

    def summary(self, skip_warmup: int = 5) -> dict[str, dict[str, float]]:
        output: dict[str, dict[str, float]] = {}
        with self._lock:
            items = list(self._records.items())
        for name, values in items:
            arr = np.array(values[skip_warmup:] if len(values) > skip_warmup else values, dtype=float)
            if arr.size == 0:
                continue
            output[name] = {
                "mean_ms": float(np.mean(arr)),
                "min_ms": float(np.min(arr)),
                "max_ms": float(np.max(arr)),
                "p99_ms": float(np.percentile(arr, 99)),
            }
        return output
