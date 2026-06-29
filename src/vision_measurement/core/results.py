"""Public result objects returned by measurement pipelines."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class DistanceResult:
    method: str
    distance_m: float
    points_px: tuple[tuple[float, float], tuple[float, float]]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SpeedEstimate:
    track_id: int | str
    speed_kmh: float
    speed_mps: float
    smoothed: bool
    metadata: dict[str, Any] = field(default_factory=dict)
