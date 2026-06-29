"""Homography-based distance measurement."""

from __future__ import annotations

from typing import Any

from vision_measurement.calibration import PlaneCalibration
from vision_measurement.core.results import DistanceResult


def measure_plane_distance(
    calibration: PlaneCalibration,
    point_a_px: tuple[float, float],
    point_b_px: tuple[float, float],
    metadata: dict[str, Any] | None = None,
) -> DistanceResult:
    """Measure distance between two image points on a calibrated plane."""
    return DistanceResult(
        method="homography",
        distance_m=calibration.distance_m(point_a_px, point_b_px),
        points_px=(point_a_px, point_b_px),
        metadata=metadata or {},
    )
