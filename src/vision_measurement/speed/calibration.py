"""Ground-truth-assisted calibration for lane speed measurement."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from vision_measurement.calibration import PlaneCalibration


@dataclass(frozen=True)
class LaneCalibrationResult:
    source_points: list[list[float]]
    target_width: float
    target_height: float
    meters_per_pixel: float
    mean_error_pct: float
    coefficient_of_variation: float
    measurement_count: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "source_points": self.source_points,
            "target_width": self.target_width,
            "target_height": self.target_height,
            "meters_per_pixel": self.meters_per_pixel,
            "mean_error_pct": self.mean_error_pct,
            "coefficient_of_variation": self.coefficient_of_variation,
            "measurement_count": self.measurement_count,
        }


def optimize_lane_calibration(
    source_points: list[list[float]],
    measurements: list[dict[str, Any]],
    target_width: float = 800.0,
    ratio_min: float = 0.5,
    ratio_max: float = 15.0,
    ratio_steps: int = 150,
) -> LaneCalibrationResult:
    """Find the BEV aspect ratio and scale with the lowest MPP variation."""
    if len(source_points) != 4:
        raise ValueError("source_points must contain exactly four points")
    if not measurements:
        raise ValueError("At least one ground-truth distance measurement is required")
    if target_width <= 0 or ratio_min <= 0 or ratio_max < ratio_min or ratio_steps <= 0:
        raise ValueError("Invalid target width or ratio search range")

    best: tuple[float, float, float] | None = None
    for ratio in np.linspace(ratio_min, ratio_max, ratio_steps):
        target_height = float(round(target_width * float(ratio)))
        calibration = _pixel_plane(source_points, target_width, target_height)
        mpp_values = _measurement_mpp_values(calibration, measurements)
        if not mpp_values:
            continue
        mean_mpp = float(np.mean(mpp_values))
        coefficient = float(np.std(mpp_values) / mean_mpp) if mean_mpp else float("inf")
        if best is None or coefficient < best[0]:
            best = (coefficient, mean_mpp, target_height)

    if best is None:
        raise ValueError("Ground-truth measurements do not contain usable distances")

    coefficient, meters_per_pixel, target_height = best
    final_calibration = PlaneCalibration.from_config(
        {"calibration": {
            "source_points": source_points,
            "target_width": target_width,
            "target_height": target_height,
            "meters_per_pixel": meters_per_pixel,
        }}
    )
    errors = []
    for item in measurements:
        real_distance = float(item["distance_m"])
        if real_distance <= 0:
            continue
        estimate = final_calibration.distance_m(item["p1"], item["p2"])
        errors.append(abs(estimate - real_distance) / real_distance * 100.0)

    return LaneCalibrationResult(
        source_points=[[float(x), float(y)] for x, y in source_points],
        target_width=float(target_width),
        target_height=float(target_height),
        meters_per_pixel=float(meters_per_pixel),
        mean_error_pct=float(np.mean(errors)) if errors else 0.0,
        coefficient_of_variation=float(coefficient),
        measurement_count=len(errors),
    )


def _pixel_plane(source_points: list[list[float]], target_width: float, target_height: float) -> PlaneCalibration:
    return PlaneCalibration.from_config(
        {"calibration": {
            "source_points": source_points,
            "target_width": target_width,
            "target_height": target_height,
            "meters_per_pixel": 1.0,
        }}
    )


def _measurement_mpp_values(
    calibration: PlaneCalibration,
    measurements: list[dict[str, Any]],
) -> list[float]:
    values = []
    for item in measurements:
        real_distance = float(item["distance_m"])
        if real_distance <= 0:
            continue
        point_a, point_b = calibration.transform_points([item["p1"], item["p2"]])
        pixel_distance = float(np.linalg.norm(point_a - point_b))
        if pixel_distance > 1.0:
            values.append(real_distance / pixel_distance)
    return values
