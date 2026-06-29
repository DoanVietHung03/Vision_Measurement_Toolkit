"""Planar calibration for homography and bird's-eye-view workflows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from vision_measurement.core.geometry import euclidean_distance, quadrilateral_from_sides


def _require_cv2():
    try:
        import cv2  # type: ignore
    except ImportError as exc:
        raise ImportError("OpenCV is required for plane calibration.") from exc
    return cv2


@dataclass
class PlaneCalibration:
    """Mapping between image pixels and a metric plane."""

    source_points: np.ndarray
    target_points: np.ndarray
    meters_per_pixel: float = 1.0

    def __post_init__(self) -> None:
        if self.source_points.shape != (4, 2):
            raise ValueError("source_points must have shape (4, 2)")
        if self.target_points.shape != (4, 2):
            raise ValueError("target_points must have shape (4, 2)")
        if self.meters_per_pixel <= 0:
            raise ValueError("meters_per_pixel must be positive")
        cv2 = _require_cv2()
        self.matrix = cv2.getPerspectiveTransform(
            self.source_points.astype(np.float32),
            self.target_points.astype(np.float32),
        )

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "PlaneCalibration":
        """Build calibration from a config dict.

        Accepted shapes:
        - calibration.source_points + target_width/target_height/meters_per_pixel
        - calibration.source_points + target_points + meters_per_pixel
        - calibration.source_points + real_world sides + scale_px_per_meter
        """
        cal = config.get("calibration", config)
        source_points = np.array(cal["source_points"], dtype=np.float32)

        if "target_points" in cal:
            target_points = np.array(cal["target_points"], dtype=np.float32)
            meters_per_pixel = float(cal.get("meters_per_pixel", 1.0))
            return cls(source_points, target_points, meters_per_pixel)

        if "real_world" in cal:
            real_world = cal["real_world"]
            scale_px_per_meter = float(cal.get("scale_px_per_meter", 1.0))
            real_points = quadrilateral_from_sides(
                float(real_world["L1"]),
                float(real_world["L2"]),
                float(real_world["L3"]),
                float(real_world["L4"]),
                float(real_world["diag_13"]),
            )
            target_points = np.array(
                [[x * scale_px_per_meter, y * scale_px_per_meter] for x, y in real_points],
                dtype=np.float32,
            )
            meters_per_pixel = 1.0 / scale_px_per_meter
            return cls(source_points, target_points, meters_per_pixel)

        target_width = float(cal["target_width"])
        target_height = float(cal["target_height"])
        target_points = np.array(
            [[0.0, 0.0], [target_width, 0.0], [target_width, target_height], [0.0, target_height]],
            dtype=np.float32,
        )
        meters_per_pixel = float(cal.get("meters_per_pixel", 1.0))
        return cls(source_points, target_points, meters_per_pixel)

    def transform_points(self, points_px: Any) -> np.ndarray:
        """Transform Nx2 image points into plane pixel coordinates."""
        cv2 = _require_cv2()
        points = np.array(points_px, dtype=np.float32).reshape(-1, 1, 2)
        transformed = cv2.perspectiveTransform(points, self.matrix)
        return transformed.reshape(-1, 2)

    def image_to_plane_m(self, points_px: Any) -> np.ndarray:
        """Transform image points into metric plane coordinates."""
        return self.transform_points(points_px) * self.meters_per_pixel

    def distance_m(self, point_a_px: Any, point_b_px: Any) -> float:
        """Measure metric distance between two image points on the calibrated plane."""
        plane_points = self.image_to_plane_m([point_a_px, point_b_px])
        return euclidean_distance(plane_points[0], plane_points[1])
