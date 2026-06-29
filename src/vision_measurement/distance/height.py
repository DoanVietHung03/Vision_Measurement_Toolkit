"""Height estimation based on camera focal length and plane homography."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np


class HeightEstimator:
    """Estimate object height from head/foot pixels and calibrated floor plane."""

    def __init__(self, allow_fallback: bool = True) -> None:
        self.fx = 0.0
        self.fy = 0.0
        self.loaded = False
        self.allow_fallback = allow_fallback
        self.status = "not_loaded"

    def load_focal_length(self, calibration_json: str | Path, current_width: int) -> bool:
        """Load focal length from calibration JSON and scale it to current width.

        When ``allow_fallback`` is true, a missing calibration file falls back to
        ``fx = fy = current_width`` and records that state in ``status``. That
        keeps demo apps usable while making the degraded path explicit.
        """
        path = Path(calibration_json)
        if not path.exists():
            if not self.allow_fallback:
                raise FileNotFoundError(f"Camera calibration file not found: {path}")
            self.fx = float(current_width)
            self.fy = float(current_width)
            self.loaded = False
            self.status = f"fallback_missing_calibration:{path}"
            return False

        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        matrix = np.array(data["camera_matrix"], dtype=float)
        resolution = data.get("image_resolution") or [current_width, current_width]
        original_width = float(resolution[0])
        scale = float(current_width) / original_width if original_width else 1.0
        self.fx = float(matrix[0, 0]) * scale
        self.fy = float(matrix[1, 1]) * scale
        self.loaded = True
        self.status = "loaded"
        return True

    def calculate(
        self,
        head_pt: tuple[float, float],
        foot_pt: tuple[float, float],
        homography_matrix: Any,
        camera_plane_position_m: tuple[float, float],
    ) -> tuple[float, float]:
        """Return (height_m, distance_from_camera_m)."""
        if homography_matrix is None or self.fy == 0:
            return 0.0, 0.0
        try:
            import cv2  # type: ignore
        except ImportError as exc:
            raise ImportError("OpenCV is required for height estimation.") from exc

        h_pixel = math.hypot(head_pt[0] - foot_pt[0], head_pt[1] - foot_pt[1])
        foot_arr = np.array([[[foot_pt[0], foot_pt[1]]]], dtype=np.float32)
        real_pt = cv2.perspectiveTransform(foot_arr, homography_matrix)[0][0]

        cam_x, cam_y = camera_plane_position_m
        distance_m = math.hypot(float(real_pt[0]) - cam_x, float(real_pt[1]) - cam_y)
        height_m = h_pixel * (distance_m / self.fy)
        return float(height_m), float(distance_m)
