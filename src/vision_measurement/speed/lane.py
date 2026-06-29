"""Vehicle lane speed pipeline helpers."""

from __future__ import annotations

from typing import Any

import numpy as np

from vision_measurement.calibration import PlaneCalibration
from vision_measurement.speed.estimator import SpeedEstimator


def bottom_center_from_xyxy(box_xyxy: tuple[float, float, float, float]) -> tuple[float, float]:
    x1, y1, x2, y2 = box_xyxy
    return ((float(x1) + float(x2)) / 2.0, float(y2))


def estimate_speeds_for_detections(
    calibration: PlaneCalibration,
    estimator: SpeedEstimator,
    detections: list[dict[str, Any]],
    frame_idx: int,
) -> list[dict[str, Any]]:
    """Estimate lane speeds for detections containing tracker_id and box_xyxy."""
    outputs = []
    for det in detections:
        if "tracker_id" not in det or "box_xyxy" not in det:
            continue
        foot_px = bottom_center_from_xyxy(tuple(det["box_xyxy"]))
        point_bev = calibration.transform_points([foot_px])[0]
        estimate = estimator.update(det["tracker_id"], point_bev, frame_idx=frame_idx)
        item = dict(det)
        item.update({"foot_px": foot_px, "point_bev": point_bev.tolist(), "speed": estimate})
        outputs.append(item)
    return outputs
