from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from vision_measurement.calibration import PlaneCalibration
from vision_measurement.speed.estimator import SpeedEstimator
from vision_measurement.speed.lane import bottom_center_from_xyxy, estimate_speeds_for_detections


class SpeedLaneTests(unittest.TestCase):
    def test_bottom_center(self) -> None:
        self.assertEqual(bottom_center_from_xyxy((10, 20, 30, 80)), (20.0, 80.0))

    def test_estimate_speeds_for_detections(self) -> None:
        try:
            import cv2  # noqa: F401
        except ImportError:
            self.skipTest("OpenCV is not installed")
        calibration = PlaneCalibration.from_config({
            "calibration": {
                "source_points": [[0, 0], [100, 0], [100, 100], [0, 100]],
                "target_width": 100,
                "target_height": 100,
                "meters_per_pixel": 0.1,
            }
        })
        estimator = SpeedEstimator(fps=10, meters_per_pixel=0.1)
        detections = [{"tracker_id": 1, "box_xyxy": [0, 0, 10, 10]}]
        rows = estimate_speeds_for_detections(calibration, estimator, detections, frame_idx=0)
        self.assertEqual(len(rows), 1)
        self.assertIn("speed", rows[0])


if __name__ == "__main__":
    unittest.main()
