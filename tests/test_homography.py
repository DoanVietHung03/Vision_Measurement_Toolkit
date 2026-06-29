from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from vision_measurement.calibration import PlaneCalibration


class HomographyTests(unittest.TestCase):
    def setUp(self) -> None:
        try:
            import cv2  # noqa: F401
        except ImportError:
            self.skipTest("OpenCV is not installed")

    def test_identity_square_distance(self) -> None:
        calibration = PlaneCalibration.from_config({
            "calibration": {
                "source_points": [[0, 0], [100, 0], [100, 100], [0, 100]],
                "target_width": 100,
                "target_height": 100,
                "meters_per_pixel": 0.1,
            }
        })
        self.assertAlmostEqual(calibration.distance_m((0, 0), (100, 0)), 10.0, places=5)

    def test_transform_point(self) -> None:
        calibration = PlaneCalibration.from_config({
            "calibration": {
                "source_points": [[10, 10], [110, 10], [110, 110], [10, 110]],
                "target_width": 100,
                "target_height": 100,
                "meters_per_pixel": 1.0,
            }
        })
        point = calibration.transform_points([(60, 60)])[0]
        self.assertAlmostEqual(float(point[0]), 50.0, places=4)
        self.assertAlmostEqual(float(point[1]), 50.0, places=4)


if __name__ == "__main__":
    unittest.main()
