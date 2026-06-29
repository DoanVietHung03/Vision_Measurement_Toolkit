from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from vision_measurement.speed import optimize_lane_calibration


class SpeedCalibrationTests(unittest.TestCase):
    def test_optimizer_recovers_uniform_scale(self) -> None:
        measurements = [
            {"p1": [0, 0], "p2": [100, 0], "distance_m": 10.0},
            {"p1": [0, 0], "p2": [0, 100], "distance_m": 10.0},
            {"p1": [0, 0], "p2": [100, 100], "distance_m": 10.0 * 2**0.5},
        ]
        result = optimize_lane_calibration(
            source_points=[[0, 0], [100, 0], [100, 100], [0, 100]],
            measurements=measurements,
            target_width=100,
            ratio_min=1.0,
            ratio_max=1.0,
            ratio_steps=1,
        )
        self.assertAlmostEqual(result.meters_per_pixel, 0.1, places=6)
        self.assertAlmostEqual(result.mean_error_pct, 0.0, places=6)


if __name__ == "__main__":
    unittest.main()
