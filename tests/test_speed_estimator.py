from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from vision_measurement.speed import SpeedEstimator


class SpeedEstimatorTests(unittest.TestCase):
    def test_constant_motion_converges_to_reasonable_speed(self) -> None:
        estimator = SpeedEstimator(fps=10.0, meters_per_pixel=0.1, alpha=0.5)
        estimate = None
        for frame_idx in range(40):
            estimate = estimator.update(track_id=1, position_xy=(frame_idx * 10.0, 0.0), frame_idx=frame_idx)
        self.assertIsNotNone(estimate)
        self.assertGreater(estimate.speed_kmh, 20.0)
        self.assertLess(estimate.speed_kmh, 50.0)

    def test_first_observation_is_zero(self) -> None:
        estimator = SpeedEstimator(fps=25.0, meters_per_pixel=0.04)
        estimate = estimator.update(track_id="car", position_xy=(0.0, 0.0), frame_idx=0)
        self.assertEqual(estimate.speed_kmh, 0.0)
        self.assertFalse(estimate.smoothed)


if __name__ == "__main__":
    unittest.main()
