from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from vision_measurement.distance import HeightEstimator


class HeightEstimatorTests(unittest.TestCase):
    def test_missing_calibration_records_fallback(self) -> None:
        estimator = HeightEstimator(allow_fallback=True)
        loaded = estimator.load_focal_length(ROOT / "missing_calibration.json", current_width=1200)
        self.assertFalse(loaded)
        self.assertEqual(estimator.fx, 1200.0)
        self.assertIn("fallback_missing_calibration", estimator.status)

    def test_missing_calibration_can_be_strict(self) -> None:
        estimator = HeightEstimator(allow_fallback=False)
        with self.assertRaises(FileNotFoundError):
            estimator.load_focal_length(ROOT / "missing_calibration.json", current_width=1200)


if __name__ == "__main__":
    unittest.main()
