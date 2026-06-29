from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from vision_measurement.distance.depth import MetricDepthProjector, sample_depth


class DepthProjectorTests(unittest.TestCase):
    def test_invalid_focal_length_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            MetricDepthProjector(scale_factor=10.0, fx=0.0, fy=100.0, cx=50.0, cy=50.0)

    def test_pixel_to_3d_matches_scale_formula(self) -> None:
        projector = MetricDepthProjector(scale_factor=10.0, fx=100.0, fy=100.0, cx=50.0, cy=50.0)
        point = projector.pixel_to_3d(60.0, 70.0, 2.0)
        self.assertIsNotNone(point)
        self.assertAlmostEqual(float(point[2]), 5.0)
        self.assertAlmostEqual(float(point[0]), 0.5)
        self.assertAlmostEqual(float(point[1]), 1.0)

    def test_sample_depth_uses_median_patch(self) -> None:
        import numpy as np

        depth = np.ones((5, 5), dtype=float)
        depth[2, 2] = 100.0
        self.assertAlmostEqual(sample_depth(depth, (2, 2), patch_radius=1), 1.0)


if __name__ == "__main__":
    unittest.main()
