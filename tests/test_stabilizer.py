from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from vision_measurement.calibration import PerspectiveStabilizer


class StabilizerTests(unittest.TestCase):
    def test_transform_points_applies_homography(self) -> None:
        matrix = np.array([[1, 0, 12], [0, 1, -4], [0, 0, 1]], dtype=np.float32)
        transformed = PerspectiveStabilizer.transform_points([[10, 20], [30, 40]], matrix)
        np.testing.assert_allclose(transformed, [[22, 16], [42, 36]], atol=1e-6)


if __name__ == "__main__":
    unittest.main()
