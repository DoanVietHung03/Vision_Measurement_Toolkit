from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
APPS = ROOT / "apps"
SRC = ROOT / "src"
for path in (APPS, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from run_distance_homography import distance_result_to_dict
from vision_measurement.core.results import DistanceResult


class AppSerializationTests(unittest.TestCase):
    def test_homography_result_converts_numpy_scalars(self) -> None:
        result = DistanceResult(
            method="homography",
            distance_m=3.0,
            points_px=(
                (np.float32(1.0), np.float32(2.0)),
                (np.float32(3.0), np.float32(4.0)),
            ),
        )
        payload = distance_result_to_dict(result)
        self.assertEqual(json.loads(json.dumps(payload))["point_a_px"], [1.0, 2.0])


if __name__ == "__main__":
    unittest.main()
