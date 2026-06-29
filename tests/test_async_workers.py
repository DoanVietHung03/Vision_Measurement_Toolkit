from __future__ import annotations

import sys
import time
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from vision_measurement.distance.depth import AsyncDepthEstimator
from vision_measurement.distance.person import AsyncPersonDetector


def wait_for_output(worker, timeout_s: float = 1.0):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        output = worker.poll()
        if output is not None:
            return output
        time.sleep(0.01)
    return None


class FakeDepthEstimator:
    def predict(self, frame, process_width=None):
        return np.ones(frame.shape[:2], dtype=float) * 2.0


class FakePersonDetector:
    def detect(self, frame):
        return [{"box": [0, 0, 1, 1]}]


class AsyncWorkerTests(unittest.TestCase):
    def test_depth_worker_returns_latest_result(self) -> None:
        worker = AsyncDepthEstimator(FakeDepthEstimator(), process_width=8)
        try:
            self.assertTrue(worker.submit(3, np.zeros((4, 5, 3), dtype=np.uint8)))
            output = wait_for_output(worker)
            self.assertIsNotNone(output)
            self.assertEqual(output[0], 3)
            self.assertEqual(output[1].shape, (4, 5))
        finally:
            worker.close()

    def test_person_worker_returns_observations(self) -> None:
        worker = AsyncPersonDetector(FakePersonDetector())
        try:
            self.assertTrue(worker.submit(7, np.zeros((2, 2, 3), dtype=np.uint8)))
            output = wait_for_output(worker)
            self.assertIsNotNone(output)
            self.assertEqual(output, (7, [{"box": [0, 0, 1, 1]}]))
        finally:
            worker.close()


if __name__ == "__main__":
    unittest.main()
