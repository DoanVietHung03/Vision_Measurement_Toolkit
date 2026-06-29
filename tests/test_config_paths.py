from __future__ import annotations

import sys
import unittest
from glob import glob
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from vision_measurement.core.config import load_json_config
from vision_measurement.core.paths import project_root, resolve_path


class ConfigPathTests(unittest.TestCase):
    def test_project_root_points_to_workspace(self) -> None:
        self.assertEqual(project_root(), ROOT)

    def test_resolve_config_path(self) -> None:
        path = resolve_path("configs/distance_homography.json")
        self.assertEqual(path, ROOT / "configs" / "distance_homography.json")
        config = load_json_config(path)
        self.assertIn("calibration", config)

    def test_bundled_image_and_calibration_glob_exist(self) -> None:
        homography = load_json_config(resolve_path("configs/distance_homography.json"))
        image = resolve_path(homography["input"]["image_path"])
        self.assertTrue(image.is_file())

        camera = load_json_config(resolve_path("configs/camera_calibration.json"))
        pattern = str(resolve_path(camera["input"]["image_glob"]))
        self.assertGreater(len(glob(pattern)), 0)


if __name__ == "__main__":
    unittest.main()
