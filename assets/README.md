# Assets

Runtime artifacts are grouped here:

- `checkerBoard/calibration_images/`: current camera calibration images and the default smoke-test image.
- `sample_images/`: optional retained sample frames for real measurement runs.
- `weights/`: optional YOLO `.pt` or `.onnx` files.
- `sample_videos/`: optional input videos.
- `ground_truth/`: optional speed-validation PKL files.
- `calibration_debug/`: generated calibration visualizations.

Models and large videos are not required for package import or unit tests.
