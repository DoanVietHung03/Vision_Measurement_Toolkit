# Migration Status

The distance and velocity repositories have been consolidated into the current
package and app layout. No Git command is part of this migration.

Transferred workflows:

- Camera calibration with reprojection diagnostics and debug undistortion.
- Homography image/video measurement, pose detection, height, stabilization,
  target tracking, interactive points, CSV, and latency reporting.
- Depth scale calibration, one-shot inference, realtime async video, YOLO
  person detection, click targets, 3D projection, and CSV.
- Lane BEV transform, speed estimation, ground-truth validation, and interactive
  calibration optimization.
- Frame extraction, pixel selection, ONNX export/validation, and method
  comparison experiments.

Large model/video/PKL artifacts remain optional runtime inputs. Their absence
must produce a clear path error and does not affect package import.
