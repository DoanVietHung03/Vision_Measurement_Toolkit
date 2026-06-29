# Source Migration Map

This map records where each legacy responsibility now lives.

| Legacy responsibility | Current replacement |
|---|---|
| Homography image UI and measurement | `apps/run_distance_homography.py` |
| Homography realtime video | `apps/run_distance_homography.py --mode video` |
| Optical-flow perspective stabilization | `src/vision_measurement/calibration/stabilizer.py` |
| Person pose/ground-point detection | `src/vision_measurement/distance/person.py` |
| Height estimation | `src/vision_measurement/distance/height.py` |
| Checkerboard camera calibration | `src/vision_measurement/calibration/camera.py` |
| Depth scale calibration | `apps/run_distance_depth.py --calibrate-person` |
| Depth realtime video | `apps/run_distance_depth.py --realtime-video` |
| Pixel-to-3D projection | `src/vision_measurement/distance/depth.py` |
| Vehicle lane runtime | `apps/run_speed_lane.py` |
| BEV transform | `src/vision_measurement/calibration/plane.py` |
| Kalman/EMA speed estimation | `src/vision_measurement/speed/estimator.py` |
| Ground-truth validation | `src/vision_measurement/speed/validation.py` |
| Lane calibration optimization | `src/vision_measurement/speed/calibration.py` and `apps/calibrate_speed_lane.py` |
| Frame extraction | `apps/extract_frame.py` |
| Pixel selection | `apps/pick_pixels.py` |
| YOLO ONNX export and validation | `apps/export_yolo_onnx.py` |
| Error comparison chart | `experiments/compare_distance_methods.py` |
| Historical measurements | `docs/legacy_measurements.md` |
| Historical chart | `experiments/reference_error_chart.png` |

Intentional changes:

- Runtime paths are JSON configuration instead of hard-coded local paths.
- YOLO and Depth models are loaded once and reused.
- Worker failures propagate to the main thread instead of failing silently.
- Missing calibration is strict by default for height measurement.
- ONNX export validates both the graph and one runtime inference.
- CSV files are rewritten per run with a stable union of row fields.
