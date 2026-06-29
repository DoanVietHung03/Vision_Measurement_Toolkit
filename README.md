# Vision Measurement Toolkit

Unified toolkit for camera calibration, planar distance, monocular-depth 3D
measurement, and lane vehicle speed.

## Verify

```powershell
python -m compileall src apps tests experiments
python -m unittest discover -s tests
```

## Camera Calibration

```powershell
python apps/run_camera_calibration.py --config configs/camera_calibration.json
```

## Homography Distance and Height

```powershell
python apps/run_distance_homography.py --config configs/distance_homography.json
python apps/run_distance_homography.py --mode image --interactive
python apps/run_distance_homography.py --mode video --detect --height --interactive
```

Video mode supports one-time model loading, asynchronous detection, optical-flow
calibration stabilization, click-selected KCF target tracking, CSV output, and
head/foot height estimation.

## Depth Distance

```powershell
python apps/run_distance_depth.py --config configs/distance_depth.json --no-display
python apps/run_distance_depth.py --infer-depth --calibrate-person --real-distance-m 15.3
python apps/run_distance_depth.py --realtime-video --video assets/sample_videos/cam_2.mp4
```

Realtime mode runs Depth Anything and YOLO workers independently, measures
manual and person-to-target 3D distances, supports click target selection, and
can write CSV output.

## Lane Speed

```powershell
python apps/calibrate_speed_lane.py --config configs/speed_vehicle_lane.json --interactive
python apps/run_speed_lane.py --config configs/speed_vehicle_lane.json --validate-pkl assets/ground_truth/gt_data.pkl
python apps/run_speed_lane.py --config configs/speed_vehicle_lane.json --no-display
```

## Utilities

```powershell
python apps/extract_frame.py --video input.mp4 --frame-index 10 --output frame.jpg
python apps/pick_pixels.py --image frame.jpg
python apps/export_yolo_onnx.py --model assets/weights/model.pt
python experiments/compare_distance_methods.py
```

Runtime models, videos, and ground-truth files are configured under
`configs/` and stored under `assets/` when kept in the workspace.
