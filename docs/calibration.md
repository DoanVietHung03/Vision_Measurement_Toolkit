# Camera Calibration

Camera calibration reads the image glob from
`configs/camera_calibration.json`. Output includes camera matrix, distortion,
RMS, per-image reprojection errors, camera poses, accepted/rejected image lists,
and optional sensor/FOV values.

```powershell
python apps/run_camera_calibration.py --config configs/camera_calibration.json
python apps/run_camera_calibration.py --sensor-size-mm 5.6x4.2 --no-undistorted
```

The current checkerboard sample directory is
`assets/checkerBoard/calibration_images/`.
