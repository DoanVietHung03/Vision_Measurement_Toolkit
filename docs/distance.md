# Distance Measurement

## Homography

The plane calibration maps four image points into either a rectangular BEV or a
reconstructed real-world quadrilateral. Distance is measured in the resulting
metric plane.

Image mode supports fixed config points and interactive two-click measurements.
Video mode additionally supports async pose detection, optical-flow
stabilization, KCF target tracking, person-to-target distance, height, and CSV.

```powershell
python apps/run_distance_homography.py --mode image --interactive
python apps/run_distance_homography.py --mode video --detect --height --interactive --csv-output output/homography.csv
```

## Monocular Depth

Metric depth uses `Z = scale_factor / raw_depth`, followed by pinhole
back-projection. Scale can be calibrated from a detected person's foot point
and a known camera-to-person distance.

```powershell
python apps/run_distance_depth.py --infer-depth
python apps/run_distance_depth.py --infer-depth --calibrate-person --real-distance-m 15.3
python apps/run_distance_depth.py --realtime-video --csv-output output/depth.csv
```
