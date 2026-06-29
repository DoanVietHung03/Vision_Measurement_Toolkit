# Speed Measurement

The lane pipeline tracks configured vehicle classes, filters detections through
a polygon zone, maps bottom-center points to BEV, and estimates speed with a
constant-velocity Kalman filter plus EMA smoothing.

Ground-truth PKL measurements can be used in two ways:

```powershell
python apps/run_speed_lane.py --validate-pkl assets/ground_truth/gt_data.pkl
python apps/calibrate_speed_lane.py --pkl assets/ground_truth/gt_data.pkl --interactive
```

The calibration app searches BEV aspect ratios and selects the target height
and meters-per-pixel with the lowest scale variation, then reports mean distance
error. The runtime app supports GUI annotations or JSON lines with
`--no-display`.
