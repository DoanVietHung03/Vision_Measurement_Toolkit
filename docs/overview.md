# Overview

The toolkit is organized into reusable package code and executable apps:

- `src/vision_measurement/core/`: config, paths, results, video, profiling.
- `src/vision_measurement/calibration/`: camera, plane, and stabilization.
- `src/vision_measurement/distance/`: Homography, height, person, and depth.
- `src/vision_measurement/speed/`: estimator, lane helpers, validation, calibration.
- `apps/`: user-facing commands.
- `configs/`: runtime configuration.
- `assets/`: local images, videos, models, and ground truth.
- `tests/`: deterministic core and parity tests.

```text
image/video
  -> camera/plane calibration
  -> detection/tracking or selected pixels
  -> metric projection
  -> distance, height, or speed estimate
  -> JSON/CSV/visualization
```
