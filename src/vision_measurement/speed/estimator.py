"""Object speed estimation with Kalman filtering and EMA smoothing."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from vision_measurement.core.results import SpeedEstimate


@dataclass
class KalmanFilter:
    dt: float = 1.0 / 25.0
    process_noise: float = 1e-2
    measurement_noise: float = 1e-1

    def __post_init__(self) -> None:
        self.x = np.zeros((4, 1), dtype=float)
        self.f = np.array(
            [[1, 0, self.dt, 0], [0, 1, 0, self.dt], [0, 0, 1, 0], [0, 0, 0, 1]],
            dtype=float,
        )
        self.h = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=float)
        self.p = np.eye(4, dtype=float) * 1000.0
        self.q = np.eye(4, dtype=float) * self.process_noise
        self.r = np.eye(2, dtype=float) * self.measurement_noise

    def predict(self) -> np.ndarray:
        self.x = self.f @ self.x
        self.p = self.f @ self.p @ self.f.T + self.q
        return self.x

    def update(self, measurement_xy: tuple[float, float] | np.ndarray) -> None:
        z = np.array(measurement_xy, dtype=float).reshape(2, 1)
        innovation = z - self.h @ self.x
        s = self.h @ self.p @ self.h.T + self.r
        k = self.p @ self.h.T @ np.linalg.inv(s)
        self.x = self.x + k @ innovation
        self.p = self.p - k @ self.h @ self.p


class SpeedEstimator:
    """Estimate speed from object positions in a metric or BEV-pixel plane."""

    def __init__(self, fps: float, meters_per_pixel: float = 1.0, alpha: float = 0.3) -> None:
        if fps <= 0:
            raise ValueError("fps must be positive")
        if meters_per_pixel <= 0:
            raise ValueError("meters_per_pixel must be positive")
        if not 0 < alpha <= 1:
            raise ValueError("alpha must be in (0, 1]")
        self.fps = float(fps)
        self.dt = 1.0 / self.fps
        self.meters_per_pixel = float(meters_per_pixel)
        self.alpha = float(alpha)
        self.filters: dict[int | str, KalmanFilter] = {}
        self.ema_speeds_kmh: dict[int | str, float] = {}
        self.last_frame_idx: dict[int | str, int] = {}
        self.last_timestamp_s: dict[int | str, float] = {}

    def update(
        self,
        track_id: int | str,
        position_xy: tuple[float, float] | np.ndarray,
        timestamp_s: float | None = None,
        frame_idx: int | None = None,
    ) -> SpeedEstimate:
        x, y = np.array(position_xy, dtype=float).reshape(2)
        dt = self._resolve_dt(track_id, timestamp_s=timestamp_s, frame_idx=frame_idx)

        if track_id not in self.filters:
            kf = KalmanFilter(dt=dt)
            kf.x[0:2] = np.array([[x], [y]], dtype=float)
            self.filters[track_id] = kf
            return SpeedEstimate(track_id, 0.0, 0.0, False, {"initialized": True})

        kf = self.filters[track_id]
        if abs(kf.dt - dt) > 1e-9:
            kf.dt = dt
            kf.f[0, 2] = dt
            kf.f[1, 3] = dt
        kf.predict()
        kf.update((x, y))

        vx = float(kf.x[2, 0])
        vy = float(kf.x[3, 0])
        speed_px_s = float(np.hypot(vx, vy))
        speed_mps = speed_px_s * self.meters_per_pixel
        current_kmh = speed_mps * 3.6

        previous = self.ema_speeds_kmh.get(track_id)
        if previous is None:
            smoothed_kmh = current_kmh
            smoothed = False
        else:
            smoothed_kmh = self.alpha * current_kmh + (1.0 - self.alpha) * previous
            smoothed = True
        self.ema_speeds_kmh[track_id] = smoothed_kmh

        return SpeedEstimate(
            track_id=track_id,
            speed_kmh=float(smoothed_kmh),
            speed_mps=float(smoothed_kmh / 3.6),
            smoothed=smoothed,
            metadata={"raw_speed_kmh": float(current_kmh), "dt": dt},
        )

    def _resolve_dt(
        self,
        track_id: int | str,
        timestamp_s: float | None,
        frame_idx: int | None,
    ) -> float:
        dt = self.dt
        if timestamp_s is not None:
            last = self.last_timestamp_s.get(track_id)
            if last is not None and timestamp_s > last:
                dt = timestamp_s - last
            self.last_timestamp_s[track_id] = float(timestamp_s)
        elif frame_idx is not None:
            last_frame = self.last_frame_idx.get(track_id)
            if last_frame is not None and frame_idx > last_frame:
                dt = (frame_idx - last_frame) / self.fps
            self.last_frame_idx[track_id] = int(frame_idx)
        return max(dt, 1e-6)
