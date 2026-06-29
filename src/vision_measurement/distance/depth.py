"""Depth-based 3D measurement primitives."""

from __future__ import annotations

import queue
import threading

import numpy as np


class MetricDepthProjector:
    """Convert relative depth values into 3D points using a scale factor."""

    def __init__(self, scale_factor: float, fx: float, fy: float, cx: float, cy: float) -> None:
        if scale_factor <= 0:
            raise ValueError("scale_factor must be positive")
        if fx <= 0 or fy <= 0:
            raise ValueError("fx and fy must be positive")
        self.scale_factor = float(scale_factor)
        self.fx = float(fx)
        self.fy = float(fy)
        self.cx = float(cx)
        self.cy = float(cy)

    def pixel_to_3d(self, u: float, v: float, raw_depth_value: float) -> np.ndarray | None:
        if raw_depth_value <= 0:
            return None
        z = self.scale_factor / float(raw_depth_value)
        x = (float(u) - self.cx) * z / self.fx
        y = (float(v) - self.cy) * z / self.fy
        return np.array([x, y, z], dtype=float)

    @staticmethod
    def distance(point_a: np.ndarray, point_b: np.ndarray) -> float:
        return float(np.linalg.norm(point_a - point_b))


class DepthAnythingEstimator:
    """Thin wrapper around Hugging Face depth-estimation pipelines."""

    def __init__(self, model_repo: str, device: int | None = None) -> None:
        try:
            import torch
            from transformers import pipeline
        except ImportError as exc:
            raise ImportError("Depth inference requires torch and transformers.") from exc

        if device is None:
            device = 0 if torch.cuda.is_available() else -1
        self.model_repo = model_repo
        self.device = device
        self.pipe = pipeline(task="depth-estimation", model=model_repo, device=device)

    def predict(self, frame_bgr: np.ndarray, process_width: int | None = None) -> np.ndarray:
        try:
            import cv2
            from PIL import Image
        except ImportError as exc:
            raise ImportError("Depth inference requires opencv-python and Pillow.") from exc

        height, width = frame_bgr.shape[:2]
        frame_for_model = frame_bgr
        if process_width:
            new_height = max(1, int(height * (process_width / float(width))))
            frame_for_model = cv2.resize(frame_bgr, (process_width, new_height))

        image_pil = Image.fromarray(cv2.cvtColor(frame_for_model, cv2.COLOR_BGR2RGB))
        depth_output = self.pipe(image_pil)
        depth_tensor = depth_output["predicted_depth"]
        depth_map = depth_tensor.squeeze().cpu().numpy()
        return cv2.resize(depth_map, (width, height), interpolation=cv2.INTER_LINEAR)


class AsyncDepthEstimator:
    """Latest-frame worker for non-blocking depth inference."""

    def __init__(self, estimator: DepthAnythingEstimator, process_width: int | None = None) -> None:
        self.estimator = estimator
        self.process_width = process_width
        self.input_queue: queue.Queue[tuple[int, np.ndarray]] = queue.Queue(maxsize=1)
        self.output_queue: queue.Queue[tuple[int, np.ndarray]] = queue.Queue(maxsize=1)
        self.stop_event = threading.Event()
        self.error: BaseException | None = None
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def submit(self, frame_idx: int, frame: np.ndarray) -> bool:
        if self.input_queue.full():
            return False
        self.input_queue.put_nowait((frame_idx, frame.copy()))
        return True

    def poll(self) -> tuple[int, np.ndarray] | None:
        if self.error is not None:
            raise RuntimeError("Depth inference worker failed") from self.error
        latest = None
        while not self.output_queue.empty():
            latest = self.output_queue.get_nowait()
        return latest

    def close(self) -> None:
        self.stop_event.set()
        self.thread.join(timeout=1.0)

    def _run(self) -> None:
        while not self.stop_event.is_set():
            try:
                frame_idx, frame = self.input_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                depth_map = self.estimator.predict(frame, process_width=self.process_width)
            except BaseException as exc:
                self.error = exc
                self.stop_event.set()
                continue
            if self.output_queue.full():
                try:
                    self.output_queue.get_nowait()
                except queue.Empty:
                    pass
            self.output_queue.put((frame_idx, depth_map))


def sample_depth(depth_map: np.ndarray, point_xy: tuple[float, float], patch_radius: int = 2) -> float:
    """Sample median depth around a pixel for noise resistance."""
    height, width = depth_map.shape[:2]
    x = max(0, min(int(round(point_xy[0])), width - 1))
    y = max(0, min(int(round(point_xy[1])), height - 1))
    y1 = max(0, y - patch_radius)
    y2 = min(height, y + patch_radius + 1)
    x1 = max(0, x - patch_radius)
    x2 = min(width, x + patch_radius + 1)
    patch = depth_map[y1:y2, x1:x2]
    if patch.size == 0:
        return float(depth_map[y, x])
    return float(np.median(patch))

