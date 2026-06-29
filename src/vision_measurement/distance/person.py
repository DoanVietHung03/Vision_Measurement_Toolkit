"""Reusable YOLO person and pose detection."""

from __future__ import annotations

import queue
import threading
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PersonObservation:
    box: list[float]
    ground_point: tuple[float, float]
    head_point: tuple[float, float]
    method: str
    confidence: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "box": self.box,
            "ground_point": self.ground_point,
            "head_point": self.head_point,
            "method": self.method,
            "confidence": self.confidence,
        }


class PersonPoseDetector:
    """Load one Ultralytics model and reuse it across frames."""

    def __init__(self, model_path: str, confidence: float = 0.35, device: Any = None) -> None:
        try:
            from ultralytics import YOLO  # type: ignore
        except ImportError as exc:
            raise ImportError("ultralytics is required for person detection") from exc
        self.model = YOLO(model_path)
        self.confidence = float(confidence)
        self.device = device

    def detect(self, frame) -> list[dict[str, Any]]:
        kwargs: dict[str, Any] = {"verbose": False, "conf": self.confidence}
        if self.device is not None:
            kwargs["device"] = self.device
        results = self.model(frame, **kwargs)
        people: list[dict[str, Any]] = []
        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy().astype(float)
            confidences = result.boxes.conf.cpu().numpy().astype(float)
            classes = result.boxes.cls.cpu().numpy().astype(int)
            keypoints = (
                result.keypoints.data.cpu().numpy()
                if result.keypoints is not None and result.keypoints.data is not None
                else None
            )
            for index, box in enumerate(boxes):
                if classes[index] != 0:
                    continue
                x1, y1, x2, y2 = box.tolist()
                ground = ((x1 + x2) / 2.0, y2)
                head = ((x1 + x2) / 2.0, y1)
                method = "bbox"
                if keypoints is not None and len(keypoints) > index:
                    points = keypoints[index]
                    ankles = [
                        (float(points[i][0]), float(points[i][1]))
                        for i in (15, 16)
                        if len(points) > i and points[i][2] > 0.5
                    ]
                    if ankles:
                        ground = (
                            sum(point[0] for point in ankles) / len(ankles),
                            sum(point[1] for point in ankles) / len(ankles),
                        )
                        method = "pose"
                    if len(points) > 0 and points[0][2] > 0.3:
                        head = (float(points[0][0]), float(points[0][1]))
                people.append(
                    PersonObservation(
                        box=[x1, y1, x2, y2],
                        ground_point=ground,
                        head_point=head,
                        method=method,
                        confidence=float(confidences[index]),
                    ).as_dict()
                )
        return people


class AsyncPersonDetector:
    """Latest-frame worker that never blocks video display on inference."""

    def __init__(self, detector: PersonPoseDetector) -> None:
        self.detector = detector
        self.input_queue: queue.Queue[tuple[int, Any]] = queue.Queue(maxsize=1)
        self.output_queue: queue.Queue[tuple[int, list[dict[str, Any]]]] = queue.Queue(maxsize=1)
        self.stop_event = threading.Event()
        self.error: BaseException | None = None
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def submit(self, frame_idx: int, frame) -> bool:
        if self.input_queue.full():
            return False
        self.input_queue.put_nowait((frame_idx, frame.copy()))
        return True

    def poll(self) -> tuple[int, list[dict[str, Any]]] | None:
        if self.error is not None:
            raise RuntimeError("Person detection worker failed") from self.error
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
                observations = self.detector.detect(frame)
            except BaseException as exc:
                self.error = exc
                self.stop_event.set()
                continue
            if self.output_queue.full():
                try:
                    self.output_queue.get_nowait()
                except queue.Empty:
                    pass
            self.output_queue.put((frame_idx, observations))

