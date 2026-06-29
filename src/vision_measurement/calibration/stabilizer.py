"""Optical-flow stabilization for image-plane calibration points."""

from __future__ import annotations

from typing import Iterable

import numpy as np


class PerspectiveStabilizer:
    """Estimate motion from an anchor frame to the current frame."""

    def __init__(self, min_points: int = 10) -> None:
        self.min_points = int(min_points)
        self.anchor_gray = None
        self.anchor_points = None
        self.current_matrix = np.eye(3, dtype=np.float32)
        self.lk_params = {
            "winSize": (21, 21),
            "maxLevel": 3,
            "criteria": (3, 30, 0.03),
        }

    def initialize(
        self,
        frame,
        excluded_polygon: np.ndarray | None = None,
        excluded_boxes: Iterable[Iterable[float]] = (),
    ) -> None:
        import cv2  # type: ignore

        self.anchor_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mask = np.full_like(self.anchor_gray, 255)
        if excluded_polygon is not None:
            polygon = np.asarray(excluded_polygon, dtype=np.int32).reshape(-1, 1, 2)
            cv2.fillPoly(mask, [polygon], 0)
        for box in excluded_boxes:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(mask, (x1, y1), (x2, y2), 0, -1)
        self.anchor_points = cv2.goodFeaturesToTrack(
            self.anchor_gray,
            mask=mask,
            maxCorners=300,
            qualityLevel=0.01,
            minDistance=10,
        )
        self.current_matrix = np.eye(3, dtype=np.float32)

    def update(self, frame) -> np.ndarray:
        import cv2  # type: ignore

        if self.anchor_gray is None or self.anchor_points is None:
            return self.current_matrix
        current_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        new_points, status, _ = cv2.calcOpticalFlowPyrLK(
            self.anchor_gray,
            current_gray,
            self.anchor_points,
            None,
            **self.lk_params,
        )
        if new_points is None or status is None:
            return self.current_matrix
        good_new = new_points[status.reshape(-1) == 1]
        good_old = self.anchor_points[status.reshape(-1) == 1]
        if len(good_new) < max(4, self.min_points):
            return self.current_matrix
        matrix, _ = cv2.findHomography(good_old, good_new, cv2.RANSAC, 5.0)
        if matrix is not None:
            self.current_matrix = matrix.astype(np.float32)
        return self.current_matrix

    @staticmethod
    def transform_points(points, matrix: np.ndarray) -> np.ndarray:
        import cv2  # type: ignore

        values = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(values, matrix).reshape(-1, 2)
