"""Geometry helpers shared by distance and speed pipelines."""

from __future__ import annotations

import math
from collections.abc import Sequence


Point2D = tuple[float, float]


def euclidean_distance(p1: Sequence[float], p2: Sequence[float]) -> float:
    """Return Euclidean distance between two 2D points."""
    return math.hypot(float(p2[0]) - float(p1[0]), float(p2[1]) - float(p1[1]))


def quadrilateral_from_sides(
    l1: float,
    l2: float,
    l3: float,
    l4: float,
    diag_13: float,
) -> list[Point2D]:
    """Reconstruct a quadrilateral from four sides and diagonal P1-P3.

    Point order is P1, P2, P3, P4. This mirrors the original homography
    scripts, where users measured four floor edges and one diagonal.
    """
    values = [l1, l2, l3, l4, diag_13]
    if any(v <= 0 for v in values):
        raise ValueError("All side lengths and diagonal must be positive")
    if l1 + l2 < diag_13 or abs(l1 - l2) > diag_13:
        raise ValueError("Invalid geometry: L1, L2, and diag_13 cannot form a triangle")

    p1 = (0.0, 0.0)
    p2 = (float(l1), 0.0)

    cos_alpha = (l1**2 + diag_13**2 - l2**2) / (2 * l1 * diag_13)
    cos_alpha = max(-1.0, min(1.0, cos_alpha))
    alpha = math.acos(cos_alpha)
    p3 = (diag_13 * math.cos(alpha), diag_13 * math.sin(alpha))

    d = diag_13
    a = (l4**2 - l3**2 + d**2) / (2 * d)
    h_sq = l4**2 - a**2
    if h_sq < -1e-9:
        raise ValueError("Invalid geometry: P4 cannot be reconstructed")
    h = math.sqrt(max(0.0, h_sq))

    x0 = a * p3[0] / d
    y0 = a * p3[1] / d
    rx = -p3[1] / d
    ry = p3[0] / d

    p4_a = (x0 + h * rx, y0 + h * ry)
    p4_b = (x0 - h * rx, y0 - h * ry)

    def cross(px: float, py: float) -> float:
        return p3[0] * py - p3[1] * px

    p4 = p4_a if cross(*p4_a) > 0 else p4_b
    return [p1, p2, p3, p4]
