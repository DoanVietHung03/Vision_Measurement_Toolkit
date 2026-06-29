"""Validation helpers for calibrated distance measurements."""

from __future__ import annotations

from typing import Any

import numpy as np

from vision_measurement.calibration import PlaneCalibration


def load_pickle_distance_measurements(path: str) -> list[dict[str, Any]]:
    import pickle

    with open(path, "rb") as f:
        data = pickle.load(f, encoding="latin1")
    measurements = []
    for item in data.get("distanceMeasurement", []):
        p1 = np.array(item["p1"]).flatten()[:2].astype(float)
        p2 = np.array(item["p2"]).flatten()[:2].astype(float)
        measurements.append({"p1": p1.tolist(), "p2": p2.tolist(), "distance_m": float(item["distance"])})
    return measurements


def validate_distance_measurements(
    calibration: PlaneCalibration,
    measurements: list[dict[str, Any]],
) -> dict[str, Any]:
    rows = []
    errors = []
    for item in measurements:
        real = float(item["distance_m"])
        estimated = calibration.distance_m(item["p1"], item["p2"])
        error_pct = abs(estimated - real) / real * 100.0 if real else 0.0
        errors.append(error_pct)
        rows.append({"real_m": real, "estimated_m": estimated, "error_pct": error_pct})
    return {
        "rows": rows,
        "mean_error_pct": float(np.mean(errors)) if errors else 0.0,
        "count": len(rows),
    }
