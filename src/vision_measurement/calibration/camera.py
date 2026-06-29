"""Camera calibration helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def calibrate_chessboard(
    image_files: list[str | Path],
    pattern_size: tuple[int, int] = (9, 6),
    square_size_m: float = 0.034,
    fisheye: bool = False,
    output_json: str | Path | None = None,
    debug_dir: str | Path | None = None,
    sensor_size_mm: tuple[float, float] | None = None,
    write_undistorted: bool = True,
) -> dict[str, Any]:
    """Calibrate a camera and return diagnostics needed to assess quality."""
    try:
        import cv2  # type: ignore
    except ImportError as exc:
        raise ImportError("OpenCV is required for camera calibration") from exc

    files = [Path(path) for path in image_files]
    if not files:
        raise ValueError("No calibration images provided")

    first = cv2.imread(str(files[0]), cv2.IMREAD_GRAYSCALE)
    if first is None:
        raise FileNotFoundError(f"Cannot read calibration image: {files[0]}")
    height, width = first.shape[:2]

    pattern_points = np.zeros((np.prod(pattern_size), 3), np.float32)
    pattern_points[:, :2] = np.indices(pattern_size).T.reshape(-1, 2)
    pattern_points = np.expand_dims(pattern_points, -2)
    pattern_points *= square_size_m

    object_points = []
    image_points = []
    used_files: list[Path] = []
    rejected_files: dict[str, str] = {}
    debug_path = Path(debug_dir) if debug_dir else None
    if debug_path:
        debug_path.mkdir(parents=True, exist_ok=True)

    for file_path in files:
        image = cv2.imread(str(file_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            rejected_files[str(file_path)] = "cannot_read"
            continue
        if image.shape[:2] != (height, width):
            rejected_files[str(file_path)] = "resolution_mismatch"
            continue
        found, corners = cv2.findChessboardCorners(image, pattern_size)
        if found:
            criteria = (
                cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT,
                30,
                0.1,
            )
            cv2.cornerSubPix(image, corners, (5, 5), (-1, -1), criteria)
            object_points.append(pattern_points)
            image_points.append(corners)
            used_files.append(file_path)
        else:
            rejected_files[str(file_path)] = "chessboard_not_found"

        if debug_path:
            visualization = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            if found:
                cv2.drawChessboardCorners(visualization, pattern_size, corners, True)
            cv2.imwrite(str(debug_path / f"{file_path.stem}_chessboard.png"), visualization)

    if not image_points:
        raise RuntimeError("No chessboards found in calibration images")

    calibration_function = cv2.fisheye.calibrate if fisheye else cv2.calibrateCamera
    rms, camera_matrix, distortion, rotation_vectors, translation_vectors = calibration_function(
        object_points,
        image_points,
        (width, height),
        None,
        None,
    )

    project_function = cv2.fisheye.projectPoints if fisheye else cv2.projectPoints
    per_image_error: dict[str, float] = {}
    errors = []
    orientations: dict[str, dict[str, Any]] = {}
    for index, file_path in enumerate(used_files):
        projected, _ = project_function(
            object_points[index],
            rotation_vectors[index],
            translation_vectors[index],
            camera_matrix,
            distortion,
        )
        error = cv2.norm(image_points[index], projected, cv2.NORM_L2) / len(projected)
        errors.append(float(error))
        per_image_error[str(file_path)] = float(error)
        rotation_matrix, _ = cv2.Rodrigues(rotation_vectors[index])
        orientations[str(file_path)] = {
            "rotation_matrix": rotation_matrix.tolist(),
            "translation": np.asarray(translation_vectors[index]).reshape(-1).astype(float).tolist(),
        }

    result: dict[str, Any] = {
        "image_resolution": [width, height],
        "chessboard_inner_corners": list(pattern_size),
        "chessboard_spacing_m": float(square_size_m),
        "chessboard_points": pattern_points.tolist(),
        "camera_matrix": camera_matrix.tolist(),
        "distortion_coefficients": distortion.ravel().tolist(),
        "fisheye": bool(fisheye),
        "rms": float(rms),
        "image_count": len(image_points),
        "input_image_count": len(files),
        "used_images": [str(path) for path in used_files],
        "rejected_images": rejected_files,
        "reprojection_error": {
            "average": float(np.mean(errors)),
            "stddev": float(np.std(errors)),
            "image": per_image_error,
        },
        "chessboard_orientations": orientations,
    }

    if sensor_size_mm is not None:
        fov_x, fov_y, focal_length, principal_point, aspect_ratio = cv2.calibrationMatrixValues(
            camera_matrix,
            (width, height),
            sensor_size_mm[0],
            sensor_size_mm[1],
        )
        result.update(
            {
                "sensor_size_mm": list(sensor_size_mm),
                "fov_degrees": [float(fov_x), float(fov_y)],
                "focal_length_mm": float(focal_length),
                "principal_point_mm": [float(principal_point[0]), float(principal_point[1])],
                "aspect_ratio": float(aspect_ratio),
            }
        )

    if debug_path and write_undistorted:
        fisheye_maps = None
        if fisheye:
            new_matrix = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
                camera_matrix,
                distortion,
                (width, height),
                np.eye(3),
                balance=1.0,
            )
            fisheye_maps = cv2.fisheye.initUndistortRectifyMap(
                camera_matrix,
                distortion,
                np.eye(3),
                new_matrix,
                (width, height),
                cv2.CV_16SC2,
            )
            roi = (0, 0, width, height)
        else:
            new_matrix, roi = cv2.getOptimalNewCameraMatrix(
                camera_matrix,
                distortion,
                (width, height),
                1,
                (width, height),
            )
        for file_path in used_files:
            image = cv2.imread(str(file_path))
            if image is None:
                continue
            if fisheye_maps is not None:
                undistorted = cv2.remap(
                    image,
                    fisheye_maps[0],
                    fisheye_maps[1],
                    cv2.INTER_LINEAR,
                )
            else:
                undistorted = cv2.undistort(image, camera_matrix, distortion, None, new_matrix)
            cv2.imwrite(str(debug_path / f"{file_path.stem}_undistorted.png"), undistorted)
            x, y, crop_width, crop_height = roi
            cropped = undistorted[y : y + crop_height, x : x + crop_width]
            cv2.imwrite(str(debug_path / f"{file_path.stem}_undistorted_cropped.png"), cropped)

    if output_json:
        output = Path(output_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result

