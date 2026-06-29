from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

from _bootstrap import bootstrap_src_path


def _point(value: str) -> tuple[float, float]:
    try:
        x, y = value.split(",", 1)
        return float(x), float(y)
    except Exception as exc:
        raise argparse.ArgumentTypeError("Point must look like x,y") from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Measure plane distance/height using homography config.")
    parser.add_argument("--config", default="configs/distance_homography.json", help="Homography config JSON.")
    parser.add_argument("--mode", choices=["points", "image", "video"], default="points", help="Run mode.")
    parser.add_argument("--image", help="Image path override for image mode.")
    parser.add_argument("--video", help="Video path override for video mode.")
    parser.add_argument("--point-a", type=_point, help="Pixel point A as x,y.")
    parser.add_argument("--point-b", type=_point, help="Pixel point B as x,y.")
    parser.add_argument("--target-point", type=_point, help="Target point for person-to-target distance.")
    parser.add_argument("--detect", action="store_true", help="Run YOLO person/pose detection.")
    parser.add_argument("--height", action="store_true", help="Estimate detected person heights.")
    parser.add_argument("--undistort", action="store_true", help="Undistort frames using camera calibration JSON.")
    parser.add_argument("--interactive", action="store_true", help="Enable click measurement/target selection.")
    parser.add_argument(
        "--stabilize",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Compensate camera motion in video mode.",
    )
    parser.add_argument("--sync-detect", action="store_true", help="Run detection synchronously instead of a worker.")
    parser.add_argument("--csv-output", help="Optional CSV output path.")
    parser.add_argument("--max-frames", type=int, default=None, help="Optional frame limit for video mode.")
    parser.add_argument("--no-display", action="store_true", help="Print JSON results without opening a GUI window.")
    return parser


def main() -> None:
    root = bootstrap_src_path()
    args = build_parser().parse_args()

    from vision_measurement.calibration import PlaneCalibration
    from vision_measurement.core.config import load_json_config
    from vision_measurement.core.paths import require_existing_path
    from vision_measurement.distance import measure_plane_distance

    config_path = require_existing_path(args.config, "Config", base_dir=root)
    config = load_json_config(config_path)
    measurement = config.get("measurement", {})
    point_a = args.point_a or tuple(measurement.get("point_a", (0.0, 0.0)))
    point_b = args.point_b or tuple(measurement.get("point_b", (0.0, 0.0)))
    target_point = args.target_point or tuple(measurement.get("target_point", ())) or None

    calibration = PlaneCalibration.from_config(config)
    base_result = measure_plane_distance(calibration, point_a, point_b, metadata={"config": str(config_path)})

    if args.mode == "points":
        row = distance_result_to_dict(base_result)
        print(json.dumps(row, indent=2))
        write_csv(args.csv_output, [row], root)
        return

    detector = build_person_detector(args, config, root)
    if args.mode == "image":
        frame = load_image_frame(args, config, root)
        people = detector.detect(frame) if detector else []
        rows = process_frame(
            frame,
            0,
            args,
            config,
            calibration,
            point_a,
            point_b,
            target_point,
            people,
            root,
            {},
        )
        if args.interactive and not args.no_display:
            rows.extend(run_interactive_image(frame, calibration, rows))
        elif not args.no_display:
            show_image(frame, rows, calibration.source_points, target_point)
        print(json.dumps(rows, indent=2))
        write_csv(args.csv_output, rows, root)
        return

    run_video_mode(
        args,
        config,
        calibration,
        point_a,
        point_b,
        target_point,
        detector,
        root,
    )


def build_person_detector(args, config: dict, root):
    if not (args.detect or args.height):
        return None
    from vision_measurement.core.paths import require_existing_path
    from vision_measurement.distance.person import PersonPoseDetector

    model_cfg = config.get("model", {})
    model_path = require_existing_path(model_cfg.get("yolo_pose_path"), "YOLO pose model", base_dir=root)
    confidence = float(model_cfg.get("confidence", 0.35))
    return PersonPoseDetector(str(model_path), confidence=confidence)


def distance_result_to_dict(result) -> dict[str, Any]:
    return {
        "type": "manual_distance",
        "method": result.method,
        "distance_m": result.distance_m,
        "point_a_px": [float(value) for value in result.points_px[0]],
        "point_b_px": [float(value) for value in result.points_px[1]],
    }


def load_image_frame(args, config: dict, root):
    import cv2  # type: ignore
    from vision_measurement.core.paths import require_existing_path

    image_path = args.image or config.get("input", {}).get("image_path")
    path = require_existing_path(image_path, "Image", base_dir=root)
    frame = cv2.imread(str(path))
    if frame is None:
        raise OSError(f"Cannot read image: {path}")
    return maybe_undistort(frame, args, config, root)


def run_video_mode(
    args,
    config: dict,
    base_calibration,
    point_a,
    point_b,
    target_point,
    detector,
    root,
) -> None:
    import cv2  # type: ignore
    import numpy as np

    from vision_measurement.calibration import PlaneCalibration
    from vision_measurement.calibration.stabilizer import PerspectiveStabilizer
    from vision_measurement.core.paths import require_existing_path
    from vision_measurement.core.profiler import LatencyProfiler
    from vision_measurement.distance.person import AsyncPersonDetector

    video_path = args.video or config.get("input", {}).get("video_path")
    path = require_existing_path(video_path, "Video", base_dir=root)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise OSError(f"Cannot open video: {path}")

    video_cfg = config.get("video", {})
    stabilization_enabled = (
        bool(video_cfg.get("stabilize", True))
        if args.stabilize is None
        else bool(args.stabilize)
    )
    detection_interval = max(1, int(video_cfg.get("detection_interval", 2)))
    async_worker = AsyncPersonDetector(detector) if detector and not args.sync_detect else None
    stabilizer = PerspectiveStabilizer()
    profiler = LatencyProfiler()
    last_people: list[dict[str, Any]] = []
    all_rows: list[dict[str, Any]] = []
    frame_idx = 0
    height_runtime: dict[str, Any] = {}
    source_initial = base_calibration.source_points.copy()
    target_initial = np.asarray([target_point], dtype=np.float32) if target_point else None
    interaction = {"frame": None, "target": target_point, "tracker": None, "manual_target": False}

    if not args.no_display:
        cv2.namedWindow("Homography Distance", cv2.WINDOW_NORMAL)
        if args.interactive:
            cv2.setMouseCallback(
                "Homography Distance",
                lambda event, x, y, flags, param: video_mouse_callback(
                    event,
                    x,
                    y,
                    interaction,
                ),
            )

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if args.max_frames is not None and frame_idx >= args.max_frames:
                break
            frame = maybe_undistort(frame, args, config, root)
            interaction["frame"] = frame

            if frame_idx == 0 and stabilization_enabled:
                stabilizer.initialize(frame, excluded_polygon=source_initial)

            motion = np.eye(3, dtype=np.float32)
            if stabilization_enabled and frame_idx > 0:
                started = time.perf_counter()
                motion = stabilizer.update(frame)
                profiler.update("stabilizer", (time.perf_counter() - started) * 1000.0)
            current_source = (
                stabilizer.transform_points(source_initial, motion)
                if stabilization_enabled
                else source_initial
            )
            calibration = PlaneCalibration(
                source_points=current_source.astype(np.float32),
                target_points=base_calibration.target_points,
                meters_per_pixel=base_calibration.meters_per_pixel,
            )

            current_target = interaction.get("target")
            tracker = interaction.get("tracker")
            if tracker is not None:
                success, box = tracker.update(frame)
                if success:
                    x, y, w, h = box
                    current_target = (x + w / 2.0, y + h / 2.0)
                    interaction["target"] = current_target
                else:
                    interaction["tracker"] = None
            elif (
                target_initial is not None
                and stabilization_enabled
                and not interaction["manual_target"]
            ):
                current_target = tuple(float(value) for value in stabilizer.transform_points(target_initial, motion)[0])
                interaction["target"] = current_target

            if detector and frame_idx % detection_interval == 0:
                started = time.perf_counter()
                if async_worker:
                    async_worker.submit(frame_idx, frame)
                else:
                    last_people = detector.detect(frame)
                    profiler.update("person_detection", (time.perf_counter() - started) * 1000.0)
            if async_worker:
                output = async_worker.poll()
                if output is not None:
                    _, last_people = output

            current_point_a = (
                tuple(float(value) for value in stabilizer.transform_points([point_a], motion)[0])
                if stabilization_enabled
                else point_a
            )
            current_point_b = (
                tuple(float(value) for value in stabilizer.transform_points([point_b], motion)[0])
                if stabilization_enabled
                else point_b
            )
            rows = process_frame(
                frame,
                frame_idx,
                args,
                config,
                calibration,
                current_point_a,
                current_point_b,
                current_target,
                last_people,
                root,
                height_runtime,
            )
            all_rows.extend(rows)

            if args.no_display:
                for row in rows:
                    print(json.dumps(row))
            else:
                annotated = annotate_frame(
                    frame.copy(),
                    rows,
                    source_points=current_source,
                    target_point=current_target,
                    people=last_people,
                )
                cv2.imshow("Homography Distance", annotated)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord(" "):
                    cv2.waitKey(0)
            frame_idx += 1
    finally:
        cap.release()
        if async_worker:
            async_worker.close()
        if not args.no_display:
            cv2.destroyAllWindows()

    write_csv(args.csv_output, all_rows, root)
    summary = profiler.summary()
    if summary:
        print(json.dumps({"latency": summary}, indent=2))


def video_mouse_callback(event, x: int, y: int, interaction: dict[str, Any]) -> None:
    import cv2  # type: ignore

    if event == cv2.EVENT_RBUTTONDOWN:
        interaction["target"] = None
        interaction["tracker"] = None
        interaction["manual_target"] = True
        return
    if event != cv2.EVENT_LBUTTONDOWN:
        return
    interaction["target"] = (float(x), float(y))
    interaction["manual_target"] = True
    frame = interaction.get("frame")
    if frame is None:
        return
    tracker = create_kcf_tracker(cv2)
    if tracker is None:
        return
    radius = 30
    bbox = (
        max(0, x - radius),
        max(0, y - radius),
        min(radius * 2, frame.shape[1] - max(0, x - radius)),
        min(radius * 2, frame.shape[0] - max(0, y - radius)),
    )
    tracker.init(frame, bbox)
    interaction["tracker"] = tracker


def create_kcf_tracker(cv2):
    if hasattr(cv2, "TrackerKCF_create"):
        return cv2.TrackerKCF_create()
    legacy = getattr(cv2, "legacy", None)
    if legacy is not None and hasattr(legacy, "TrackerKCF_create"):
        return legacy.TrackerKCF_create()
    return None


def maybe_undistort(frame, args, config: dict, root):
    if not args.undistort:
        return frame
    import cv2  # type: ignore
    import numpy as np

    from vision_measurement.core.config import load_json_config
    from vision_measurement.core.paths import resolve_path

    calibration_json = config.get("input", {}).get("camera_calibration_json")
    path = resolve_path(calibration_json, base_dir=root)
    if path is None or not path.exists():
        raise FileNotFoundError(f"Camera calibration not found: {path}")
    data = load_json_config(path)
    matrix = np.array(data["camera_matrix"], dtype=float)
    distortion = np.array(data["distortion_coefficients"], dtype=float)
    h, w = frame.shape[:2]
    if "image_resolution" in data:
        calibration_width, calibration_height = data["image_resolution"]
        sx = w / float(calibration_width)
        sy = h / float(calibration_height)
        matrix[0, 0] *= sx
        matrix[1, 1] *= sy
        matrix[0, 2] *= sx
        matrix[1, 2] *= sy
    new_matrix, _ = cv2.getOptimalNewCameraMatrix(matrix, distortion, (w, h), 1, (w, h))
    return cv2.undistort(frame, matrix, distortion, None, new_matrix)


def process_frame(
    frame,
    frame_idx: int,
    args,
    config: dict,
    calibration,
    point_a,
    point_b,
    target_point,
    people: list[dict[str, Any]],
    root,
    runtime: dict[str, Any],
) -> list[dict[str, Any]]:
    from vision_measurement.distance import HeightEstimator, measure_plane_distance

    rows = []
    manual = measure_plane_distance(calibration, point_a, point_b, metadata={"frame_idx": frame_idx})
    rows.append(distance_result_to_dict(manual) | {"frame_idx": frame_idx})

    if target_point is not None:
        for index, person in enumerate(people):
            rows.append(
                {
                    "type": "person_to_target_distance",
                    "frame_idx": frame_idx,
                    "person_index": index,
                    "distance_m": calibration.distance_m(person["ground_point"], target_point),
                    "ground_point_px": [float(value) for value in person["ground_point"]],
                    "target_point_px": [float(value) for value in target_point],
                    "box_xyxy": person["box"],
                    "method": person["method"],
                }
            )

    if args.height and people:
        from vision_measurement.core.paths import require_existing_path

        height_cfg = config.get("height", {})
        calibration_json = require_existing_path(
            config.get("input", {}).get("camera_calibration_json"),
            "Camera calibration",
            base_dir=root,
        )
        estimator = runtime.get("height_estimator")
        if estimator is None:
            estimator = HeightEstimator(allow_fallback=bool(height_cfg.get("allow_fallback", False)))
            estimator.load_focal_length(calibration_json, frame.shape[1])
            runtime["height_estimator"] = estimator
        metric_matrix = metric_homography(calibration)
        camera_position = tuple(height_cfg.get("camera_position_m", (0.0, 0.0)))
        for index, person in enumerate(people):
            height_m, distance_m = estimator.calculate(
                person["head_point"],
                person["ground_point"],
                metric_matrix,
                camera_position,
            )
            rows.append(
                {
                    "type": "person_height",
                    "frame_idx": frame_idx,
                    "person_index": index,
                    "height_m": height_m,
                    "distance_from_camera_m": distance_m,
                    "head_point_px": [float(value) for value in person["head_point"]],
                    "ground_point_px": [float(value) for value in person["ground_point"]],
                    "calibration_status": estimator.status,
                    "box_xyxy": person["box"],
                }
            )
    return rows


def metric_homography(calibration):
    import cv2  # type: ignore
    import numpy as np

    return cv2.getPerspectiveTransform(
        calibration.source_points.astype(np.float32),
        (calibration.target_points * calibration.meters_per_pixel).astype(np.float32),
    )


def run_interactive_image(frame, calibration, base_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    import cv2  # type: ignore

    clicks: list[tuple[float, float]] = []
    added_rows: list[dict[str, Any]] = []
    cursor = {"x": -1, "y": -1}
    window = "Homography Distance"

    def on_mouse(event, x, y, flags, param):  # noqa: ANN001
        cursor["x"], cursor["y"] = x, y
        if event == cv2.EVENT_RBUTTONDOWN:
            clicks.clear()
        elif event == cv2.EVENT_LBUTTONDOWN:
            clicks.append((float(x), float(y)))
            if len(clicks) >= 2:
                point_a, point_b = clicks[-2:]
                added_rows.append(
                    {
                        "type": "interactive_distance",
                        "method": "homography",
                        "distance_m": calibration.distance_m(point_a, point_b),
                        "point_a_px": list(point_a),
                        "point_b_px": list(point_b),
                        "frame_idx": 0,
                    }
                )

    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window, on_mouse)
    while True:
        display = annotate_frame(
            frame.copy(),
            base_rows + added_rows,
            source_points=calibration.source_points,
        )
        add_magnifier(display, cursor["x"], cursor["y"])
        cv2.imshow(window, display)
        key = cv2.waitKey(20) & 0xFF
        if key in (ord("q"), 27):
            break
    cv2.destroyAllWindows()
    return added_rows


def add_magnifier(frame, x: int, y: int, radius: int = 35, zoom: int = 4) -> None:
    import cv2  # type: ignore

    if x < 0 or y < 0:
        return
    height, width = frame.shape[:2]
    x1, x2 = max(0, x - radius), min(width, x + radius)
    y1, y2 = max(0, y - radius), min(height, y + radius)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return
    magnified = cv2.resize(crop, None, fx=zoom, fy=zoom, interpolation=cv2.INTER_NEAREST)
    magnified_height, magnified_width = magnified.shape[:2]
    max_width = min(magnified_width, width)
    max_height = min(magnified_height, height)
    frame[0:max_height, width - max_width : width] = magnified[0:max_height, 0:max_width]
    cv2.rectangle(frame, (width - max_width, 0), (width - 1, max_height - 1), (255, 255, 255), 1)


def show_image(frame, rows, source_points, target_point) -> None:
    import cv2  # type: ignore

    annotated = annotate_frame(
        frame.copy(),
        rows,
        source_points=source_points,
        target_point=target_point,
    )
    cv2.namedWindow("Homography Distance", cv2.WINDOW_NORMAL)
    cv2.imshow("Homography Distance", annotated)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def annotate_frame(
    frame,
    rows: list[dict[str, Any]],
    source_points=None,
    target_point=None,
    people: list[dict[str, Any]] | None = None,
):
    import cv2  # type: ignore
    import numpy as np

    if source_points is not None:
        polygon = np.asarray(source_points, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(frame, [polygon], True, (255, 120, 0), 2)
    if target_point is not None:
        cv2.circle(frame, tuple(map(int, target_point)), 6, (0, 255, 255), -1)
    for person in people or []:
        x1, y1, x2, y2 = map(int, person["box"])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 2)

    for row in rows:
        row_type = row["type"]
        if row_type in ("manual_distance", "interactive_distance"):
            point_a = tuple(map(int, row["point_a_px"]))
            point_b = tuple(map(int, row["point_b_px"]))
            cv2.circle(frame, point_a, 5, (0, 0, 255), -1)
            cv2.circle(frame, point_b, 5, (0, 255, 255), -1)
            cv2.line(frame, point_a, point_b, (0, 165, 255), 2)
            midpoint = ((point_a[0] + point_b[0]) // 2, (point_a[1] + point_b[1]) // 2)
            cv2.putText(
                frame,
                f"{row['distance_m']:.2f}m",
                midpoint,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 165, 255),
                2,
            )
        elif row_type == "person_to_target_distance":
            ground = tuple(map(int, row["ground_point_px"]))
            target = tuple(map(int, row["target_point_px"]))
            cv2.line(frame, ground, target, (255, 255, 0), 2)
            midpoint = ((ground[0] + target[0]) // 2, (ground[1] + target[1]) // 2)
            cv2.putText(
                frame,
                f"{row['distance_m']:.2f}m",
                midpoint,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),
                2,
            )
        elif row_type == "person_height":
            head = tuple(map(int, row["head_point_px"]))
            ground = tuple(map(int, row["ground_point_px"]))
            cv2.line(frame, head, ground, (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"H {row['height_m']:.2f}m",
                head,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
    return frame


def write_csv(path_value: str | None, rows: list[dict[str, Any]], root) -> None:
    if not path_value or not rows:
        return
    from vision_measurement.core.paths import resolve_path

    path = resolve_path(path_value, base_dir=root)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value) if isinstance(value, (list, dict, tuple)) else value
                    for key, value in row.items()
                }
            )


if __name__ == "__main__":
    main()






