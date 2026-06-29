from __future__ import annotations

import argparse
import csv
import json
import time
from typing import Any

from _bootstrap import bootstrap_src_path


def _point(value: str) -> tuple[float, float]:
    try:
        x, y = value.split(",", 1)
        return float(x), float(y)
    except Exception as exc:
        raise argparse.ArgumentTypeError("Point must look like x,y") from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Measure 3D distance with Depth-Anything.")
    parser.add_argument("--config", default="configs/distance_depth.json", help="Depth distance config JSON.")
    parser.add_argument("--image", help="Image path override.")
    parser.add_argument("--video", help="Video path override.")
    parser.add_argument("--frame-index", type=int, default=0, help="Frame index for one-shot video inference.")
    parser.add_argument("--point-a", type=_point, help="Pixel point A as x,y.")
    parser.add_argument("--point-b", type=_point, help="Pixel point B as x,y.")
    parser.add_argument("--raw-depth-a", type=float, help="Raw depth value at point A.")
    parser.add_argument("--raw-depth-b", type=float, help="Raw depth value at point B.")
    parser.add_argument("--infer-depth", action="store_true", help="Run one-shot Depth-Anything inference.")
    parser.add_argument("--calibrate-person", action="store_true", help="Calibrate scale from a detected person.")
    parser.add_argument("--real-distance-m", type=float, help="Known camera-to-person distance.")
    parser.add_argument("--realtime-video", action="store_true", help="Run continuous async Depth + YOLO video measurement.")
    parser.add_argument("--max-frames", type=int, help="Optional realtime frame limit.")
    parser.add_argument("--display-width", type=int, help="Realtime processing/display width override.")
    parser.add_argument("--depth-skip", type=int, help="Submit depth inference every N frames.")
    parser.add_argument("--detect-skip", type=int, help="Submit person detection every N frames.")
    parser.add_argument("--csv-output", help="Optional realtime CSV output path.")
    parser.add_argument("--no-display", action="store_true", help="Do not open an OpenCV window.")
    return parser


def main() -> None:
    root = bootstrap_src_path()
    args = build_parser().parse_args()

    from vision_measurement.core.config import load_json_config
    from vision_measurement.core.paths import require_existing_path
    from vision_measurement.distance import DepthAnythingEstimator, MetricDepthProjector
    from vision_measurement.distance.depth import sample_depth

    config_path = require_existing_path(args.config, "Config", base_dir=root)
    config = load_json_config(config_path)
    if args.realtime_video:
        run_depth_video(args, config, root)
        return

    input_cfg = config.get("input", {})
    depth_cfg = config["depth"]
    camera_cfg = config["camera"]
    measurement_cfg = config.get("measurement", {})
    model_cfg = config.get("model", {})

    point_a = args.point_a or tuple(measurement_cfg.get("point_a", (0.0, 0.0)))
    point_b = args.point_b or tuple(measurement_cfg.get("point_b", (0.0, 0.0)))
    raw_a = args.raw_depth_a if args.raw_depth_a is not None else measurement_cfg.get("raw_depth_a")
    raw_b = args.raw_depth_b if args.raw_depth_b is not None else measurement_cfg.get("raw_depth_b")

    need_inference = args.infer_depth or raw_a is None or raw_b is None or args.calibrate_person
    frame = None
    depth_map = None
    if need_inference:
        frame = load_input_frame(args, input_cfg, root)
        estimator = DepthAnythingEstimator(depth_cfg.get("model_repo", "depth-anything/Depth-Anything-V2-Small-hf"))
        depth_map = estimator.predict(frame, process_width=depth_cfg.get("process_width"))
        raw_a = sample_depth(depth_map, point_a)
        raw_b = sample_depth(depth_map, point_b)

    scale_factor = float(depth_cfg["scale_factor"])
    calibration_info = None
    if args.calibrate_person:
        if frame is None or depth_map is None:
            raise RuntimeError("Depth inference did not produce a frame and depth map")
        real_distance_m = args.real_distance_m or depth_cfg.get("reference_distance_m")
        if real_distance_m is None:
            raise SystemExit("--calibrate-person requires a known real distance")
        yolo_path = require_existing_path(model_cfg.get("yolo_path"), "YOLO model", base_dir=root)
        foot_point, box = detect_reference_person_foot(frame, str(yolo_path))
        reference_raw = sample_depth(depth_map, foot_point)
        scale_factor = float(real_distance_m) * reference_raw
        calibration_info = {
            "reference_foot_px": foot_point,
            "reference_box_xyxy": box,
            "reference_raw_depth": reference_raw,
            "real_distance_m": float(real_distance_m),
            "scale_factor": scale_factor,
        }

    projector = projector_from_config(scale_factor, camera_cfg)
    point_3d_a = projector.pixel_to_3d(point_a[0], point_a[1], float(raw_a))
    point_3d_b = projector.pixel_to_3d(point_b[0], point_b[1], float(raw_b))
    if point_3d_a is None or point_3d_b is None:
        raise SystemExit("Raw depth values must be positive")

    result = {
        "method": "depth",
        "distance_m": projector.distance(point_3d_a, point_3d_b),
        "point_a_px": point_a,
        "point_b_px": point_b,
        "raw_depth_a": float(raw_a),
        "raw_depth_b": float(raw_b),
        "point_a_3d": point_3d_a.tolist(),
        "point_b_3d": point_3d_b.tolist(),
        "scale_factor": scale_factor,
        "calibration": calibration_info,
    }
    print(json.dumps(result, indent=2))
    if frame is not None and not args.no_display:
        show_depth_preview(frame, point_a, point_b, result["distance_m"])


def projector_from_config(scale_factor: float, camera_cfg: dict, frame_width: int | None = None):
    from vision_measurement.distance import MetricDepthProjector

    reference_width = float(camera_cfg.get("reference_width", frame_width or 1.0))
    scale = float(frame_width) / reference_width if frame_width and reference_width else 1.0
    return MetricDepthProjector(
        scale_factor=scale_factor,
        fx=float(camera_cfg["fx"]) * scale,
        fy=float(camera_cfg["fy"]) * scale,
        cx=float(camera_cfg["cx"]) * scale,
        cy=float(camera_cfg["cy"]) * scale,
    )


def load_input_frame(args, input_cfg: dict, root):
    import cv2  # type: ignore

    from vision_measurement.core.paths import require_existing_path

    image_override = args.image
    video_override = args.video
    if image_override:
        image_path = require_existing_path(image_override, "Image", base_dir=root)
        frame = cv2.imread(str(image_path))
        if frame is None:
            raise OSError(f"Cannot read image: {image_path}")
        return frame
    if video_override:
        return read_video_frame(video_override, args.frame_index, root)

    configured_image = input_cfg.get("image_path")
    if configured_image:
        try:
            image_path = require_existing_path(configured_image, "Image", base_dir=root)
            frame = cv2.imread(str(image_path))
            if frame is not None:
                return frame
        except FileNotFoundError:
            pass
    configured_video = input_cfg.get("video_path")
    if configured_video:
        return read_video_frame(configured_video, args.frame_index, root)
    raise SystemExit("Depth inference requires an existing image or video path")


def read_video_frame(video_path: str, frame_index: int, root):
    import cv2  # type: ignore

    from vision_measurement.core.paths import require_existing_path

    path = require_existing_path(video_path, "Video", base_dir=root)
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise OSError(f"Cannot open video: {path}")
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise RuntimeError(f"Cannot read frame {frame_index} from {path}")
    return frame


def detect_reference_person_foot(frame, yolo_path: str):
    from vision_measurement.distance.person import PersonPoseDetector

    people = PersonPoseDetector(yolo_path).detect(frame)
    if not people:
        raise RuntimeError("No person detected for scale calibration")
    person = max(
        people,
        key=lambda item: (item["box"][2] - item["box"][0]) * (item["box"][3] - item["box"][1]),
    )
    return person["ground_point"], person["box"]


def run_depth_video(args, config: dict, root) -> None:
    import cv2  # type: ignore

    from vision_measurement.core.paths import require_existing_path
    from vision_measurement.core.profiler import LatencyProfiler
    from vision_measurement.distance import AsyncDepthEstimator, DepthAnythingEstimator
    from vision_measurement.distance.depth import sample_depth
    from vision_measurement.distance.person import AsyncPersonDetector, PersonPoseDetector

    input_cfg = config.get("input", {})
    depth_cfg = config["depth"]
    camera_cfg = config["camera"]
    model_cfg = config["model"]
    video_cfg = config.get("video", {})
    measurement_cfg = config.get("measurement", {})

    video_path = require_existing_path(args.video or input_cfg.get("video_path"), "Video", base_dir=root)
    model_path = require_existing_path(model_cfg.get("yolo_path"), "YOLO model", base_dir=root)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise OSError(f"Cannot open video: {video_path}")

    source_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    source_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
    display_width = int(args.display_width or video_cfg.get("display_width", 1200))
    display_height = max(1, int(source_height * display_width / max(1, source_width)))
    depth_skip = max(1, int(args.depth_skip or video_cfg.get("depth_skip", 2)))
    detect_skip = max(1, int(args.detect_skip or video_cfg.get("detect_skip", 5)))
    scale_factor = float(depth_cfg["scale_factor"])
    projector = projector_from_config(scale_factor, camera_cfg, frame_width=display_width)

    depth_estimator = DepthAnythingEstimator(depth_cfg["model_repo"])
    depth_worker = AsyncDepthEstimator(depth_estimator, process_width=int(depth_cfg.get("process_width", 640)))
    person_worker = AsyncPersonDetector(
        PersonPoseDetector(str(model_path), confidence=float(model_cfg.get("confidence", 0.35)))
    )
    profiler = LatencyProfiler()
    depth_map = None
    people: list[dict[str, Any]] = []
    target = list(args.point_b or measurement_cfg.get("point_b", (display_width / 2, display_height / 2)))
    manual_a = tuple(args.point_a or measurement_cfg.get("point_a", (0.0, 0.0)))
    manual_b = tuple(args.point_b or measurement_cfg.get("point_b", (0.0, 0.0)))
    interaction = {"target": target}
    rows: list[dict[str, Any]] = []
    frame_idx = 0
    paused = False

    if not args.no_display:
        cv2.namedWindow("Depth Distance", cv2.WINDOW_NORMAL)

        def on_mouse(event, x, y, flags, param):  # noqa: ANN001
            if event == cv2.EVENT_LBUTTONDOWN:
                interaction["target"] = [float(x), float(y)]
            elif event == cv2.EVENT_RBUTTONDOWN:
                interaction["target"] = None

        cv2.setMouseCallback("Depth Distance", on_mouse)

    try:
        while True:
            loop_started = time.perf_counter()
            if not paused:
                ok, raw_frame = capture.read()
                if not ok:
                    break
                frame = cv2.resize(raw_frame, (display_width, display_height))

                if frame_idx % depth_skip == 0:
                    depth_worker.submit(frame_idx, frame)
                if frame_idx % detect_skip == 0:
                    person_worker.submit(frame_idx, frame)

                depth_output = depth_worker.poll()
                if depth_output is not None:
                    _, depth_map = depth_output
                person_output = person_worker.poll()
                if person_output is not None:
                    _, people = person_output

                frame_rows = measure_depth_frame(
                    frame_idx,
                    depth_map,
                    people,
                    interaction.get("target"),
                    manual_a,
                    manual_b,
                    projector,
                )
                rows.extend(frame_rows)
                if args.no_display:
                    for row in frame_rows:
                        print(json.dumps(row))
                else:
                    cv2.imshow(
                        "Depth Distance",
                        annotate_depth_frame(frame.copy(), frame_rows, people, interaction.get("target")),
                    )
                elapsed = time.perf_counter() - loop_started
                profiler.update("main_loop", elapsed * 1000.0)
                frame_idx += 1
                if args.max_frames is not None and frame_idx >= args.max_frames:
                    break
                if args.no_display:
                    time.sleep(max(0.0, (1.0 / fps) - elapsed))

            if not args.no_display:
                wait_ms = 30 if paused else max(1, int(1000.0 / fps))
                key = cv2.waitKey(wait_ms) & 0xFF
                if key == ord("q"):
                    break
                if key == ord(" "):
                    paused = not paused
            elif paused:
                paused = False
    finally:
        capture.release()
        depth_worker.close()
        person_worker.close()
        if not args.no_display:
            cv2.destroyAllWindows()

    write_rows_csv(args.csv_output, rows, root)
    print(json.dumps({"processed_frames": frame_idx, "latency": profiler.summary()}, indent=2))


def measure_depth_frame(
    frame_idx: int,
    depth_map,
    people: list[dict[str, Any]],
    target,
    manual_a,
    manual_b,
    projector,
) -> list[dict[str, Any]]:
    from vision_measurement.distance.depth import sample_depth

    if depth_map is None:
        return []
    rows: list[dict[str, Any]] = []

    point_a = projector.pixel_to_3d(*manual_a, sample_depth(depth_map, manual_a))
    point_b = projector.pixel_to_3d(*manual_b, sample_depth(depth_map, manual_b))
    if point_a is not None and point_b is not None:
        rows.append(
            {
                "type": "manual_depth_distance",
                "frame_idx": frame_idx,
                "distance_m": projector.distance(point_a, point_b),
                "point_a_px": list(manual_a),
                "point_b_px": list(manual_b),
            }
        )

    target_3d = None
    if target is not None:
        target_3d = projector.pixel_to_3d(*target, sample_depth(depth_map, target))
    for index, person in enumerate(people):
        ground = tuple(person["ground_point"])
        ground_3d = projector.pixel_to_3d(*ground, sample_depth(depth_map, ground))
        if ground_3d is None or target_3d is None:
            continue
        rows.append(
            {
                "type": "person_to_target_depth",
                "frame_idx": frame_idx,
                "person_index": index,
                "distance_m": projector.distance(ground_3d, target_3d),
                "ground_point_px": list(ground),
                "target_point_px": list(target),
                "box_xyxy": person["box"],
            }
        )
    return rows


def annotate_depth_frame(frame, rows, people, target):
    import cv2  # type: ignore

    if target is not None:
        cv2.circle(frame, tuple(map(int, target)), 6, (0, 255, 255), -1)
    for person in people:
        x1, y1, x2, y2 = map(int, person["box"])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 2)
        cv2.circle(frame, tuple(map(int, person["ground_point"])), 5, (0, 200, 0), -1)
    for row in rows:
        if row["type"] == "manual_depth_distance":
            point_a = tuple(map(int, row["point_a_px"]))
            point_b = tuple(map(int, row["point_b_px"]))
        else:
            point_a = tuple(map(int, row["ground_point_px"]))
            point_b = tuple(map(int, row["target_point_px"]))
        cv2.line(frame, point_a, point_b, (255, 255, 0), 2)
        midpoint = ((point_a[0] + point_b[0]) // 2, (point_a[1] + point_b[1]) // 2)
        cv2.putText(
            frame,
            f"{row['distance_m']:.2f}m",
            midpoint,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2,
        )
    return frame


def show_depth_preview(frame, point_a, point_b, distance_m: float) -> None:
    import cv2  # type: ignore

    row = {
        "type": "manual_depth_distance",
        "point_a_px": point_a,
        "point_b_px": point_b,
        "distance_m": distance_m,
    }
    cv2.imshow("Depth Distance", annotate_depth_frame(frame.copy(), [row], [], None))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def write_rows_csv(path_value: str | None, rows: list[dict[str, Any]], root) -> None:
    if not path_value or not rows:
        return
    from vision_measurement.core.paths import resolve_path

    output = resolve_path(path_value, base_dir=root)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value) if isinstance(value, (list, tuple, dict)) else value
                    for key, value in row.items()
                }
            )


if __name__ == "__main__":
    main()

