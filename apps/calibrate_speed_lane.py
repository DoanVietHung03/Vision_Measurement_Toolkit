from __future__ import annotations

import argparse
import json

from _bootstrap import bootstrap_src_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Optimize lane BEV calibration from ground-truth distances.")
    parser.add_argument("--config", default="configs/speed_vehicle_lane.json", help="Lane speed config JSON.")
    parser.add_argument("--pkl", help="Ground-truth PKL override.")
    parser.add_argument("--video", help="Video override used by interactive point selection.")
    parser.add_argument("--interactive", action="store_true", help="Select four source points on the first video frame.")
    parser.add_argument("--target-width", type=float, help="BEV target width override.")
    parser.add_argument("--ratio-min", type=float, default=0.5)
    parser.add_argument("--ratio-max", type=float, default=15.0)
    parser.add_argument("--ratio-steps", type=int, default=150)
    parser.add_argument("--output", help="Optional result JSON path.")
    return parser


def main() -> None:
    root = bootstrap_src_path()
    args = build_parser().parse_args()

    from vision_measurement.core.config import load_json_config
    from vision_measurement.core.paths import require_existing_path, resolve_path
    from vision_measurement.speed import optimize_lane_calibration
    from vision_measurement.speed.validation import load_pickle_distance_measurements

    config_path = require_existing_path(args.config, "Config", base_dir=root)
    config = load_json_config(config_path)
    input_cfg = config.get("input", {})
    calibration_cfg = config["calibration"]

    pkl_path = require_existing_path(
        args.pkl or input_cfg.get("ground_truth_pkl"),
        "Ground-truth PKL",
        base_dir=root,
    )
    measurements = load_pickle_distance_measurements(str(pkl_path))
    source_points = calibration_cfg["source_points"]
    if args.interactive:
        video_path = require_existing_path(
            args.video or input_cfg.get("video_path"),
            "Video",
            base_dir=root,
        )
        source_points = pick_source_points(str(video_path), measurements)

    result = optimize_lane_calibration(
        source_points=source_points,
        measurements=measurements,
        target_width=args.target_width or float(calibration_cfg.get("target_width", 800.0)),
        ratio_min=args.ratio_min,
        ratio_max=args.ratio_max,
        ratio_steps=args.ratio_steps,
    )
    payload = result.as_dict()
    print(json.dumps(payload, indent=2))
    if args.output:
        output_path = resolve_path(args.output, base_dir=root)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def pick_source_points(video_path: str, measurements: list[dict]) -> list[list[float]]:
    import cv2  # type: ignore
    import numpy as np

    cap = cv2.VideoCapture(video_path)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Cannot read first frame from {video_path}")

    canvas = frame.copy()
    for item in measurements:
        p1 = tuple(np.asarray(item["p1"], dtype=int))
        p2 = tuple(np.asarray(item["p2"], dtype=int))
        cv2.line(canvas, p1, p2, (255, 255, 0), 2)

    points: list[list[float]] = []

    def on_mouse(event, x, y, flags, param):  # noqa: ANN001
        if event != cv2.EVENT_LBUTTONDOWN or len(points) >= 4:
            return
        points.append([float(x), float(y)])
        cv2.circle(canvas, (x, y), 5, (0, 0, 255), -1)
        if len(points) == 4:
            cv2.polylines(canvas, [np.array(points, dtype=np.int32)], True, (0, 255, 0), 2)

    window = "Lane Calibration"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window, on_mouse)
    while len(points) < 4:
        cv2.imshow(window, canvas)
        key = cv2.waitKey(20) & 0xFF
        if key in (ord("q"), 27):
            break
    cv2.destroyAllWindows()
    if len(points) != 4:
        raise RuntimeError("Lane calibration requires exactly four source points")
    return points


if __name__ == "__main__":
    main()
