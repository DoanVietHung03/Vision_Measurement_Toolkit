from __future__ import annotations

import argparse
import json

from _bootstrap import bootstrap_src_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run or validate lane speed measurement config.")
    parser.add_argument("--config", default="configs/speed_vehicle_lane.json", help="Lane speed config JSON.")
    parser.add_argument("--validate-pkl", help="Optional ground-truth .pkl for BEV distance validation.")
    parser.add_argument("--dry-run", action="store_true", help="Only resolve config and print readiness.")
    parser.add_argument("--max-frames", type=int, default=None, help="Optional frame limit for smoke tests.")
    parser.add_argument("--no-display", action="store_true", help="Print JSON lines instead of opening a GUI window.")
    return parser


def main() -> None:
    root = bootstrap_src_path()
    args = build_parser().parse_args()

    from vision_measurement.calibration import PlaneCalibration
    from vision_measurement.core.config import load_json_config
    from vision_measurement.core.paths import require_existing_path, resolve_path
    from vision_measurement.speed.validation import load_pickle_distance_measurements, validate_distance_measurements

    config_path = resolve_path(args.config, base_dir=root)
    config = load_json_config(config_path)
    calibration = PlaneCalibration.from_config(config)

    if args.validate_pkl:
        pkl_path = require_existing_path(args.validate_pkl, "Ground-truth PKL", base_dir=root)
        measurements = load_pickle_distance_measurements(str(pkl_path))
        print(json.dumps(validate_distance_measurements(calibration, measurements), indent=2))
        return

    input_cfg = config.get("input", {})
    model_cfg = config.get("model", {})
    readiness = {
        "config": str(config_path),
        "video_path": str(resolve_path(input_cfg.get("video_path"), base_dir=root)),
        "model_path": str(resolve_path(model_cfg.get("path"), base_dir=root)),
        "source_points": config["calibration"]["source_points"],
        "target_width": config["calibration"].get("target_width"),
        "target_height": config["calibration"].get("target_height"),
        "meters_per_pixel": config["calibration"].get("meters_per_pixel"),
    }
    if args.dry_run:
        print(json.dumps(readiness, indent=2))
        return

    video_path = require_existing_path(input_cfg.get("video_path"), "Video", base_dir=root)
    model_path = require_existing_path(model_cfg.get("path"), "YOLO model", base_dir=root)
    run_video_speed_pipeline(
        video_path=str(video_path),
        model_path=str(model_path),
        config=config,
        calibration=calibration,
        max_frames=args.max_frames,
        no_display=args.no_display,
    )


def run_video_speed_pipeline(
    video_path: str,
    model_path: str,
    config: dict,
    calibration,
    max_frames: int | None,
    no_display: bool,
) -> None:
    try:
        import cv2  # type: ignore
        import numpy as np
        import supervision as sv  # type: ignore
        import torch
        from ultralytics import YOLO  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "Running lane speed video requires opencv-python, supervision, torch, and ultralytics."
        ) from exc

    from vision_measurement.speed import SpeedEstimator

    model_cfg = config.get("model", {})
    speed_cfg = config.get("speed", {})
    target_classes = model_cfg.get("target_classes")
    tracker = model_cfg.get("tracker", "bytetrack.yaml")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    video_info = sv.VideoInfo.from_video_path(video_path)
    fps = float(video_info.fps or speed_cfg.get("default_fps", 25.0))
    estimator = SpeedEstimator(
        fps=fps,
        meters_per_pixel=float(config["calibration"].get("meters_per_pixel", calibration.meters_per_pixel)),
        alpha=float(speed_cfg.get("ema_alpha", 0.3)),
    )
    model = YOLO(model_path, task="detect")

    polygon = np.array(config["calibration"]["source_points"], dtype=np.int32)
    zone = sv.PolygonZone(polygon=polygon)
    frame_generator = sv.get_video_frames_generator(video_path)

    if not no_display:
        box_annotator = sv.BoxAnnotator(thickness=2)
        label_annotator = sv.LabelAnnotator(
            text_scale=0.5,
            text_thickness=1,
            text_padding=10,
            text_position=sv.Position.TOP_CENTER,
        )
        trace_annotator = sv.TraceAnnotator(
            thickness=2,
            trace_length=int(fps * 2),
            position=sv.Position.BOTTOM_CENTER,
        )
        zone_annotator = sv.PolygonZoneAnnotator(zone=zone, color=sv.Color.GREEN, thickness=2)
        cv2.namedWindow("Vision Measurement Speed Lane", cv2.WINDOW_NORMAL)

    for frame_idx, frame in enumerate(frame_generator):
        if max_frames is not None and frame_idx >= max_frames:
            break

        result = model.track(
            frame,
            classes=target_classes,
            persist=True,
            verbose=False,
            tracker=tracker,
            device=device,
        )[0]
        detections = sv.Detections.from_ultralytics(result)

        if detections.tracker_id is None:
            if not no_display:
                cv2.imshow("Vision Measurement Speed Lane", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            continue

        valid = detections[zone.trigger(detections=detections)]
        points = valid.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
        points_bev = calibration.transform_points(points)

        labels = []
        rows = []
        for track_id, point_bev in zip(valid.tracker_id, points_bev):
            estimate = estimator.update(int(track_id), point_bev, frame_idx=frame_idx)
            labels.append(f"#{int(track_id)} {estimate.speed_kmh:.0f} km/h")
            rows.append(
                {
                    "frame_idx": frame_idx,
                    "track_id": int(track_id),
                    "point_bev": [float(point_bev[0]), float(point_bev[1])],
                    "speed_kmh": estimate.speed_kmh,
                    "raw_speed_kmh": estimate.metadata.get("raw_speed_kmh"),
                }
            )

        if no_display:
            for row in rows:
                print(json.dumps(row))
            continue

        frame = zone_annotator.annotate(scene=frame)
        frame = trace_annotator.annotate(scene=frame, detections=detections)
        frame = box_annotator.annotate(scene=frame, detections=valid)
        frame = label_annotator.annotate(scene=frame, detections=valid, labels=labels)
        cv2.imshow("Vision Measurement Speed Lane", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    if not no_display:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
