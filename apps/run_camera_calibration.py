from __future__ import annotations

import argparse
from glob import glob

from _bootstrap import bootstrap_src_path


def _pair(value: str) -> tuple[float, float]:
    try:
        width, height = value.lower().split("x", 1)
        return float(width), float(height)
    except Exception as exc:
        raise argparse.ArgumentTypeError("Value must look like 9x6 or 5.6x4.2") from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Calibrate a camera from chessboard images.")
    parser.add_argument("images", nargs="*", help="Image paths or glob patterns.")
    parser.add_argument("--config", default="configs/camera_calibration.json", help="Calibration config JSON.")
    parser.add_argument("--pattern-size", type=_pair, help="Inner corners, e.g. 9x6.")
    parser.add_argument("--square-size-m", type=float, help="Chessboard square size in meters.")
    parser.add_argument("--fisheye", action="store_true", help="Use OpenCV fisheye calibration.")
    parser.add_argument("--sensor-size-mm", type=_pair, help="Optional physical sensor size, e.g. 5.6x4.2.")
    parser.add_argument("--output", help="Output JSON file.")
    parser.add_argument("--debug-dir", help="Debug image directory.")
    parser.add_argument("--no-undistorted", action="store_true", help="Skip undistorted debug images.")
    return parser


def main() -> None:
    root = bootstrap_src_path()
    args = build_parser().parse_args()

    from vision_measurement.calibration import calibrate_chessboard
    from vision_measurement.core.config import load_json_config
    from vision_measurement.core.paths import require_existing_path, resolve_path

    config_path = require_existing_path(args.config, "Config", base_dir=root)
    config = load_json_config(config_path)
    input_cfg = config.get("input", {})
    calibration_cfg = config.get("camera_calibration", {})

    patterns = args.images or [input_cfg.get("image_glob")]
    image_files: list[str] = []
    for pattern in (item for item in patterns if item):
        resolved_pattern = str(resolve_path(pattern, base_dir=root))
        matches = glob(resolved_pattern)
        if not matches:
            raise FileNotFoundError(f"No calibration images match: {resolved_pattern}")
        image_files.extend(matches)

    pattern = args.pattern_size or tuple(calibration_cfg.get("pattern_size", [9, 6]))
    pattern_size = (int(pattern[0]), int(pattern[1]))
    square_size = (
        args.square_size_m
        if args.square_size_m is not None
        else float(calibration_cfg.get("square_size_m", 0.034))
    )
    sensor_size = args.sensor_size_mm or calibration_cfg.get("sensor_size_mm")
    if sensor_size:
        sensor_size = (float(sensor_size[0]), float(sensor_size[1]))
    output = resolve_path(
        args.output or calibration_cfg.get("output_json", "configs/generated_camera_calibration.json"),
        base_dir=root,
    )
    debug_dir_value = args.debug_dir or calibration_cfg.get("debug_dir")
    debug_dir = resolve_path(debug_dir_value, base_dir=root) if debug_dir_value else None

    result = calibrate_chessboard(
        image_files=image_files,
        pattern_size=pattern_size,
        square_size_m=square_size,
        fisheye=args.fisheye or bool(calibration_cfg.get("fisheye", False)),
        output_json=output,
        debug_dir=debug_dir,
        sensor_size_mm=sensor_size,
        write_undistorted=not args.no_undistorted,
    )
    error = result["reprojection_error"]
    print(
        f"Calibrated {result['image_count']}/{result['input_image_count']} images; "
        f"RMS={result['rms']:.6f}; reprojection={error['average']:.6f}+/-{error['stddev']:.6f}; "
        f"wrote {output}"
    )


if __name__ == "__main__":
    main()
