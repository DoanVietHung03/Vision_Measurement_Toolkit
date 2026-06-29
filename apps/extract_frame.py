from __future__ import annotations

import argparse

from _bootstrap import bootstrap_src_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract one frame from a video.")
    parser.add_argument("--video", required=True, help="Input video path.")
    parser.add_argument("--frame-index", type=int, default=0, help="0-based frame index.")
    parser.add_argument("--output", required=True, help="Output image path.")
    return parser


def main() -> None:
    bootstrap_src_path()
    args = build_parser().parse_args()
    from vision_measurement.core.paths import require_existing_path, resolve_path
    from vision_measurement.core.video import extract_frame

    video = require_existing_path(args.video, "Video")
    output = resolve_path(args.output)
    saved = extract_frame(video, args.frame_index, output)
    print(f"Saved frame {args.frame_index} to {saved}")


if __name__ == "__main__":
    main()
