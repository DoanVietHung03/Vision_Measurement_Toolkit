from __future__ import annotations

import argparse

from _bootstrap import bootstrap_src_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Click image/video-frame points and print pixel coordinates.")
    parser.add_argument("--image", help="Image path to inspect.")
    parser.add_argument("--video", help="Video path; first frame is used unless --frame-index is provided.")
    parser.add_argument("--frame-index", type=int, default=0, help="Frame index when using --video.")
    parser.add_argument("--display-width", type=int, default=1200, help="Display width while preserving aspect ratio.")
    return parser


def main() -> None:
    bootstrap_src_path()
    args = build_parser().parse_args()
    if not args.image and not args.video:
        raise SystemExit("Provide either --image or --video.")

    from vision_measurement.core.paths import require_existing_path
    from vision_measurement.core.video import require_cv2

    cv2 = require_cv2()
    if args.image:
        img_path = require_existing_path(args.image, "Image")
        frame = cv2.imread(str(img_path))
        if frame is None:
            raise OSError(f"Cannot read image: {img_path}")
    else:
        video_path = require_existing_path(args.video, "Video")
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise OSError(f"Cannot open video: {video_path}")
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame_index)
        ok, frame = cap.read()
        cap.release()
        if not ok:
            raise RuntimeError(f"Cannot read frame {args.frame_index}")

    h, w = frame.shape[:2]
    scale = args.display_width / float(w)
    display = cv2.resize(frame, (args.display_width, int(h * scale)))
    points: list[tuple[int, int]] = []

    def on_mouse(event, x, y, flags, param):  # noqa: ANN001
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        points.append((x, y))
        original = (x / scale, y / scale)
        print(f"{len(points)}: display=({x}, {y}) original=({original[0]:.1f}, {original[1]:.1f})")
        cv2.circle(display, (x, y), 5, (0, 0, 255), -1)
        cv2.imshow("Pick Pixels", display)

    cv2.namedWindow("Pick Pixels", cv2.WINDOW_NORMAL)
    cv2.imshow("Pick Pixels", display)
    cv2.setMouseCallback("Pick Pixels", on_mouse)
    print("Left-click to add points. Press q or ESC to quit.")
    while True:
        key = cv2.waitKey(20) & 0xFF
        if key in (ord("q"), 27):
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
