"""Video and image helpers shared by app entrypoints."""

from __future__ import annotations

from pathlib import Path


def require_cv2():
    """Import OpenCV lazily so non-video tests can run without import side effects."""
    try:
        import cv2  # type: ignore
    except ImportError as exc:
        raise ImportError("OpenCV is required for this operation. Install opencv-python.") from exc
    return cv2


def extract_frame(video_path: str | Path, frame_index: int, output_path: str | Path) -> Path:
    """Extract a single frame from a video."""
    cv2 = require_cv2()
    video = Path(video_path)
    output = Path(output_path)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise OSError(f"Cannot open video: {video}")
    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames > 0 and frame_index >= total_frames:
            raise ValueError(f"frame_index {frame_index} exceeds total frames {total_frames}")
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError(f"Cannot read frame {frame_index} from {video}")
        output.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(output), frame):
            raise OSError(f"Cannot write frame to: {output}")
    finally:
        cap.release()
    return output
