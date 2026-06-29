from __future__ import annotations

import argparse
import json
from pathlib import Path

from _bootstrap import bootstrap_src_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export and validate an Ultralytics YOLO model as ONNX.")
    parser.add_argument("--model", required=True, help="Input .pt model path.")
    parser.add_argument("--imgsz", type=int, default=640, help="Export image size.")
    parser.add_argument("--half", action="store_true", help="Export FP16 when supported.")
    parser.add_argument("--dynamic", action="store_true", help="Enable dynamic input shape.")
    parser.add_argument("--opset", type=int, default=12, help="ONNX opset.")
    parser.add_argument(
        "--simplify",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Simplify the exported graph.",
    )
    parser.add_argument(
        "--validate",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run ONNX checker and ONNX Runtime smoke inference.",
    )
    return parser


def main() -> None:
    bootstrap_src_path()
    args = build_parser().parse_args()

    from vision_measurement.core.paths import require_existing_path

    model_path = require_existing_path(args.model, "YOLO model")
    try:
        from ultralytics import YOLO  # type: ignore
    except ImportError as exc:
        raise ImportError("ultralytics is required for ONNX export") from exc

    output_value = YOLO(str(model_path)).export(
        format="onnx",
        imgsz=args.imgsz,
        half=args.half,
        dynamic=args.dynamic,
        opset=args.opset,
        simplify=args.simplify,
    )
    output_path = Path(output_value)
    payload = {"output": str(output_path), "validated": False}
    if args.validate:
        payload["validation"] = validate_onnx(output_path, args.imgsz)
        payload["validated"] = True
    print(json.dumps(payload, indent=2))


def validate_onnx(model_path: Path, image_size: int) -> dict:
    try:
        import numpy as np
        import onnx
        import onnxruntime as ort
    except ImportError as exc:
        raise ImportError("ONNX validation requires onnx and onnxruntime") from exc

    model = onnx.load(str(model_path))
    onnx.checker.check_model(model)
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    input_meta = session.get_inputs()[0]
    shape = [dimension if isinstance(dimension, int) and dimension > 0 else 1 for dimension in input_meta.shape]
    if len(shape) == 4:
        shape[0] = 1
        shape[1] = 3
        shape[2] = image_size
        shape[3] = image_size
    dtype = np.float16 if "float16" in input_meta.type else np.float32
    outputs = session.run(None, {input_meta.name: np.random.randn(*shape).astype(dtype)})
    return {
        "input_name": input_meta.name,
        "input_shape": shape,
        "output_shapes": [list(output.shape) for output in outputs],
        "opsets": [
            {"domain": item.domain, "version": int(item.version)}
            for item in model.opset_import
        ],
    }


if __name__ == "__main__":
    main()
