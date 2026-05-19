"""Export trained YOLOv8n .pt to ONNX opset 17 + verify output shape.

The output shape contract from ARCHITECTURE.md §4.3 is non-negotiable for the
C++ postprocessor: [1, 4 + num_classes, anchors] = [1, 6, 8400] for a 640x640
input with 2 classes. This script exports the model and then runs a single
forward pass through onnxruntime to confirm the shape — if it disagrees, the
C++ inference service cannot consume the model.

Anchor count derivation (640x640 input, YOLOv8 P3/P4/P5 strides 8/16/32):
    80*80 + 40*40 + 20*20 = 6400 + 1600 + 400 = 8400

Requires the training Poetry group:
    poetry install --with training

Usage:
    poetry run python training/export.py
    poetry run python training/export.py --weights models/catbowlwatch.pt --opset 17
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

DEFAULT_WEIGHTS = "models/catbowlwatch.pt"
DEFAULT_OUTPUT = "models/catbowlwatch.onnx"
DEFAULT_OPSET = 17
DEFAULT_IMGSZ = 640
NUM_CLASSES = 2
EXPECTED_ANCHORS = 8400
EXPECTED_OUTPUT_SHAPE: tuple[int, int, int] = (1, 4 + NUM_CLASSES, EXPECTED_ANCHORS)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Export YOLOv8n .pt to ONNX and verify output shape.")
    p.add_argument("--weights", default=DEFAULT_WEIGHTS, help=f"Trained .pt (default: {DEFAULT_WEIGHTS})")
    p.add_argument("--output", default=DEFAULT_OUTPUT, help=f"Destination .onnx (default: {DEFAULT_OUTPUT})")
    p.add_argument("--opset", type=int, default=DEFAULT_OPSET)
    p.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ)
    p.add_argument("--no-simplify", action="store_true", help="Disable onnxsim graph simplification")
    return p


def verify_onnx_output_shape(
    onnx_path: Path,
    expected: tuple[int, int, int] = EXPECTED_OUTPUT_SHAPE,
) -> tuple[int, ...]:
    """Run one forward pass with a zero input and return the output shape.

    Raises if the shape disagrees with expected.
    """
    import numpy as np
    import onnxruntime as ort

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0]
    dummy_shape = tuple(d if isinstance(d, int) and d > 0 else 1 for d in inp.shape)
    dummy = np.zeros(dummy_shape, dtype=np.float32)
    out = sess.run(None, {inp.name: dummy})[0]
    actual = tuple(out.shape)
    if actual != expected:
        raise RuntimeError(
            f"ONNX output shape {actual} != expected {expected}. "
            "C++ postprocessor in ARCHITECTURE.md §4.3 assumes the expected layout."
        )
    return actual


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)

    weights = Path(args.weights)
    if not weights.is_file():
        print(f"Error: weights not found: {weights}", file=sys.stderr)
        return 2

    from ultralytics import YOLO

    model = YOLO(str(weights))
    exported = model.export(
        format="onnx",
        opset=args.opset,
        imgsz=args.imgsz,
        simplify=not args.no_simplify,
    )

    exported_path = Path(exported)
    target = Path(args.output)
    if exported_path.resolve() != target.resolve():
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(exported_path), target)

    try:
        actual = verify_onnx_output_shape(target)
    except RuntimeError as e:
        print(f"FAIL: {e}", file=sys.stderr)
        return 1
    print(f"OK: {target}  shape={actual}  opset={args.opset}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
