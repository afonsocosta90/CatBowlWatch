"""Tests for training/export.py argument parsing and the shape contract.

Does not import ultralytics or onnxruntime — both are lazy-imported.
"""

from training.export import (
    DEFAULT_IMGSZ,
    DEFAULT_OPSET,
    DEFAULT_OUTPUT,
    DEFAULT_WEIGHTS,
    EXPECTED_ANCHORS,
    EXPECTED_OUTPUT_SHAPE,
    NUM_CLASSES,
    build_arg_parser,
)


def test_arg_parser_defaults():
    args = build_arg_parser().parse_args([])
    assert args.weights == DEFAULT_WEIGHTS
    assert args.output == DEFAULT_OUTPUT
    assert args.opset == DEFAULT_OPSET
    assert args.imgsz == DEFAULT_IMGSZ
    assert args.no_simplify is False


def test_arg_parser_overrides():
    args = build_arg_parser().parse_args([
        "--weights", "foo.pt",
        "--output", "bar.onnx",
        "--opset", "16",
        "--imgsz", "416",
        "--no-simplify",
    ])
    assert args.weights == "foo.pt"
    assert args.output == "bar.onnx"
    assert args.opset == 16
    assert args.imgsz == 416
    assert args.no_simplify is True


def test_expected_output_shape_matches_architecture():
    # ARCHITECTURE.md §4.3: [1, 4 bbox + num_classes, anchors]
    assert NUM_CLASSES == 2
    assert EXPECTED_ANCHORS == 8400
    assert EXPECTED_OUTPUT_SHAPE == (1, 4 + NUM_CLASSES, EXPECTED_ANCHORS)
    assert EXPECTED_OUTPUT_SHAPE == (1, 6, 8400)


def test_anchor_count_matches_640_input():
    # Derivation for 640x640 input with P3/P4/P5 strides 8/16/32:
    #   80*80 + 40*40 + 20*20 = 8400
    strides = (8, 16, 32)
    img = 640
    anchors = sum((img // s) ** 2 for s in strides)
    assert anchors == EXPECTED_ANCHORS
