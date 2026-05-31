"""Unit tests for scripts/benchmark_inference.py.

Tests cover pure helper functions only — no onnxruntime or model required.
"""
import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from benchmark_inference import (
    compute_stats,
    make_dummy_input,
    build_providers,
    format_markdown_row,
    INPUT_SHAPE,
)


# ── compute_stats ─────────────────────────────────────────────────────────────

def test_compute_stats_fps():
    # 100 ms per frame → 10 FPS
    stats = compute_stats([100.0] * 10)
    assert abs(stats["fps"] - 10.0) < 0.01


def test_compute_stats_mean():
    stats = compute_stats([100.0, 200.0, 300.0])
    assert abs(stats["mean_ms"] - 200.0) < 0.01


def test_compute_stats_p99_single_frame():
    stats = compute_stats([42.0])
    assert stats["p99_ms"] == pytest.approx(42.0)


def test_compute_stats_p99_sorted_correctly():
    # 100 frames: 1..100 ms. P99 index = int(100*0.99)-1 = 98 → 99 ms
    times = list(range(1, 101))
    stats = compute_stats(times)
    assert stats["p99_ms"] >= 99.0


def test_compute_stats_empty_raises():
    with pytest.raises(ValueError):
        compute_stats([])


# ── make_dummy_input ──────────────────────────────────────────────────────────

def test_dummy_input_shape():
    dummy = make_dummy_input()
    assert dummy["input"].shape == INPUT_SHAPE


def test_dummy_input_dtype():
    dummy = make_dummy_input()
    assert dummy["input"].dtype.name == "float32"


def test_dummy_input_custom_shape():
    dummy = make_dummy_input(shape=(1, 3, 416, 416))
    assert dummy["input"].shape == (1, 3, 416, 416)


# ── build_providers ───────────────────────────────────────────────────────────

def test_build_providers_onnx_cpu():
    providers = build_providers("onnx-cpu")
    assert providers == ["CPUExecutionProvider"]


def test_build_providers_onnx_cuda_includes_cpu_fallback():
    providers = build_providers("onnx-cuda")
    assert "CUDAExecutionProvider" in providers
    assert "CPUExecutionProvider" in providers


def test_build_providers_trt_fp16_sets_fp16_flag():
    providers = build_providers("trt-fp16")
    trt_ep_name, trt_ep_opts = providers[0]
    assert trt_ep_name == "TensorrtExecutionProvider"
    assert trt_ep_opts.get("trt_fp16_enable") is True


def test_build_providers_trt_int8_sets_int8_flag():
    providers = build_providers("trt-int8")
    trt_ep_name, trt_ep_opts = providers[0]
    assert trt_ep_name == "TensorrtExecutionProvider"
    assert trt_ep_opts.get("trt_int8_enable") is True


def test_build_providers_unknown_backend_raises():
    with pytest.raises(ValueError, match="Unknown backend"):
        build_providers("some-invalid-backend")


# ── format_markdown_row ───────────────────────────────────────────────────────

def test_format_markdown_row_starts_with_backend():
    row = format_markdown_row({
        "backend": "onnx-cpu",
        "fps": 12.3,
        "mean_ms": 81.2,
        "p99_ms": 95.4,
        "rss_mb": 312.0,
    })
    assert row.startswith("| `onnx-cpu`")


def test_format_markdown_row_contains_all_metrics():
    row = format_markdown_row({
        "backend": "trt-fp16",
        "fps": 34.5,
        "mean_ms": 29.0,
        "p99_ms": 38.1,
        "rss_mb": 450.0,
    })
    assert "34.5" in row
    assert "29.0" in row
    assert "38.1" in row
    assert "450" in row
