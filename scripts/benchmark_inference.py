#!/usr/bin/env python3
"""
CatBowlWatch — Inference KPI benchmark (Phase 5).

Measures throughput and latency across ONNX and TensorRT execution providers.
Designed to run on Orin Nano (JetPack 6.x) but onnx-cpu works on any machine.

Usage
-----
  python scripts/benchmark_inference.py \\
      --model models/catbowlwatch.onnx \\
      --backend onnx-cpu \\
      --frames 200 \\
      --warmup 20

  # JSON output for scripting:
  python scripts/benchmark_inference.py --model ... --backend onnx-cpu --json

Backends
--------
  onnx-cpu   ONNX Runtime, CPU execution provider  (laptop + Orin Nano)
  onnx-cuda  ONNX Runtime, CUDA execution provider (Orin Nano)
  trt-fp16   ONNX Runtime, TensorRT EP, FP16        (Orin Nano)
  trt-int8   ONNX Runtime, TensorRT EP, INT8        (Orin Nano)

Output columns (docs/BENCHMARKS.md)
------------------------------------
  backend | fps | mean_ms | p99_ms | rss_mb
"""
import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import psutil

INPUT_SHAPE = (1, 3, 640, 640)

# ── Pure stat helpers (unit-testable without onnxruntime) ────────────────────

def compute_stats(frame_times_ms: list) -> dict:
    """Return {fps, mean_ms, p99_ms} from a list of per-frame wall-clock times in ms."""
    if not frame_times_ms:
        raise ValueError("frame_times_ms must not be empty")
    sorted_times = sorted(frame_times_ms)
    n = len(sorted_times)
    p99_idx = max(0, int(n * 0.99) - 1)
    mean_ms = statistics.mean(frame_times_ms)
    return {
        "fps":     1000.0 / mean_ms,
        "mean_ms": mean_ms,
        "p99_ms":  sorted_times[p99_idx],
    }


def make_dummy_input(shape: tuple = INPUT_SHAPE) -> dict:
    """Return a random float32 input dict matching the YOLOv8n input contract."""
    return {"input": np.random.rand(*shape).astype(np.float32)}


def get_rss_mb() -> float:
    """Return RSS memory of the current process in MB."""
    return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)


def build_providers(backend: str) -> list:
    """Return an onnxruntime providers list for the requested backend."""
    if backend == "onnx-cpu":
        return ["CPUExecutionProvider"]
    if backend == "onnx-cuda":
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    if backend == "trt-fp16":
        return [
            ("TensorrtExecutionProvider", {
                "trt_fp16_enable": True,
                "trt_engine_cache_enable": True,
                "trt_engine_cache_path": "/tmp/trt_cache_fp16",
            }),
            "CPUExecutionProvider",
        ]
    if backend == "trt-int8":
        return [
            ("TensorrtExecutionProvider", {
                "trt_int8_enable": True,
                "trt_engine_cache_enable": True,
                "trt_engine_cache_path": "/tmp/trt_cache_int8",
            }),
            "CPUExecutionProvider",
        ]
    raise ValueError(f"Unknown backend: {backend!r}. "
                     f"Choose from: onnx-cpu, onnx-cuda, trt-fp16, trt-int8")


def format_markdown_row(result: dict) -> str:
    """Format a BENCHMARKS.md table row for one backend result."""
    return (
        f"| `{result['backend']}` "
        f"| {result['fps']:.1f} "
        f"| {result['mean_ms']:.1f} "
        f"| {result['p99_ms']:.1f} "
        f"| {result['rss_mb']:.0f} |"
    )


# ── Benchmark runner (requires onnxruntime) ───────────────────────────────────

def run_benchmark(model_path: str, backend: str,
                  n_warmup: int = 20, n_frames: int = 200) -> dict:
    """Load model and time inference. Returns KPI dict."""
    try:
        import onnxruntime as ort
    except ImportError:
        print(
            "ERROR: onnxruntime not installed.\n"
            "  Poetry env:  poetry install --with training\n"
            "  ROCm/CUDA:   pip install onnxruntime  (in your active venv)",
            file=sys.stderr,
        )
        sys.exit(1)

    providers = build_providers(backend)
    try:
        session = ort.InferenceSession(model_path, providers=providers)
    except Exception as exc:
        print(f"ERROR: could not load {model_path!r} with backend {backend!r}:\n  {exc}",
              file=sys.stderr)
        sys.exit(1)

    input_name = session.get_inputs()[0].name
    dummy = np.random.rand(*INPUT_SHAPE).astype(np.float32)

    print(f"Warm-up ({n_warmup} frames)...", file=sys.stderr)
    for _ in range(n_warmup):
        session.run(None, {input_name: dummy})

    print(f"Timing ({n_frames} frames)...", file=sys.stderr)
    rss_before = get_rss_mb()
    frame_times_ms = []
    for _ in range(n_frames):
        t0 = time.perf_counter()
        session.run(None, {input_name: dummy})
        frame_times_ms.append((time.perf_counter() - t0) * 1000.0)
    rss_after = get_rss_mb()

    result = compute_stats(frame_times_ms)
    result["rss_mb"]  = rss_after
    result["backend"] = backend
    result["model"]   = Path(model_path).name
    result["n_frames"] = n_frames
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model",   required=True,
                        help="Path to .onnx model")
    parser.add_argument("--backend", required=True,
                        choices=["onnx-cpu", "onnx-cuda", "trt-fp16", "trt-int8"],
                        help="Execution provider")
    parser.add_argument("--frames",  type=int, default=200,
                        help="Frames to time (default: 200)")
    parser.add_argument("--warmup",  type=int, default=20,
                        help="Warm-up frames (default: 20)")
    parser.add_argument("--json",    action="store_true",
                        help="Print JSON instead of human-readable table")
    args = parser.parse_args()

    result = run_benchmark(args.model, args.backend, args.warmup, args.frames)

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"\nBackend:    {result['backend']}")
        print(f"Model:      {result['model']}")
        print(f"Frames:     {result['n_frames']}")
        print(f"Throughput: {result['fps']:.1f} FPS")
        print(f"Mean:       {result['mean_ms']:.1f} ms/frame")
        print(f"P99:        {result['p99_ms']:.1f} ms/frame")
        print(f"RSS:        {result['rss_mb']:.0f} MB")
        print()
        print("Markdown row for docs/BENCHMARKS.md:")
        print(format_markdown_row(result))


if __name__ == "__main__":
    main()
