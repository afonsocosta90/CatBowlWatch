# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Status

Phase 0 (documentation complete, no code written yet). Phases 1–5 are planned sequentially — do not implement features from a later phase when working in an earlier one.

## Commands

Commands below reflect the planned scripts and Docker setup. Update this section as files are created.

```bash
# Demo (Docker — no Jetson needed)
cp demo/.env.example .env   # fill in TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID
docker compose -f docker/demo.yml up

# C++ inference service (laptop, ONNX)
mkdir -p inference/build && cd inference/build
cmake -DWITH_TENSORRT=OFF ..
make -j$(nproc)

# C++ inference service (Jetson, TensorRT)
cmake -DWITH_TENSORRT=ON ..
make -j$(nproc)

# Python training pipeline
cd training
python train.py          # trains YOLOv8n, saves .pt to models/
python export.py         # exports .pt → ONNX opset 17

# Verify ONNX output shape (run before writing C++ postprocessor)
python -c "import onnxruntime as ort, numpy as np; s=ort.InferenceSession('models/catbowlwatch.onnx'); print(s.run(None,{s.get_inputs()[0].name:np.zeros((1,3,640,640),dtype='f4')})[0].shape)"

# Tests
cd tests && python -m pytest           # ONNX/TRT parity + debounce unit tests
```

## Architecture

Single-device edge pipeline: camera → YOLOv8n → C++17 service → Telegram. No cloud component except the outbound Telegram Bot API call.

**Frame pipeline (per frame):**
```
Frame Capture → Preprocessor → YOLOv8n → Postprocessor → Bowl Tracker → Debounce Engine → Telegram Notifier
                                                        ↘                ↘
                                                         HTTP Service (/status, /photo)
```

**Capture:** `cv::VideoCapture` on laptop; GStreamer `nvarguscamerasrc` on Jetson (Phase 5 swap — interface is identical, single `read(frame)` call).

**Preprocessor output:** `float32` tensor `[1, 3, 640, 640]`, RGB, pixel values `[0,1]`. If mean frame brightness < `BRIGHTNESS_THRESHOLD` (default 50/255), apply low-light transform (grayscale → 3-channel + CLAHE). On Jetson this also triggers GPIO IR floodlight. Call this "low-light adaptive preprocessing" — not "sensor fusion".

**ONNX output shape and postprocessor:** Raw output is `[1, 6, 8400]`. Postprocessor **must transpose to `[8400, 6]`** before iterating. Columns after transpose: `[cx, cy, w, h, bowl_empty_score, bowl_not_empty_score]`. There is no single confidence column — compute `confidence = max(col[4], col[5])`, `class_id = argmax(col[4], col[5])`. NMS: IoU threshold 0.45, confidence threshold 0.50. Bowl identity by x-coordinate: left = `bowl_1`, right = `bowl_2`.

**Bowl registration warm-up:** Both bowls must be detected simultaneously in ≥ 3 of the first 30 processed frames before debounce timers arm. Until registered, `/status` returns `"registration_pending": true` and no alerts fire. This prevents x-coordinate misassignment at startup.

**Debounce logic (per bowl):** Alert fires when `state == "empty"` AND `(now_ms − empty_since_ms) >= 60000` AND `(now_ms − last_alert_ms) >= 300000`. All state is in-memory — timers reset on restart (acceptable for MVP). If a bowl goes undetected, its state is held for up to `DETECTION_HOLD_FRAMES` (default 5) frames before being marked `undetected`.

**HTTP service:** `cpp-httplib` (header-only, no external deps), thread-pool size 4. `latest_frame` shared between inference loop and HTTP thread via `std::mutex`; `/photo` acquires lock, clones frame, releases lock, then JPEG-encodes outside the lock.

**Telegram notifier:** `POST /sendPhoto` multipart via libcurl (or Python subprocess). Retries 3× with 2 s backoff. Runs in a dedicated thread — never blocks inference.

**ONNX ↔ TensorRT swap:** Abstract `InferenceBackend` interface with `OnnxBackend` and `TrtBackend`. Selected via `INFERENCE_BACKEND=onnx|tensorrt` env var. `TrtBackend` is compiled only with `-DWITH_TENSORRT=ON`; laptop builds always use `-DWITH_TENSORRT=OFF`.

## Environment Variables

All runtime config via env vars (see `demo/.env.example` for full list):

| Variable | Default | Notes |
|---|---|---|
| `TELEGRAM_BOT_TOKEN` | — | Required |
| `TELEGRAM_CHAT_ID` | — | Required |
| `INFERENCE_BACKEND` | `onnx` | `onnx` or `tensorrt` |
| `MODEL_PATH` | — | Path to `.onnx` or `.engine` |
| `DEBOUNCE_SECONDS` | `60` | Demo sets this to `8` |
| `ALERT_COOLDOWN_SECONDS` | `300` | Per-bowl cooldown |
| `CONFIDENCE_THRESHOLD` | `0.50` | Tunable, never hardcode |
| `IOU_THRESHOLD` | `0.45` | NMS threshold |
| `BRIGHTNESS_THRESHOLD` | `50` | 0–255; triggers low-light path |
| `HTTP_PORT` | `8080` | |

## Key Constraints

- **Laptop-first:** Every component must run on macOS/Ubuntu 22.04 CPU with no GPU until the Jetson arrives. No CUDA dependency in laptop builds.
- **No manual ROI:** Bowl detection is fully automatic via YOLOv8n — never add hardcoded region-of-interest parameters.
- **MVP scope only:** Do not implement multi-camera, historical logging, web dashboard, cat identification, or audio alerts. These are explicitly deferred (see `docs/DESIGN_REQUIREMENTS.md §8`).
- **spdlog** (≥ 1.12, header-only) is required for all C++ service logging. **libcurl** (≥ 7.68) is required for the Telegram HTTP client.
- **ONNX/TRT parity test acceptance criteria:** On a 50-image held-out val set — IoU between matched detections > 0.90, class agreement > 98%, mAP50 delta ≤ 3 pp.
- Input resolution is 640×640 by default; switch to 416×416 if 30 FPS is required on Jetson (TRT FP16 at 640×640 measures 12–18 FPS on Nano 4GB).
