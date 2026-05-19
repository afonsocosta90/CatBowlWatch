# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Status

**Phase 1 — Data Collection. Plumbing done (1a); data capture & labelling in progress (1b).** Phases 1–5 are planned sequentially: do not implement features from a later phase when working in an earlier one. Phase boundaries are defined in `docs/DESIGN_REQUIREMENTS.md §9`.

**What exists today (Phase 1a — done):**
- Poetry env: `pyproject.toml` (base: `opencv-python`, `numpy`, `pytest`; optional `training` group: `torch`, `torchvision`, `ultralytics`).
- `Makefile` — `make data` (organise → validate → split), plus `collect` / `validate` / `split` / `clean-data` / `test`. `PYTHON ?= poetry run python`.
- `scripts/collect_data.py` — frame sampler.
- `scripts/organise_raw.py` — flat → `images/`+`labels/` by stem, idempotent.
- `scripts/validate_labels.py` — YOLO label sanity checks, UTF-8 BOM tolerated.
- `scripts/split_dataset.py` — seeded 70/15/15 split, writes `data/data.yaml`.
- `scripts/_generate_synthetic.py` — synthetic image+label generator for pipeline dry-runs (no real data required).
- `training/dataset.py` — `BowlDataset` (used in Phase 2).
- `tests/` — 21 unit tests covering organise / validate / split / dataset.
- **Pipeline dry-run validated end-to-end** with synthetic fixtures (organise → validate → split → `data.yaml`). Ready to receive real labelled images.

**What's left for Phase 1b:** capture iPhone footage, label in Roboflow, unzip to `data/raw/labelled/`, run `make data`. Exit at ≥ 200 labelled images + `data/videos/sample_video.mp4` committed.

**Phase 2 — in progress (started early, in parallel with 1b's data gathering):**
- `training/augmentations.py` ✓ — low-light adaptive preprocessing (grayscale + CLAHE). Public API: `mean_brightness`, `low_light_transform`, `adaptive_low_light`, `LowLightAugmenter` (callable for `BowlDataset.transform`). Defaults (`BRIGHTNESS_THRESHOLD=50`, `CLAHE_CLIP_LIMIT=2.0`, `CLAHE_TILE_GRID=(8,8)`) MUST be mirrored by the C++ inference preprocessor in Phase 3 to keep train/inference distributions matched.
- `training/train.py` ✓ — Ultralytics `YOLO.train` wrapper. CLI args via `parse_args`/`TrainConfig` dataclass. Copies `best.pt` to `models/catbowlwatch.pt` after training. `make train`. Lazy `ultralytics` import — module is importable without the training group installed.
- `training/export.py` ✓ — `.pt` → ONNX opset 17 with **shape verification** against the `EXPECTED_OUTPUT_SHAPE = (1, 6, 8400)` contract from ARCHITECTURE.md §4.3. Fails loudly if the C++ postprocessor would receive an unexpected layout. `make export-onnx`. Lazy `ultralytics` + `onnxruntime` imports.
- `tests/test_augmentations.py` ✓ (11), `tests/test_train.py` ✓ (5), `tests/test_export.py` ✓ (4) — 20 new tests. Full suite: 41 passed, 1 skipped (torch dep).
- `onnxruntime` added to the `training` Poetry group for export-time shape verification.
- ☐ ONNX/TRT parity tests — deferred until a real `.onnx` exists (needs Phase 1b data + a training run).

**Phase 2 runnable end-to-end** requires `poetry install --with training` AND a populated `data/data.yaml` (so ultimately still gated on Phase 1b's ≥ 200 labelled images).

**Phase 3 — toolchain scaffold (started early in WSL2 Ubuntu):**
- `scripts/setup_wsl_dev.sh` ✓ — apt-installs build-essential / cmake / pkg-config / libopencv-dev / libcurl4-openssl-dev; downloads ONNX Runtime C++ 1.20.1 release tarball to `$HOME/onnxruntime-linux-x64-<ver>/`. TensorRT intentionally not installed (laptop default is `WITH_TENSORRT=OFF`).
- `inference/CMakeLists.txt` ✓ — top-level CMake. `WITH_TENSORRT=OFF` default; `find_package(OpenCV REQUIRED)`; ONNX Runtime imported via `ONNXRUNTIME_ROOT` env or `-D` arg; spdlog (v1.14.1), cpp-httplib (v0.18.1), GoogleTest (v1.15.2) vendored via FetchContent.
- `inference/src/main.cpp` ✓ — smoke binary `catbowlwatch_smoke` that prints OpenCV / spdlog / ONNX Runtime versions and instantiates an `httplib::Server` to confirm linkage. This is a toolchain validator only; will be replaced by the real service entrypoint once components land.
- `inference/tests/test_smoke.cpp` ✓ — GoogleTest smoke (`Smoke.ToolchainOk`) to confirm `ctest --output-on-failure` works.
- ☐ Real components — Capture, Preprocessor, OnnxBackend, Postprocessor, BowlTracker, DebounceEngine, HttpServer. All gated on the WSL toolchain validating end-to-end first.

**Phase 3 build sequence** (inside WSL Ubuntu):
```bash
bash scripts/setup_wsl_dev.sh
export ONNXRUNTIME_ROOT=~/onnxruntime-linux-x64-1.20.1
cd inference
cmake -B build -DCMAKE_BUILD_TYPE=Release -DWITH_TENSORRT=OFF
cmake --build build -j$(nproc)
ctest --test-dir build --output-on-failure
./build/src/catbowlwatch_smoke   # should print versions and "OK"
```

Canonical specs (do not duplicate — link, then read):
- `docs/DESIGN_REQUIREMENTS.md` — functional/non-functional requirements, Telegram setup, MVP scope.
- `docs/ARCHITECTURE.md` — component contracts, data flow, swap plan.

## Commands

Most commands below describe planned scripts and Docker setup that don't exist yet. Update this section as files land. Phase 1 dataset pipeline is `make`-driven (see below).

```bash
# Phase 1 — dataset pipeline (works today)
poetry install                  # one-time; adds opencv-python, numpy, pytest
make data                       # organise → validate → split (70/15/15, seed 42)
make collect                    # sample frames from videos in data/raw/incoming/
make test                       # run pytest

# Sample frames directly (also runs as `make collect`)
poetry run python scripts/collect_data.py --source data/videos/sample_video.mp4 --interval 1.0
poetry run python scripts/collect_data.py --source 0 --interval 0.5  # webcam

# Phase 2 deps (when starting training work)
poetry install --with training  # adds torch, torchvision, ultralytics

# Demo (Docker — no Jetson needed)  [planned]
cp demo/.env.example .env   # fill in TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID
docker compose -f docker/demo.yml up

# C++ inference service — laptop (ONNX)  [planned]
mkdir -p inference/build && cd inference/build
cmake -DWITH_TENSORRT=OFF ..
make -j$(nproc)

# C++ inference service — Jetson (TensorRT)  [planned]
cmake -DWITH_TENSORRT=ON ..
make -j$(nproc)

# Python training pipeline  [planned]
cd training
python train.py          # trains YOLOv8n, saves .pt to models/
python export.py         # exports .pt → ONNX opset 17

# Verify ONNX output shape (run before writing C++ postprocessor)
python -c "import onnxruntime as ort, numpy as np; s=ort.InferenceSession('models/catbowlwatch.onnx'); print(s.run(None,{s.get_inputs()[0].name:np.zeros((1,3,640,640),dtype='f4')})[0].shape)"

# Tests  [planned]
cd tests && python -m pytest           # ONNX/TRT parity + debounce unit tests
```

## Architecture

Single-device edge pipeline: camera → YOLOv8n → C++17 service → Telegram. No cloud component except the outbound Telegram Bot API call.

**Per-frame pipeline:**
```
Capture → Preprocessor → YOLOv8n → Postprocessor → Bowl Tracker → Debounce → Telegram
                                                          │
                                                          └→ HTTP service (/status, /photo)
```

**Capture:** `cv::VideoCapture` on laptop; GStreamer `nvarguscamerasrc` on Jetson (Phase 5 swap — interface identical, single `read(frame)` call).

**Preprocessor:** outputs `float32 [1,3,640,640]`, RGB, `[0,1]`. If mean frame brightness < `BRIGHTNESS_THRESHOLD` (default 50/255), apply low-light transform (grayscale → 3-channel + CLAHE). On Jetson this also triggers a GPIO IR floodlight. The same transform parameters are used in training-time augmentation so train and inference see the same distribution.

**ONNX output and postprocessor:** Raw output is `[1, 6, 8400]`. Postprocessor **must transpose to `[8400, 6]`** before iterating. Columns after transpose: `[cx, cy, w, h, bowl_empty_score, bowl_not_empty_score]`. There is no single confidence column — compute `confidence = max(col[4], col[5])`, `class_id = argmax(col[4], col[5])`. NMS: IoU 0.45, conf 0.50. **Class map:** `0 = bowl_empty`, `1 = bowl_not_empty`. **Bowl identity** by x-coordinate: left = `bowl_1`, right = `bowl_2` (camera is fixed overhead, so x-ordering is stable).

**Bowl registration warm-up:** Both bowls must be detected simultaneously in ≥ 3 of the first 30 processed frames before debounce timers arm. Until then `/status` returns `"registration_pending": true` and no alerts fire. Prevents x-coordinate misassignment at startup.

**Debounce (per bowl):** Alert fires when `state == "empty"` AND `(now_ms − empty_since_ms) >= 60000` AND `(now_ms − last_alert_ms) >= 300000`. All state in-memory — timers reset on restart (acceptable for MVP). If a bowl goes undetected its state is held for up to `DETECTION_HOLD_FRAMES` (default 5) frames before being marked `undetected`.

**HTTP service:** `cpp-httplib` (header-only), thread-pool size 4. `latest_frame` shared between inference loop and HTTP thread via `std::mutex`. `/photo` acquires the lock, clones the frame, releases the lock, then JPEG-encodes outside the lock.

**Telegram notifier:** `POST /sendPhoto` multipart via libcurl. 3× retry with 2 s backoff. Dedicated thread — never blocks inference.

**ONNX ↔ TensorRT swap:** Abstract `InferenceBackend` interface with `OnnxBackend` and `TrtBackend`. Selected via `INFERENCE_BACKEND=onnx|tensorrt`. `TrtBackend` is compiled only with `-DWITH_TENSORRT=ON`; laptop builds always use `-DWITH_TENSORRT=OFF` (NvInfer.h is unavailable without CUDA).

## Environment Variables

All runtime config via env vars (see `demo/.env.example` for full list):

| Variable | Default | Notes |
|---|---|---|
| `TELEGRAM_BOT_TOKEN` | — | Required |
| `TELEGRAM_CHAT_ID` | — | Required |
| `INFERENCE_BACKEND` | `onnx` | `onnx` or `tensorrt` |
| `MODEL_PATH` | — | Path to `.onnx` or `.engine` |
| `DEBOUNCE_SECONDS` | `60` | Demo overrides to `8` |
| `ALERT_COOLDOWN_SECONDS` | `300` | Per-bowl cooldown |
| `CONFIDENCE_THRESHOLD` | `0.50` | Tunable, never hardcode |
| `IOU_THRESHOLD` | `0.45` | NMS threshold |
| `BRIGHTNESS_THRESHOLD` | `50` | 0–255; triggers low-light path |
| `HTTP_PORT` | `8080` | |

## Key Constraints

- **Laptop-first:** Every component must run on macOS or Ubuntu 22.04 CPU with no GPU until the Jetson arrives. No CUDA dependency in laptop builds.
- **No manual ROI:** Bowl detection is fully automatic via YOLOv8n — never add hardcoded region-of-interest parameters.
- **MVP scope only:** Do not implement multi-camera, historical logging, web dashboard, cat identification, or audio alerts. Deferred list in `docs/DESIGN_REQUIREMENTS.md §8`.
- **Required libraries:** `spdlog` ≥ 1.12 (header-only) for all C++ service logging; `libcurl` ≥ 7.68 for the Telegram client.
- **ONNX/TRT parity acceptance:** On a 50-image held-out val set — matched-detection IoU > 0.90, class agreement > 98%, mAP50 delta ≤ 3 pp.
- **Input resolution:** 640×640 by default. Switch to 416×416 only if 30 FPS is required on Jetson (TRT FP16 at 640×640 measures 12–18 FPS on Nano 4GB; a stationary bowl rarely needs more).
- **Terminology:** The brightness-triggered preprocessing path is "low-light adaptive preprocessing" — **not "sensor fusion"**. There is only one sensor.
- **Config, not constants:** Thresholds (`CONFIDENCE_THRESHOLD`, `IOU_THRESHOLD`, `DEBOUNCE_SECONDS`, etc.) are env-driven. Never hardcode them in source.

## Dataset Pipeline (Phase 1)

iPhone → Roboflow → `make data`. Drop raw videos in `data/raw/incoming/`, label in Roboflow (classes: `bowl_empty`=0, `bowl_not_empty`=1, no split on export), unpack into `data/raw/labelled/`, run `make data`. The pipeline produces `data/{images,labels}/{train,val,test}/` and `data/data.yaml`. Bowl identity (`bowl_1`/`bowl_2`) is **not** encoded in labels — it's assigned downstream by x-coordinate at the tracker layer. The model is N-bowl-clean; the 2-bowl assumption lives only in the tracker.
