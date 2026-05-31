# CatBowlWatch — Design Requirements

> **Status:** Phases 1a, 2, and 3 done. Phase 1b (data capture & labelling) in progress. Phase 4 (Notification + Demo) is next. Orin Nano hardware in hand.
> **Last updated:** 2026-05-31

---

## 1. Purpose

CatBowlWatch monitors two cat food bowls using a single overhead camera. It detects when either bowl has been empty for a sustained period and notifies the owner via Telegram — without any manual configuration after initial deployment.

---

## 2. Functional Requirements

### FR-1 — Bowl Detection
- The system **shall** detect both food bowls in every processed frame using YOLOv8n object detection.
- No manual region-of-interest (ROI) specification shall be required.
- Detection confidence threshold: **≥ 0.50** (tunable via config, not hardcoded).

### FR-2 — Bowl State Classification
- Each detected bowl **shall** be classified as `empty` or `not_empty`.
- Classification is produced by the same YOLOv8n model (two classes: `bowl_empty`, `bowl_not_empty`).
- If a bowl is not detected in a frame, its previous state is retained (no spurious resets).

### FR-3 — Debounce Logic
- A Telegram alert **shall** fire only after a bowl has been continuously classified as `empty` for **≥ 60 consecutive seconds**.
- The timer resets independently per bowl the moment a `not_empty` detection is observed for that bowl.
- Cooldown: **≥ 300 s** per bowl after each alert fires (prevents spam).

### FR-4 — Telegram Notification
- Alert **shall** include: (a) a JPEG snapshot at threshold-crossing, (b) text message with bowl ID, duration, timestamp.
- Delivery within **≤ 5 s** of threshold crossing (network permitting).
- Credentials via environment variables only — never hardcoded.

### FR-5 — HTTP Status Endpoint
- `GET /status` — JSON with per-bowl state and empty-timer value.
- `GET /photo` — latest JPEG frame.
- Default port: `8080` (configurable).

### FR-6 — Demo Mode
- `docker compose up` loops `data/videos/sample_video.mp4`, fires a real Telegram alert, requires no Orin Nano hardware.
- Demo mode overrides `DEBOUNCE_SECONDS=8` (set in `demo/.env.example`). Production default is 60 s. The `sample_video.mp4` must show a bowl empty from the first frame. Telegram notification arrives ~15 s after `docker compose up`.
- The demo debounce override is explicit and documented — not a hidden hack. Production deployments use the 60 s default via the systemd unit env block.

---

## 3. Non-Functional Requirements

| ID | Requirement | Target |
|---|---|---|
| NFR-1 | Inference latency (ONNX CPU, laptop) | ≤ 200 ms / frame |
| NFR-2 | Throughput (TensorRT FP16, Orin Nano) | ≥ 30 FPS @ 640×640. Orin Nano Ampere GPU significantly exceeds the old Nano 4GB baseline of 12–18 FPS. 416×416 fallback not required. Measured via `scripts/benchmark_inference.py`. |
| NFR-3 | False-positive alert rate | < 1 per day under normal lighting |
| NFR-4 | Service memory footprint (Orin Nano) | ≤ 512 MB RSS |
| NFR-5 | Cold-start time (service ready, Orin Nano) | ≤ 10 s |
| NFR-6 | Uptime (systemd watchdog, Orin Nano) | ≥ 99.5% over 30-day window |
| NFR-7 | Dev loop: train → ONNX export → inference test | ≤ 5 min end-to-end |
| NFR-8 | Inference benchmark coverage (Orin Nano) | All 4 backends benchmarked (`onnx-cpu`, `onnx-cuda`, `trt-fp16`, `trt-int8`); FPS + mean latency + P99 latency + RSS committed to `docs/BENCHMARKS.md` |

---

## 4. Hardware & Environment Constraints

| Item | Specification |
|---|---|
| Target SBC | NVIDIA Orin Nano (JetPack 6.x / CUDA 12.x / TensorRT 10.x / Ubuntu 22.04) |
| Training machine | Windows 11 PC, AMD GPU, ROCm PyTorch venv (not the Poetry `training` group). `train.py` and `export.py` run unchanged — ROCm exposes itself as `cuda` to PyTorch. ONNX is the hardware-agnostic handoff to the Orin Nano. |
| Camera | Raspberry Pi CSI IMX219 |
| Inference runtime (prod) | TensorRT 10.x, FP16 |
| Inference runtime (dev) | ONNX Runtime ≥ 1.17, CPU |
| IR illumination | IR floodlight, GPIO-triggered at low ambient brightness |
| Network | Wi-Fi or Ethernet; Telegram API reachable |
| Power | 5V/4A barrel jack |
| C++ logging | spdlog ≥ 1.12 (header-only, MIT) — required for structured service logging on headless Orin Nano |
| HTTP client (notification) | libcurl ≥ 7.68 — required for Telegram `POST /sendPhoto` multipart from C++ (or use Python subprocess — see ARCHITECTURE.md §4.8) |

**Laptop-first constraint (Phases 1–4):** All inference, notification, and demo components must run on a development laptop (macOS or Ubuntu 22.04) with no GPU required. Training is handled separately on the Windows AMD/ROCm machine; ONNX export is the hardware-agnostic handoff to the Orin Nano.

---

## 5. Model Constraints

| Decision | Choice | Rationale |
|---|---|---|
| Architecture | YOLOv8n | Smallest YOLOv8 variant; fits Orin Nano at TRT FP16; TensorRT-friendly |
| Export format (dev) | ONNX opset 17 | Portable, CPU-runnable, no NVIDIA dependency during development |
| Export format (prod) | TensorRT engine (FP16) | 3–5× speedup over ONNX on Orin Nano; required to meet NFR-2 |
| Classes | `bowl_empty`, `bowl_not_empty` | Detection + classification in one pass; no separate classifier network |
| Input resolution | 640 × 640 | YOLOv8n default; **change to 416 × 416 if 30 FPS is required** — see NFR-2 |
| Training data (Phase 1) | Representative of real deployment | Include photos of your actual bowls from day 1. Non-representative data produces a fake mAP score. Supplement with Roboflow Universe images but anchor the val set on your real setup. |

### ONNX/TRT Parity Tolerance

TRT FP16 introduces quantization error. The parity test in `tests/` must define an explicit tolerance — otherwise the test is meaningless.

**Acceptance criteria:** On a 50-image held-out validation set:
- Per-image IoU between ONNX and TRT matched detections: **> 0.90**
- Class agreement (same `class_id` for the same anchor): **> 98%**
- mAP50 delta (ONNX vs TRT): **≤ 3 pp** (e.g., ONNX 0.82 → TRT 0.79 is acceptable)

### Why YOLOv8n and not something smaller?

| Option | Pros | Cons |
|---|---|---|
| **YOLOv8n (chosen)** | Best tooling, clean TRT export, detect+classify in one pass | Slightly more compute than a pure classifier |
| MobileNet + custom head | Lighter on compute | Manual detection logic; ONNX export less stable; harder TRT path |
| YOLOv5n | Proven on Jetson | Older tooling, less ergonomic API |
| Custom tiny CNN | Minimal runtime cost | No pretrained backbone = far more data needed |

---

## 6. Notification Design

### Telegram Bot Setup (one-time)

1. Message `@BotFather` → `/newbot` → save the **bot token**.
2. Send any message to your bot, then query:
   ```
   https://api.telegram.org/bot<TOKEN>/getUpdates
   ```
   Extract `message.chat.id` — this is your **chat ID**.
3. Export credentials:
   ```bash
   export TELEGRAM_BOT_TOKEN="123456789:ABC-..."
   export TELEGRAM_CHAT_ID="987654321"
   ```

### Alert Payload Format

```
🐱 Bowl 1 is EMPTY
Empty for: 60 seconds
Time: 2026-05-13 14:32:07
[attached JPEG of the bowl]
```

### Anti-spam Cooldown

Per-bowl cooldown: 300 s after each alert. Configurable via `ALERT_COOLDOWN_SECONDS`.

---

## 7. Low-Light Adaptive Preprocessing

| Mode | Trigger | Action |
|---|---|---|
| Normal | Ambient brightness > threshold | Standard inference |
| Low-light (laptop) | Mean frame brightness < threshold | Grayscale + CLAHE (`cv::createCLAHE`) |
| Low-light (Orin Nano prod) | Brightness heuristic | GPIO IR floodlight ON + same CLAHE transform |

The brightness threshold and CLAHE parameters are matched to the training-time augmentation pipeline so inference and training see the same distribution.

**Terminology note:** This is a brightness-triggered preprocessing transform applied to a single camera feed — not sensor fusion. Do not use the word "fusion" when describing this feature in documentation or presentations.

---

## 8. Out of Scope (MVP)

- Multiple cameras or rooms
- Cloud storage / historical data logging
- Mobile app / web dashboard
- Automatic bowl refill tracking
- Cat identification (which cat ate)
- Audio alerts

These are explicitly deferred. Do not add them to the inference service.

---

## 9. Phase Tracking

| Phase | Description | Entry Criteria | Exit Criteria | Status |
|---|---|---|---|---|
| 0 | Documentation | Project kick-off | README + DESIGN_REQUIREMENTS + ARCHITECTURE complete | ✅ Done |
| 1a | Phase 1 plumbing | Phase 0 done | Poetry env (`pyproject.toml`); Makefile; `organise_raw.py` + `validate_labels.py` + `split_dataset.py`; ≥ 20 unit tests green | ✅ Done (2026-05-19) |
| 1b | Phase 1 data capture & labelling | Phase 1a done | ≥ 200 labelled images (YOLO format) in `data/{images,labels}/{train,val,test}/`; `data/data.yaml` committed; `data/videos/sample_video.mp4` committed | 🔄 In Progress |
| 2 | Training Pipeline | Phase 1b done | `training/train.py` trains YOLOv8n to `mAP50 ≥ 0.80`; `training/export.py` produces `catbowlwatch.onnx` (opset 17) with shape `[1, 6, 8400]`; ONNX-vs-runtime parity tests pass | ✅ Done (2026-05-19) — E2E run gated on Phase 1b data |
| 3 | Inference Service | Phase 2 done | C++17 service processes video, fires debounce, `/status` + `/photo` respond; ONNX backend on laptop | ✅ Done (2026-05-19) — needs `MODEL_PATH` for E2E |
| 4a | TelegramNotifier | Phase 3 done | C++ `TelegramNotifier` wired in `service_main.cpp`; libcurl `POST /sendPhoto`; 13 unit tests green | ✅ Done (2026-05-31) |
| 4b | Docker demo | Phase 4a done | `docker compose -f docker/demo.yml up` loops `sample_video.mp4`, fires real Telegram alert | 🔜 Next |
| 5 | Hardware Deployment & KPI Benchmark | Orin Nano in hand | ≥ 30 FPS TRT FP16 @ 640×640; all 4 backends benchmarked in `docs/BENCHMARKS.md`; systemd survives reboot | Entry gated on Phase 4 |
