# CatBowlWatch — Design Requirements

> **Status:** Phase 0 — Documentation. No code written yet.
> **Last updated:** 2026-05-13

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
- `docker compose up` loops `data/videos/sample_video.mp4`, fires a real Telegram alert, requires no Jetson hardware.

---

## 3. Non-Functional Requirements

| ID | Requirement | Target |
|---|---|---|
| NFR-1 | Inference latency (ONNX CPU, laptop) | ≤ 200 ms / frame |
| NFR-2 | Inference latency (TensorRT FP16, Jetson) | ≤ 33 ms / frame (≥ 30 FPS) |
| NFR-3 | False-positive alert rate | < 1 per day under normal lighting |
| NFR-4 | Service memory footprint (Jetson) | ≤ 512 MB RSS |
| NFR-5 | Cold-start time (service ready, Jetson) | ≤ 10 s |
| NFR-6 | Uptime (systemd watchdog, Jetson) | ≥ 99.5% over 30-day window |
| NFR-7 | Dev loop: train → ONNX export → inference test | ≤ 5 min end-to-end |

---

## 4. Hardware & Environment Constraints

| Item | Specification |
|---|---|
| Target SBC | NVIDIA Jetson Nano 4GB (JetPack 5.x) |
| Camera | Raspberry Pi CSI IMX219 |
| Inference runtime (prod) | TensorRT 8.x, FP16 |
| Inference runtime (dev) | ONNX Runtime ≥ 1.17, CPU |
| IR illumination | IR floodlight, GPIO-triggered at low ambient brightness |
| Network | Wi-Fi or Ethernet; Telegram API reachable |
| Power | 5V/4A barrel jack |

**Laptop-first constraint:** Until the Jetson arrives, every component — inference, notification, demo — must run on a development laptop (macOS or Ubuntu 22.04) with no GPU required.

---

## 5. Model Constraints

| Decision | Choice | Rationale |
|---|---|---|
| Architecture | YOLOv8n | Smallest YOLOv8 variant; fits Jetson Nano 4GB at FP16; TensorRT-friendly |
| Export format (dev) | ONNX opset 17 | Portable, CPU-runnable, no NVIDIA dependency during development |
| Export format (prod) | TensorRT engine (FP16) | 3–5× speedup over ONNX on Jetson; required to meet NFR-2 |
| Classes | `bowl_empty`, `bowl_not_empty` | Detection + classification in one pass; no separate classifier network |
| Input resolution | 640 × 640 | YOLOv8n default; reducible to 416 × 416 if Jetson latency is tight |
| Training data (Phase 1) | Non-representative | Laptop-first; retrained with real data at Phase 5 |

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

## 7. Low-Light Handling

| Mode | Trigger | Action |
|---|---|---|
| Normal | Ambient brightness > threshold | Standard inference |
| Low-light (laptop sim) | Mean frame brightness < threshold | Grayscale + contrast stretch (IR simulation) |
| Low-light (Jetson prod) | Brightness heuristic | GPIO IR floodlight ON + same transform |

The brightness threshold and IR sim parameters are matched to the training-time augmentation pipeline so inference and training see the same distribution.

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

| Phase | Description | Entry Criteria | Exit Criteria |
|---|---|---|---|
| 0 | Documentation | Project kick-off | README + DESIGN_REQUIREMENTS + ARCHITECTURE complete |
| 1 | Data Collection | Phase 0 done | ≥ 200 labelled images (YOLO format), sample_video.mp4 in repo |
| 2 | Training Pipeline | Phase 1 done | YOLOv8n trains to mAP50 ≥ 0.80; ONNX export passes validation |
| 3 | Inference Service | Phase 2 done | C++ service processes video, fires debounce, /status + /photo respond |
| 4 | Notification | Phase 3 done | Telegram alert delivered end-to-end from video input on laptop |
| 5 | Hardware Deployment | Jetson in hand | ≥ 30 FPS via TensorRT; systemd survives reboot |
