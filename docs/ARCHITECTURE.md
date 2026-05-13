# CatBowlWatch — Architecture

> **Status:** Phase 0 — Documentation.
> **Last updated:** 2026-05-13

---

## 1. System Context

All inference, debounce logic, and notification run on a single device. No cloud component except the Telegram Bot API (outbound only).

```mermaid
C4Context
    title CatBowlWatch — System Context

    Person(owner, "Cat Owner", "Receives Telegram alerts on phone")
    System(cbw, "CatBowlWatch", "Edge ML service on Jetson Nano / Laptop")
    System_Ext(telegram, "Telegram Bot API", "Delivers photo + text notification")
    System_Ext(camera, "CSI IMX219 Camera", "Live video feed")

    Rel(camera, cbw, "Raw frames")
    Rel(cbw, telegram, "POST sendPhoto + sendMessage", "HTTPS")
    Rel(telegram, owner, "Push notification")
    Rel(owner, cbw, "GET /status, GET /photo", "HTTP LAN")
```

---

## 2. Component Architecture

```mermaid
flowchart TD
    subgraph Capture["Frame Capture Layer"]
        A1["GStreamer nvarguscamerasrc\n(Jetson — Phase 5)"]
        A2["OpenCV VideoCapture\n(Laptop — Phases 1-4)"]
    end

    subgraph Inference["Inference Layer"]
        B1["Preprocessor\nResize 640x640, normalize"]
        B2["YOLOv8n\nONNX Runtime CPU (dev)\nTensorRT FP16 (prod)"]
        B3["Postprocessor\nNMS, conf >= 0.5, class labels"]
    end

    subgraph State["State and Debounce Layer"]
        C1["Bowl State Tracker\nPer-bowl: state, timer, last_seen_ts"]
        C2["Debounce Engine\n60s continuous empty -> fire\n300s cooldown per bowl"]
    end

    subgraph Service["C++17 HTTP Service"]
        D1["cpp-httplib\nGET /status  GET /photo"]
    end

    subgraph Notify["Notification Layer"]
        E1["Telegram Notifier\nPOST sendPhoto + sendMessage"]
    end

    A1 --> B1
    A2 --> B1
    B1 --> B2 --> B3 --> C1 --> C2
    C2 -->|threshold crossed| E1
    C1 --> D1
    B1 -->|latest frame| D1
```

---

## 3. Data Flow — Single Frame

```mermaid
sequenceDiagram
    participant CAP as Frame Capture
    participant PRE as Preprocessor
    participant INF as YOLOv8n
    participant POST as Postprocessor
    participant TRK as Bowl Tracker
    participant DEB as Debounce Engine
    participant TG as Telegram Notifier

    CAP->>PRE: BGR frame (raw resolution)
    PRE->>INF: float32 tensor [1,3,640,640] normalized [0,1]
    INF->>POST: raw detections [N,6] (cx,cy,w,h,conf,class)
    POST->>TRK: [{bowl_id, state, bbox, conf}]
    TRK->>DEB: per-bowl state + elapsed_empty_s
    alt elapsed_empty_s >= 60 AND cooldown elapsed
        DEB->>TG: (bowl_id, snapshot_jpeg, timestamp)
        TG-->>DEB: HTTP 200
    end
```

---

## 4. Component Contracts

### 4.1 Frame Capture
- **Output:** Raw BGR `cv::Mat`, variable resolution.
- **Contract:** Loop on end-of-file (video mode). Consistent frame rate.
- **Swap point:** Replace `cv::VideoCapture` with GStreamer pipeline string at Phase 5. Interface is unchanged — single `read(frame)` call.

### 4.2 Preprocessor
- **Output:** `float32` tensor `[1, 3, 640, 640]`, RGB channel order, pixel values in `[0, 1]`.
- If mean frame brightness < `BRIGHTNESS_THRESHOLD` (default: 50/255): apply IR simulation transform (grayscale → 3-channel, contrast stretch).
- On Jetson: low-light condition also triggers GPIO IR floodlight.

### 4.3 YOLOv8n Inference
- **Output:** Raw ONNX output `[1, N, 6]` — columns are `[cx, cy, w, h, confidence, class_id]`.
- **Backend contract:** Abstract `InferenceBackend` interface with two concrete implementations:
  - `OnnxBackend` — wraps ONNX Runtime session (laptop, CPU).
  - `TrtBackend` — wraps TensorRT execution context (Jetson, FP16).
- Swap via config flag: `INFERENCE_BACKEND=onnx|tensorrt`. No C++ code change required.

### 4.4 Postprocessor
- NMS: IoU threshold 0.45, confidence threshold 0.50 (both configurable).
- Class map: `bowl_empty=0`, `bowl_not_empty=1`.
- Bowl identity assigned by x-coordinate: left bowl = `bowl_1`, right bowl = `bowl_2`. Deterministic for a fixed overhead camera.

### 4.5 Bowl State Tracker

```cpp
struct BowlState {
    std::string bowl_id;          // "bowl_1" | "bowl_2"
    std::string state;            // "empty" | "not_empty" | "undetected"
    float       confidence;
    int64_t     empty_since_ms;   // epoch ms when bowl became empty; -1 if not empty
    cv::Rect2f  bbox;
};
```

- If a bowl is not detected in a frame, state is held for up to `DETECTION_HOLD_FRAMES` (default: 5) frames before being marked `undetected`.

### 4.6 Debounce Engine

```
Per bowl:
  if state == "empty"
     AND (now_ms - empty_since_ms) >= 60000
     AND (now_ms - last_alert_ms)  >= 300000:
       fire alert
       last_alert_ms = now_ms
```

All state is in-memory. Timers reset on service restart — acceptable for MVP.

### 4.7 HTTP Service (`/status`, `/photo`)

```
GET /status  →  200 OK  application/json
{
  "bowls": [
    {"id": "bowl_1", "state": "empty",     "empty_for_s": 42, "confidence": 0.87},
    {"id": "bowl_2", "state": "not_empty", "empty_for_s": 0,  "confidence": 0.93}
  ],
  "fps": 14.2,
  "uptime_s": 3820
}

GET /photo  →  200 OK  image/jpeg  (latest annotated frame)
```

### 4.8 Telegram Notifier
- Sends `POST /sendPhoto` (multipart/form-data) with JPEG + caption.
- Retries up to 3× with 2 s backoff on network error.
- Runs in a dedicated thread — does not block the inference loop.

---

## 5. Backend Swap Plan (ONNX → TensorRT)

| Step | Action | Where |
|---|---|---|
| 1 | Export trained `.pt` to TensorRT `.engine` on Jetson | `scripts/export_trt.sh` |
| 2 | Set `INFERENCE_BACKEND=tensorrt` | `deployment/catbowlwatch.service` env block |
| 3 | Set `MODEL_PATH=/models/catbowlwatch.engine` | same |
| 4 | `systemctl restart catbowlwatch` | Jetson |

No C++ code changes. `TrtBackend` is compiled into the binary from day one.

---

## 6. Low-Light Architecture

```mermaid
flowchart LR
    FRAME["Raw frame"] --> BRIGHT{"mean brightness\n< threshold?"}
    BRIGHT -->|No| NORM["Normal inference path"]
    BRIGHT -->|Yes, laptop| IRSIM["IR Simulation\ngrayscale + contrast stretch"]
    BRIGHT -->|Yes, Jetson| GPIO["GPIO IR floodlight ON\n+ IR simulation transform"]
    IRSIM --> INF["Inference"]
    GPIO --> INF
    NORM --> INF
```

The brightness threshold and IR simulation parameters are matched to the training-time augmentation pipeline so inference and training see the same distribution.

---

## 7. Directory–Component Mapping

| Directory | Component(s) |
|---|---|
| `training/` | dataset.py, train.py, augmentations.py (IR sim), export.py |
| `inference/` | Capture, Preprocessor, OnnxBackend, TrtBackend, Postprocessor, BowlTracker, DebounceEngine, HTTP server |
| `notification/` | Telegram notifier |
| `deployment/` | GStreamer config, systemd unit, GPIO IR trigger, deploy.sh |
| `demo/` | docker-compose.yml, .env.example |
| `models/` | .pt, .onnx, .engine |
| `scripts/` | collect_data.py, train.sh, build.sh, export_trt.sh |
| `docker/` | Dockerfile.training, Dockerfile.demo |
| `tests/` | ONNX/TRT parity tests, debounce unit tests |
| `docs/` | DESIGN_REQUIREMENTS.md, ARCHITECTURE.md |

---

## 8. Key Architectural Decisions

| Decision | Chosen | Alternative | Rationale |
|---|---|---|---|
| Single YOLOv8n for detect+classify | Yes | Separate detector + classifier | One inference pass, simpler TRT export, fewer moving parts |
| Bowl identity by x-coordinate | Yes | SORT/DeepSORT tracker | Camera is fixed overhead; x-ordering is stable and deterministic |
| Debounce state in-memory | Yes | Redis / SQLite | Zero deps on Jetson; restart resets timers — acceptable for MVP |
| C++17 inference service | Yes | Python FastAPI | Portfolio goal; no GIL; clean TRT integration |
| ONNX on laptop / TRT on Jetson | Yes | TRT everywhere | TRT requires CUDA; laptop-first constraint demands ONNX for dev |
| cpp-httplib | Yes | Crow / Pistache | Header-only, zero external deps, two endpoints is all we need |
| Telegram | Yes | SMTP / Pushover | Free, phone-native push, photo attachment in one API call |

---

## 9. Future Extension Points (post-MVP, do not implement now)

- **Multi-camera:** Abstract capture layer; instantiate N capture + inference threads.
- **Historical data:** SQLite event log in debounce engine; expose `GET /history`.
- **Cat ID:** Second classification head or lightweight re-ID model.
- **Web dashboard:** WebSocket stream + minimal React frontend.
