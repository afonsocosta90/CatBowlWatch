#pragma once
#include <string>
#include <cstdlib>
#include <cstdint>

namespace cbw {

struct Config {
    // Capture
    std::string video_source          = "0";       // VIDEO_SOURCE (path or webcam index)

    // Inference
    std::string model_path;                        // MODEL_PATH (required)
    float       confidence_threshold  = 0.50f;     // CONFIDENCE_THRESHOLD
    float       iou_threshold         = 0.45f;     // IOU_THRESHOLD

    // Preprocessing — must mirror training/augmentations.py defaults
    int         brightness_threshold  = 50;        // BRIGHTNESS_THRESHOLD (0-255)
    float       clahe_clip_limit      = 2.0f;      // CLAHE_CLIP_LIMIT
    int         clahe_tile_grid       = 8;         // CLAHE_TILE_GRID (NxN)

    // Debounce
    int64_t     debounce_ms           = 60000;     // DEBOUNCE_SECONDS * 1000
    int64_t     cooldown_ms           = 300000;    // ALERT_COOLDOWN_SECONDS * 1000

    // Tracker
    int         detection_hold_frames = 5;         // DETECTION_HOLD_FRAMES

    // HTTP service
    int         http_port             = 8080;      // HTTP_PORT

    static Config from_env() {
        Config c;
        auto get = [](const char* key) -> const char* { return std::getenv(key); };
        if (auto* v = get("VIDEO_SOURCE"))             c.video_source            = v;
        if (auto* v = get("MODEL_PATH"))               c.model_path              = v;
        if (auto* v = get("CONFIDENCE_THRESHOLD"))     c.confidence_threshold    = std::stof(v);
        if (auto* v = get("IOU_THRESHOLD"))            c.iou_threshold           = std::stof(v);
        if (auto* v = get("BRIGHTNESS_THRESHOLD"))     c.brightness_threshold    = std::stoi(v);
        if (auto* v = get("CLAHE_CLIP_LIMIT"))         c.clahe_clip_limit        = std::stof(v);
        if (auto* v = get("CLAHE_TILE_GRID"))          c.clahe_tile_grid         = std::stoi(v);
        if (auto* v = get("DEBOUNCE_SECONDS"))         c.debounce_ms             = std::stoll(v) * 1000;
        if (auto* v = get("ALERT_COOLDOWN_SECONDS"))   c.cooldown_ms             = std::stoll(v) * 1000;
        if (auto* v = get("DETECTION_HOLD_FRAMES"))    c.detection_hold_frames   = std::stoi(v);
        if (auto* v = get("HTTP_PORT"))                c.http_port               = std::stoi(v);
        return c;
    }
};

} // namespace cbw
