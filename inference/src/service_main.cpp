// CatBowlWatch — Phase 3 real service entrypoint.
// Wires Capture → Preprocessor → OnnxBackend → Postprocessor →
//        BowlTracker → DebounceEngine → HttpServer.
// All thresholds are env-driven via Config::from_env(); see CLAUDE.md §Environment Variables.

#include "config.hpp"
#include "capture.hpp"
#include "preprocessor.hpp"
#include "inference_backend.hpp"
#include "postprocessor.hpp"
#include "bowl_tracker.hpp"
#include "debounce_engine.hpp"
#include "http_server.hpp"
#include "telegram_notifier.hpp"

#include <spdlog/spdlog.h>
#include <chrono>
#include <cstdlib>

int main() {
    auto config = cbw::Config::from_env();

    if (config.model_path.empty()) {
        spdlog::critical("MODEL_PATH env var not set — cannot start service");
        return 1;
    }

    spdlog::info("CatBowlWatch starting");
    spdlog::info("  source:    {}", config.video_source);
    spdlog::info("  model:     {}", config.model_path);
    spdlog::info("  conf_thr:  {:.2f}   iou_thr: {:.2f}", config.confidence_threshold, config.iou_threshold);
    spdlog::info("  debounce:  {}s   cooldown: {}s",
                 config.debounce_ms / 1000, config.cooldown_ms / 1000);
    spdlog::info("  http port: {}", config.http_port);
    spdlog::info("  telegram:  {}",
                 config.telegram_bot_token.empty() ? "not configured" : "configured");

    cbw::Capture           capture(config.video_source);
    cbw::Preprocessor      preprocessor(config.brightness_threshold,
                                        config.clahe_clip_limit,
                                        config.clahe_tile_grid);
    auto                   backend = cbw::InferenceBackend::create("onnx", config.model_path);
    cbw::Postprocessor     postprocessor(config.confidence_threshold, config.iou_threshold);
    cbw::BowlTracker       tracker(config.detection_hold_frames);
    cbw::DebounceEngine    debounce(config.debounce_ms, config.cooldown_ms);
    cbw::HttpServer        http_server(config.http_port);
    cbw::TelegramNotifier  notifier(config.telegram_bot_token, config.telegram_chat_id);

    http_server.start();

    auto     fps_start = std::chrono::steady_clock::now();
    int      fps_count = 0;
    double   fps       = 0.0;
    cv::Mat  frame;

    while (capture.read(frame)) {
        auto tensor      = preprocessor.process(frame);
        auto raw_output  = backend->run(tensor);
        auto detections  = postprocessor.process(raw_output);
        auto states      = tracker.update(detections);

        if (tracker.is_registered()) {
            for (const auto& alert : debounce.update(states)) {
                spdlog::warn("ALERT: {} empty for {}s — sending Telegram notification",
                             alert.bowl_id,
                             (alert.timestamp_ms - alert.empty_since_ms) / 1000);
                notifier.send(alert, frame);
            }
        }

        ++fps_count;
        auto now = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration<double>(now - fps_start).count();
        if (elapsed >= 1.0) {
            fps       = fps_count / elapsed;
            fps_count = 0;
            fps_start = now;
        }

        http_server.update(states, frame, !tracker.is_registered(), fps);
    }

    http_server.stop();
    return 0;
}
