#include "bowl_tracker.hpp"
#include <algorithm>
#include <chrono>
#include <spdlog/spdlog.h>

namespace cbw {

BowlTracker::BowlTracker(int hold_frames) : hold_frames_(hold_frames) {
    states_[0].bowl_id = "bowl_1";
    states_[0].state   = "undetected";
    states_[1].bowl_id = "bowl_2";
    states_[1].state   = "undetected";
}

int64_t BowlTracker::now_ms() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::system_clock::now().time_since_epoch()).count();
}

std::array<BowlState, 2> BowlTracker::update(const std::vector<Detection>& detections) {
    ++processed_frames_;

    if (!registered_ && processed_frames_ <= REGISTRATION_WINDOW) {
        if (detections.size() >= 2 && ++dual_detection_count_ >= REGISTRATION_NEEDED) {
            registered_ = true;
            spdlog::info("BowlTracker: registered after {} frames", processed_frames_);
        }
    }

    // Sort left → right to assign bowl_1 / bowl_2 by x-coordinate
    std::vector<Detection> sorted = detections;
    std::sort(sorted.begin(), sorted.end(),
              [](const Detection& a, const Detection& b) { return a.cx < b.cx; });

    for (int slot = 0; slot < 2; ++slot) {
        if (static_cast<int>(sorted.size()) > slot) {
            const Detection& d = sorted[slot];
            hold_counters_[slot] = hold_frames_;
            states_[slot].confidence = d.confidence;
            states_[slot].bbox = cv::Rect2f(d.cx - d.w / 2.f, d.cy - d.h / 2.f, d.w, d.h);

            if (d.class_id == CLASS_BOWL_EMPTY) {
                if (states_[slot].state != "empty") {
                    states_[slot].empty_since_ms = now_ms();
                    states_[slot].state = "empty";
                }
            } else {
                states_[slot].state          = "not_empty";
                states_[slot].empty_since_ms = -1;
            }
        } else {
            if (hold_counters_[slot] > 0) {
                --hold_counters_[slot];
            } else {
                states_[slot].state          = "undetected";
                states_[slot].empty_since_ms = -1;
                states_[slot].confidence     = 0.f;
            }
        }
    }

    return states_;
}

} // namespace cbw
