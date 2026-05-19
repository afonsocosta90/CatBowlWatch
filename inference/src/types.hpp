#pragma once
#include <string>
#include <cstdint>
#include <opencv2/core.hpp>

namespace cbw {

constexpr int CLASS_BOWL_EMPTY     = 0;
constexpr int CLASS_BOWL_NOT_EMPTY = 1;

struct Detection {
    float cx, cy, w, h;  // normalised [0,1] centre + size relative to 640x640
    float confidence;
    int   class_id;       // CLASS_BOWL_EMPTY | CLASS_BOWL_NOT_EMPTY
};

struct BowlState {
    std::string bowl_id;           // "bowl_1" | "bowl_2"
    std::string state;             // "empty" | "not_empty" | "undetected"
    float       confidence  = 0.f;
    int64_t     empty_since_ms = -1; // epoch ms when became empty; -1 if not
    cv::Rect2f  bbox;
};

struct AlertEvent {
    std::string bowl_id;
    int64_t     timestamp_ms;
};

} // namespace cbw
