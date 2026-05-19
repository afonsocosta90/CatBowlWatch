#include "postprocessor.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace cbw {

Postprocessor::Postprocessor(float conf_threshold, float iou_threshold)
    : conf_threshold_(conf_threshold), iou_threshold_(iou_threshold) {}

std::vector<Detection> Postprocessor::process(const std::vector<float>& raw_output) const {
    // Raw layout: [1, 6, 8400] — transpose to [8400, 6]
    // Columns after transpose: [cx, cy, w, h, bowl_empty_score, bowl_not_empty_score]
    // cx/cy/w/h are in pixel coordinates relative to 640×640 input.
    constexpr int N = 8400;
    constexpr int C = 6;

    if (static_cast<int>(raw_output.size()) != 1 * C * N)
        throw std::runtime_error("Postprocessor::process: unexpected output size");

    std::vector<Detection> candidates;
    candidates.reserve(64);

    for (int i = 0; i < N; ++i) {
        float s0 = raw_output[4 * N + i];  // bowl_empty_score
        float s1 = raw_output[5 * N + i];  // bowl_not_empty_score
        float conf = std::max(s0, s1);
        if (conf < conf_threshold_) continue;

        Detection d;
        d.cx         = raw_output[0 * N + i] / 640.f;
        d.cy         = raw_output[1 * N + i] / 640.f;
        d.w          = raw_output[2 * N + i] / 640.f;
        d.h          = raw_output[3 * N + i] / 640.f;
        d.confidence = conf;
        d.class_id   = (s1 > s0) ? CLASS_BOWL_NOT_EMPTY : CLASS_BOWL_EMPTY;
        candidates.push_back(d);
    }

    return nms(std::move(candidates));
}

float Postprocessor::iou(const Detection& a, const Detection& b) {
    float ax1 = a.cx - a.w / 2, ay1 = a.cy - a.h / 2;
    float ax2 = a.cx + a.w / 2, ay2 = a.cy + a.h / 2;
    float bx1 = b.cx - b.w / 2, by1 = b.cy - b.h / 2;
    float bx2 = b.cx + b.w / 2, by2 = b.cy + b.h / 2;

    float iw = std::max(0.f, std::min(ax2, bx2) - std::max(ax1, bx1));
    float ih = std::max(0.f, std::min(ay2, by2) - std::max(ay1, by1));
    float inter = iw * ih;
    return inter / (a.w * a.h + b.w * b.h - inter + 1e-7f);
}

std::vector<Detection> Postprocessor::nms(std::vector<Detection> dets) const {
    std::sort(dets.begin(), dets.end(),
              [](const Detection& a, const Detection& b) {
                  return a.confidence > b.confidence;
              });

    std::vector<bool>      suppressed(dets.size(), false);
    std::vector<Detection> result;
    result.reserve(4);

    for (size_t i = 0; i < dets.size(); ++i) {
        if (suppressed[i]) continue;
        result.push_back(dets[i]);
        for (size_t j = i + 1; j < dets.size(); ++j) {
            if (!suppressed[j] && iou(dets[i], dets[j]) > iou_threshold_)
                suppressed[j] = true;
        }
    }
    return result;
}

} // namespace cbw
