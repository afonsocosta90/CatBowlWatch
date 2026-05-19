#pragma once
#include "types.hpp"
#include <vector>

namespace cbw {

// Converts raw ONNX output [1,6,8400] → filtered, NMS-deduped detections.
// See ARCHITECTURE.md §4.3 for layout spec.
class Postprocessor {
public:
    Postprocessor(float conf_threshold = 0.50f, float iou_threshold = 0.45f);

    // raw_output: flat float32 from OnnxBackend::run(), 1*6*8400 values
    std::vector<Detection> process(const std::vector<float>& raw_output) const;

private:
    float conf_threshold_;
    float iou_threshold_;

    static float iou(const Detection& a, const Detection& b);
    std::vector<Detection> nms(std::vector<Detection> dets) const;
};

} // namespace cbw
