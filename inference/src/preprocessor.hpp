#pragma once
#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

namespace cbw {

// Mirrors training/augmentations.py:
//   - If mean frame brightness < brightness_threshold: grayscale → CLAHE → 3-channel
//   - Resize to 640×640, BGR→RGB, float32 [0,1], CHW layout
// Defaults MUST stay in sync with augmentations.py (BRIGHTNESS_THRESHOLD=50,
// CLAHE_CLIP_LIMIT=2.0, CLAHE_TILE_GRID=(8,8)) to keep train/inference distributions matched.
class Preprocessor {
public:
    Preprocessor(int   brightness_threshold = 50,
                 float clahe_clip_limit     = 2.0f,
                 int   clahe_tile_grid      = 8);

    // Returns float32 tensor: [3 * 640 * 640], CHW, RGB, [0, 1]
    std::vector<float> process(const cv::Mat& bgr_frame);

    bool last_was_low_light() const { return last_low_light_; }

private:
    int                brightness_threshold_;
    cv::Ptr<cv::CLAHE> clahe_;
    bool               last_low_light_ = false;

    static constexpr int INPUT_W = 640;
    static constexpr int INPUT_H = 640;
};

} // namespace cbw
