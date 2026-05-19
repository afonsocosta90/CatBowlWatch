#include "preprocessor.hpp"
#include <stdexcept>

namespace cbw {

Preprocessor::Preprocessor(int brightness_threshold, float clahe_clip_limit, int clahe_tile_grid)
    : brightness_threshold_(brightness_threshold) {
    clahe_ = cv::createCLAHE(clahe_clip_limit, cv::Size(clahe_tile_grid, clahe_tile_grid));
}

std::vector<float> Preprocessor::process(const cv::Mat& bgr_frame) {
    if (bgr_frame.empty())
        throw std::runtime_error("Preprocessor::process: empty frame");

    cv::Mat resized;
    cv::resize(bgr_frame, resized, cv::Size(INPUT_W, INPUT_H));

    cv::Mat gray;
    cv::cvtColor(resized, gray, cv::COLOR_BGR2GRAY);
    float mean_brightness = static_cast<float>(cv::mean(gray)[0]);
    last_low_light_ = mean_brightness < static_cast<float>(brightness_threshold_);

    cv::Mat processed;
    if (last_low_light_) {
        cv::Mat enhanced;
        clahe_->apply(gray, enhanced);
        cv::cvtColor(enhanced, processed, cv::COLOR_GRAY2BGR);
    } else {
        processed = resized;
    }

    cv::Mat rgb;
    cv::cvtColor(processed, rgb, cv::COLOR_BGR2RGB);
    rgb.convertTo(rgb, CV_32F, 1.0 / 255.0);

    // HWC → CHW
    std::vector<cv::Mat> channels(3);
    cv::split(rgb, channels);

    std::vector<float> tensor;
    tensor.reserve(3 * INPUT_H * INPUT_W);
    for (const auto& ch : channels) {
        tensor.insert(tensor.end(),
                      reinterpret_cast<const float*>(ch.datastart),
                      reinterpret_cast<const float*>(ch.dataend));
    }
    return tensor;
}

} // namespace cbw
