#include "capture.hpp"
#include <spdlog/spdlog.h>
#include <stdexcept>
#include <algorithm>
#include <cctype>

namespace cbw {

Capture::Capture(const std::string& source) : source_(source) {
    bool is_index = !source.empty() &&
                    std::all_of(source.begin(), source.end(),
                                [](unsigned char c) { return std::isdigit(c); });
    is_file_ = !is_index;

    if (is_index) {
        cap_.open(std::stoi(source));
    } else {
        cap_.open(source);
    }

    if (!cap_.isOpened())
        throw std::runtime_error("Capture: cannot open source: " + source);

    spdlog::info("Capture opened: {}", source);
}

bool Capture::is_open() const { return cap_.isOpened(); }

bool Capture::read(cv::Mat& frame) {
    if (!cap_.read(frame)) {
        if (is_file_) {
            cap_.set(cv::CAP_PROP_POS_FRAMES, 0);
            return cap_.read(frame) && !frame.empty();
        }
        return false;
    }
    return !frame.empty();
}

} // namespace cbw
