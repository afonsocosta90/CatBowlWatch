#pragma once
#include <string>
#include <opencv2/videoio.hpp>

namespace cbw {

class Capture {
public:
    // source: file path, or "0" / "1" / ... for webcam index
    explicit Capture(const std::string& source);

    // Returns false only on unrecoverable error; loops video files automatically
    bool read(cv::Mat& frame);

    bool is_open() const;

private:
    cv::VideoCapture cap_;
    std::string      source_;
    bool             is_file_;
};

} // namespace cbw
