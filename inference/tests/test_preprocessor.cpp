#include <gtest/gtest.h>
#include "preprocessor.hpp"
#include <opencv2/imgproc.hpp>

using cbw::Preprocessor;

TEST(Preprocessor, OutputLengthIs3x640x640) {
    Preprocessor pp;
    cv::Mat frame(480, 640, CV_8UC3, cv::Scalar(128, 128, 128));
    auto tensor = pp.process(frame);
    EXPECT_EQ(tensor.size(), static_cast<size_t>(3 * 640 * 640));
}

TEST(Preprocessor, ValuesInUnitRange) {
    Preprocessor pp;
    cv::Mat frame(480, 640, CV_8UC3, cv::Scalar(128, 100, 80));
    auto tensor = pp.process(frame);
    for (float v : tensor) {
        EXPECT_GE(v, 0.f);
        EXPECT_LE(v, 1.f);
    }
}

TEST(Preprocessor, DarkFrameTriggersLowLight) {
    Preprocessor pp(/*brightness_threshold=*/50);
    cv::Mat dark(480, 640, CV_8UC3, cv::Scalar(5, 5, 5));
    pp.process(dark);
    EXPECT_TRUE(pp.last_was_low_light());
}

TEST(Preprocessor, BrightFrameDoesNotTriggerLowLight) {
    Preprocessor pp(/*brightness_threshold=*/50);
    cv::Mat bright(480, 640, CV_8UC3, cv::Scalar(200, 200, 200));
    pp.process(bright);
    EXPECT_FALSE(pp.last_was_low_light());
}

TEST(Preprocessor, LowLightOutputStillInRange) {
    Preprocessor pp(/*brightness_threshold=*/200);  // high threshold → always low-light
    cv::Mat frame(480, 640, CV_8UC3, cv::Scalar(100, 100, 100));
    auto tensor = pp.process(frame);
    EXPECT_TRUE(pp.last_was_low_light());
    EXPECT_EQ(tensor.size(), static_cast<size_t>(3 * 640 * 640));
    for (float v : tensor) {
        EXPECT_GE(v, 0.f);
        EXPECT_LE(v, 1.f);
    }
}

TEST(Preprocessor, EmptyFrameThrows) {
    Preprocessor pp;
    cv::Mat empty;
    EXPECT_THROW(pp.process(empty), std::runtime_error);
}
