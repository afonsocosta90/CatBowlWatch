#include <gtest/gtest.h>
#include "postprocessor.hpp"

using cbw::Postprocessor;
using cbw::Detection;

// Build a [1,6,8400] output with one detection at the given anchor slot
static std::vector<float> make_raw(float cx_px, float cy_px, float w_px, float h_px,
                                   float s0, float s1, int anchor = 0) {
    constexpr int N = 8400;
    std::vector<float> raw(6 * N, 0.f);
    raw[0 * N + anchor] = cx_px;
    raw[1 * N + anchor] = cy_px;
    raw[2 * N + anchor] = w_px;
    raw[3 * N + anchor] = h_px;
    raw[4 * N + anchor] = s0;
    raw[5 * N + anchor] = s1;
    return raw;
}

TEST(Postprocessor, DetectsBowlEmpty) {
    Postprocessor pp(0.3f, 0.45f);
    auto raw = make_raw(320, 240, 100, 80, 0.9f, 0.1f);
    auto dets = pp.process(raw);
    ASSERT_EQ(dets.size(), 1u);
    EXPECT_EQ(dets[0].class_id, cbw::CLASS_BOWL_EMPTY);
    EXPECT_NEAR(dets[0].confidence, 0.9f, 1e-4f);
}

TEST(Postprocessor, DetectsBowlNotEmpty) {
    Postprocessor pp(0.3f, 0.45f);
    auto raw = make_raw(320, 240, 100, 80, 0.1f, 0.95f);
    auto dets = pp.process(raw);
    ASSERT_EQ(dets.size(), 1u);
    EXPECT_EQ(dets[0].class_id, cbw::CLASS_BOWL_NOT_EMPTY);
}

TEST(Postprocessor, BelowThresholdFiltered) {
    Postprocessor pp(0.5f, 0.45f);
    auto raw = make_raw(320, 240, 100, 80, 0.3f, 0.1f);
    EXPECT_TRUE(pp.process(raw).empty());
}

TEST(Postprocessor, NmsDeduplicatesOverlappingAnchors) {
    // Two nearly identical anchors — should collapse to one
    constexpr int N = 8400;
    std::vector<float> raw(6 * N, 0.f);
    for (int a : {0, 1}) {
        raw[0 * N + a] = 320.f;
        raw[1 * N + a] = 240.f;
        raw[2 * N + a] = 100.f;
        raw[3 * N + a] = 80.f;
        raw[4 * N + a] = 0.9f - a * 0.02f;
        raw[5 * N + a] = 0.05f;
    }
    Postprocessor pp(0.3f, 0.45f);
    EXPECT_EQ(pp.process(raw).size(), 1u);
}

TEST(Postprocessor, TwoSeparatedBowlsBothPass) {
    // Two detections far apart — both should survive NMS
    constexpr int N = 8400;
    std::vector<float> raw(6 * N, 0.f);
    // bowl at left
    raw[0 * N + 0] = 160.f; raw[1 * N + 0] = 320.f;
    raw[2 * N + 0] = 80.f;  raw[3 * N + 0] = 80.f;
    raw[4 * N + 0] = 0.9f;  raw[5 * N + 0] = 0.05f;
    // bowl at right
    raw[0 * N + 1] = 480.f; raw[1 * N + 1] = 320.f;
    raw[2 * N + 1] = 80.f;  raw[3 * N + 1] = 80.f;
    raw[4 * N + 1] = 0.05f; raw[5 * N + 1] = 0.88f;

    Postprocessor pp(0.3f, 0.45f);
    auto dets = pp.process(raw);
    EXPECT_EQ(dets.size(), 2u);
}

TEST(Postprocessor, WrongSizeThrows) {
    Postprocessor pp;
    EXPECT_THROW(pp.process({}), std::runtime_error);
    EXPECT_THROW(pp.process({1.f, 2.f, 3.f}), std::runtime_error);
}
