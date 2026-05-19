#include <gtest/gtest.h>
#include "bowl_tracker.hpp"

using cbw::BowlTracker;
using cbw::Detection;

static Detection make_det(float cx, int class_id, float conf = 0.9f) {
    Detection d;
    d.cx = cx; d.cy = 0.5f; d.w = 0.15f; d.h = 0.15f;
    d.confidence = conf;
    d.class_id   = class_id;
    return d;
}

TEST(BowlTracker, SlotIdsAreAlwaysSet) {
    BowlTracker tracker;
    auto states = tracker.update({});
    EXPECT_EQ(states[0].bowl_id, "bowl_1");
    EXPECT_EQ(states[1].bowl_id, "bowl_2");
}

TEST(BowlTracker, LeftAssignedBowl1RightBowl2) {
    BowlTracker tracker;
    auto dets = std::vector<Detection>{
        make_det(0.2f, cbw::CLASS_BOWL_EMPTY),
        make_det(0.8f, cbw::CLASS_BOWL_NOT_EMPTY)
    };
    auto states = tracker.update(dets);
    EXPECT_EQ(states[0].state, "empty");
    EXPECT_EQ(states[1].state, "not_empty");
}

TEST(BowlTracker, RegistrationRequires3DualDetectionFrames) {
    BowlTracker tracker;
    EXPECT_FALSE(tracker.is_registered());

    auto two_dets = std::vector<Detection>{
        make_det(0.2f, cbw::CLASS_BOWL_EMPTY),
        make_det(0.8f, cbw::CLASS_BOWL_NOT_EMPTY)
    };

    for (int i = 0; i < 2; ++i) {
        tracker.update(two_dets);
        EXPECT_FALSE(tracker.is_registered()) << "after frame " << (i + 1);
    }
    tracker.update(two_dets);
    EXPECT_TRUE(tracker.is_registered());
}

TEST(BowlTracker, SingleDetectionFrameDoesNotCountForRegistration) {
    BowlTracker tracker;
    for (int i = 0; i < 30; ++i)
        tracker.update({make_det(0.5f, cbw::CLASS_BOWL_EMPTY)});
    // Only single-detection frames — never reaches registration
    EXPECT_FALSE(tracker.is_registered());
}

TEST(BowlTracker, HoldFramesDelaysUndetected) {
    BowlTracker tracker(/*hold_frames=*/3);
    // Establish a state
    tracker.update({make_det(0.2f, cbw::CLASS_BOWL_EMPTY),
                    make_det(0.8f, cbw::CLASS_BOWL_NOT_EMPTY)});

    // No detections — state should be held for 3 frames
    for (int i = 0; i < 3; ++i) {
        auto states = tracker.update({});
        EXPECT_NE(states[0].state, "undetected") << "frame " << i;
        EXPECT_NE(states[1].state, "undetected") << "frame " << i;
    }
    // 4th frame without detection — now undetected
    auto states = tracker.update({});
    EXPECT_EQ(states[0].state, "undetected");
    EXPECT_EQ(states[1].state, "undetected");
}

TEST(BowlTracker, EmptySinceSetOnTransitionToEmpty) {
    BowlTracker tracker;
    // First frame: not_empty
    tracker.update({make_det(0.2f, cbw::CLASS_BOWL_NOT_EMPTY),
                    make_det(0.8f, cbw::CLASS_BOWL_NOT_EMPTY)});
    EXPECT_EQ(tracker.update({make_det(0.2f, cbw::CLASS_BOWL_NOT_EMPTY),
                               make_det(0.8f, cbw::CLASS_BOWL_NOT_EMPTY)})[0].empty_since_ms,
              -1);

    // Transition to empty
    auto states = tracker.update({make_det(0.2f, cbw::CLASS_BOWL_EMPTY),
                                  make_det(0.8f, cbw::CLASS_BOWL_NOT_EMPTY)});
    EXPECT_GT(states[0].empty_since_ms, 0);
}
