// CatBowlWatch — Phase 3 GoogleTest smoke test.
//
// Confirms the test runner builds, discovers, and executes tests. Real
// component tests (DebounceEngine timer math, BowlTracker registration
// warm-up, Postprocessor NMS) land in subsequent commits.

#include <gtest/gtest.h>

TEST(Smoke, ToolchainOk) {
    EXPECT_EQ(2 + 2, 4);
}
