#include <gtest/gtest.h>
#include "debounce_engine.hpp"

using cbw::DebounceEngine;
using cbw::BowlState;

static BowlState make_empty(const std::string& id, int64_t empty_since_ms) {
    BowlState s;
    s.bowl_id       = id;
    s.state         = "empty";
    s.empty_since_ms = empty_since_ms;
    return s;
}

static BowlState make_not_empty(const std::string& id) {
    BowlState s;
    s.bowl_id        = id;
    s.state          = "not_empty";
    s.empty_since_ms = -1;
    return s;
}

TEST(DebounceEngine, NoAlertBeforeDebounceWindow) {
    int64_t t = 1000000;
    DebounceEngine engine(60000, 300000, [&t] { return t; });

    std::array<BowlState, 2> states = {
        make_empty("bowl_1", t - 30000),  // only 30s empty — under the 60s threshold
        make_not_empty("bowl_2")
    };
    auto alerts = engine.update(states);
    EXPECT_TRUE(alerts.empty());
}

TEST(DebounceEngine, AlertFiresAfterDebounceWindow) {
    int64_t t = 1000000;
    DebounceEngine engine(60000, 300000, [&t] { return t; });

    std::array<BowlState, 2> states = {
        make_empty("bowl_1", t - 65000),  // 65s empty — over the 60s threshold
        make_not_empty("bowl_2")
    };
    auto alerts = engine.update(states);
    ASSERT_EQ(alerts.size(), 1u);
    EXPECT_EQ(alerts[0].bowl_id, "bowl_1");
}

TEST(DebounceEngine, CooldownPreventsSecondAlert) {
    int64_t t = 1000000;
    DebounceEngine engine(60000, 300000, [&t] { return t; });

    std::array<BowlState, 2> states = {
        make_empty("bowl_1", t - 65000),
        make_not_empty("bowl_2")
    };

    // First alert fires
    auto first = engine.update(states);
    EXPECT_EQ(first.size(), 1u);

    // Advance time by less than cooldown (100s < 300s)
    t += 100000;
    auto second = engine.update(states);
    EXPECT_TRUE(second.empty());

    // Advance past cooldown (>300s total)
    t += 210000;
    auto third = engine.update(states);
    EXPECT_EQ(third.size(), 1u);
}

TEST(DebounceEngine, BothBowlsAlertIndependently) {
    int64_t t = 1000000;
    DebounceEngine engine(60000, 300000, [&t] { return t; });

    std::array<BowlState, 2> states = {
        make_empty("bowl_1", t - 70000),
        make_empty("bowl_2", t - 80000)
    };
    auto alerts = engine.update(states);
    EXPECT_EQ(alerts.size(), 2u);
}

TEST(DebounceEngine, NotEmptyStateNeverAlerts) {
    int64_t t = 1000000;
    DebounceEngine engine(1, 1, [&t] { return t; });  // very short thresholds

    std::array<BowlState, 2> states = {
        make_not_empty("bowl_1"),
        make_not_empty("bowl_2")
    };
    EXPECT_TRUE(engine.update(states).empty());
}

TEST(DebounceEngine, UndetectedStateNeverAlerts) {
    int64_t t = 1000000;
    DebounceEngine engine(1, 1, [&t] { return t; });

    BowlState undetected;
    undetected.bowl_id        = "bowl_1";
    undetected.state          = "undetected";
    undetected.empty_since_ms = -1;

    std::array<BowlState, 2> states = {undetected, make_not_empty("bowl_2")};
    EXPECT_TRUE(engine.update(states).empty());
}
