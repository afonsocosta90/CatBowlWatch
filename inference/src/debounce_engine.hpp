#pragma once
#include "types.hpp"
#include <array>
#include <cstdint>
#include <functional>
#include <vector>

namespace cbw {

// Per-bowl debounce: fires when state=="empty" AND empty_for >= debounce_ms
// AND cooldown since last alert >= cooldown_ms.
// Accepts an optional clock injection for deterministic unit tests.
class DebounceEngine {
public:
    using ClockFn = std::function<int64_t()>;

    DebounceEngine(int64_t debounce_ms = 60000,
                   int64_t cooldown_ms = 300000,
                   ClockFn clock       = nullptr);

    std::vector<AlertEvent> update(const std::array<BowlState, 2>& states);

private:
    int64_t                 debounce_ms_;
    int64_t                 cooldown_ms_;
    ClockFn                 clock_;
    std::array<int64_t, 2>  last_alert_ms_ = {-1, -1};

    int64_t now_ms() const;
    static  int64_t wall_ms();
};

} // namespace cbw
