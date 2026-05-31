#include "debounce_engine.hpp"
#include <chrono>

namespace cbw {

DebounceEngine::DebounceEngine(int64_t debounce_ms, int64_t cooldown_ms, ClockFn clock)
    : debounce_ms_(debounce_ms), cooldown_ms_(cooldown_ms),
      clock_(clock ? std::move(clock) : ClockFn{}) {}

int64_t DebounceEngine::wall_ms() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::system_clock::now().time_since_epoch()).count();
}

int64_t DebounceEngine::now_ms() const {
    return clock_ ? clock_() : wall_ms();
}

std::vector<AlertEvent> DebounceEngine::update(const std::array<BowlState, 2>& states) {
    int64_t now = now_ms();
    std::vector<AlertEvent> alerts;

    for (int i = 0; i < 2; ++i) {
        const BowlState& s = states[i];
        if (s.state != "empty" || s.empty_since_ms < 0) continue;

        bool debounce_ok = (now - s.empty_since_ms) >= debounce_ms_;
        bool cooldown_ok = (last_alert_ms_[i] < 0) ||
                           (now - last_alert_ms_[i]) >= cooldown_ms_;

        if (debounce_ok && cooldown_ok) {
            alerts.push_back({s.bowl_id, now, s.empty_since_ms});
            last_alert_ms_[i] = now;
        }
    }
    return alerts;
}

} // namespace cbw
