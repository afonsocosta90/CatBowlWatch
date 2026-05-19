#pragma once
#include "types.hpp"
#include <array>
#include <vector>

namespace cbw {

// Assigns bowl identity by x-coordinate (left=bowl_1, right=bowl_2) and
// implements detection hold and registration warm-up logic.
// See ARCHITECTURE.md §4.5 for the BowlState contract.
class BowlTracker {
public:
    explicit BowlTracker(int hold_frames = 5);

    // Update with detections from current frame; always returns [bowl_1, bowl_2]
    std::array<BowlState, 2> update(const std::vector<Detection>& detections);

    // True once both bowls have been seen together ≥3 times in the first 30 frames
    bool is_registered() const { return registered_; }

private:
    int  hold_frames_;
    bool registered_           = false;
    int  processed_frames_     = 0;
    int  dual_detection_count_ = 0;

    std::array<BowlState, 2> states_;
    std::array<int, 2>       hold_counters_ = {0, 0};

    static constexpr int REGISTRATION_WINDOW = 30;
    static constexpr int REGISTRATION_NEEDED = 3;

    static int64_t now_ms();
};

} // namespace cbw
