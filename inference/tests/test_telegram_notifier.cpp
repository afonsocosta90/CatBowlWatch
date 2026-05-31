#include <gtest/gtest.h>
#include "telegram_notifier.hpp"
#include <atomic>
#include <opencv2/core.hpp>

using cbw::TelegramNotifier;
using cbw::AlertEvent;

// alert with timestamp=1000000ms, empty_since=937000ms → 63s empty
static AlertEvent make_alert(const std::string& bowl_id = "bowl_1",
                              int64_t timestamp_ms  = 1000000,
                              int64_t empty_since_ms = 937000) {
    return {bowl_id, timestamp_ms, empty_since_ms};
}

static cv::Mat make_frame() {
    return cv::Mat(10, 10, CV_8UC3, cv::Scalar(128, 64, 32));
}

// ── Caption formatting ────────────────────────────────────────────────────────

TEST(TelegramNotifier, CaptionContainsBowlId) {
    auto caption = TelegramNotifier::format_caption(make_alert("bowl_2"));
    EXPECT_NE(caption.find("bowl_2"), std::string::npos);
}

TEST(TelegramNotifier, CaptionContainsEmptyDuration) {
    // 1000000 - 937000 = 63000ms = 63s
    auto caption = TelegramNotifier::format_caption(make_alert("bowl_1", 1000000, 937000));
    EXPECT_NE(caption.find("63"), std::string::npos);
}

TEST(TelegramNotifier, CaptionContainsEmptyKeyword) {
    EXPECT_NE(TelegramNotifier::format_caption(make_alert()).find("EMPTY"), std::string::npos);
}

TEST(TelegramNotifier, CaptionContainsTimeString) {
    auto caption = TelegramNotifier::format_caption(make_alert());
    // Verify format_caption produces a non-trivial caption with a colon (hh:mm:ss)
    EXPECT_NE(caption.find(':'), std::string::npos);
}

// ── No-op when credentials are missing ───────────────────────────────────────

TEST(TelegramNotifier, NoopWhenTokenEmpty) {
    int calls = 0;
    TelegramNotifier n("", "12345", 3, 0,
                       [&](auto&, auto&, auto&, auto&) { ++calls; return true; });
    n.send(make_alert(), make_frame());
    n.stop();
    EXPECT_EQ(calls, 0);
}

TEST(TelegramNotifier, NoopWhenChatIdEmpty) {
    int calls = 0;
    TelegramNotifier n("token123", "", 3, 0,
                       [&](auto&, auto&, auto&, auto&) { ++calls; return true; });
    n.send(make_alert(), make_frame());
    n.stop();
    EXPECT_EQ(calls, 0);
}

// ── Successful send ───────────────────────────────────────────────────────────

TEST(TelegramNotifier, SendCallsHttpPostOnce) {
    std::atomic<int> calls{0};
    TelegramNotifier n("tok", "chat", 3, 0,
                       [&](auto&, auto&, auto&, auto&) { ++calls; return true; });
    n.send(make_alert(), make_frame());
    n.stop();
    EXPECT_EQ(calls.load(), 1);
}

TEST(TelegramNotifier, SendPassesCorrectCredentials) {
    std::string captured_token, captured_chat;
    TelegramNotifier n("my-bot-token", "my-chat-id", 1, 0,
                       [&](const std::string& tok, const std::string& chat,
                           auto&, auto&) {
                           captured_token = tok;
                           captured_chat  = chat;
                           return true;
                       });
    n.send(make_alert(), make_frame());
    n.stop();
    EXPECT_EQ(captured_token, "my-bot-token");
    EXPECT_EQ(captured_chat,  "my-chat-id");
}

TEST(TelegramNotifier, JpegBytesNonEmpty) {
    std::vector<uint8_t> received_bytes;
    TelegramNotifier n("tok", "chat", 1, 0,
                       [&](auto&, auto&, auto&,
                           const std::vector<uint8_t>& jpeg) {
                           received_bytes = jpeg;
                           return true;
                       });
    n.send(make_alert(), make_frame());
    n.stop();
    EXPECT_FALSE(received_bytes.empty());
}

// ── Retry behaviour ───────────────────────────────────────────────────────────

TEST(TelegramNotifier, RetriesOnFailure) {
    std::atomic<int> calls{0};
    TelegramNotifier n("tok", "chat", 3, 0,  // 3 retries, 0ms delay for fast test
                       [&](auto&, auto&, auto&, auto&) { ++calls; return false; });
    n.send(make_alert(), make_frame());
    n.stop();
    EXPECT_EQ(calls.load(), 3);
}

TEST(TelegramNotifier, StopsRetryingOnFirstSuccess) {
    std::atomic<int> calls{0};
    TelegramNotifier n("tok", "chat", 3, 0,
                       [&](auto&, auto&, auto&, auto&) {
                           return ++calls >= 2;  // fail once, then succeed
                       });
    n.send(make_alert(), make_frame());
    n.stop();
    EXPECT_EQ(calls.load(), 2);
}

// ── Queue and threading ───────────────────────────────────────────────────────

TEST(TelegramNotifier, MultipleAlertsAllDelivered) {
    std::atomic<int> calls{0};
    TelegramNotifier n("tok", "chat", 1, 0,
                       [&](auto&, auto&, auto&, auto&) { ++calls; return true; });
    n.send(make_alert("bowl_1"), make_frame());
    n.send(make_alert("bowl_2"), make_frame());
    n.stop();
    EXPECT_EQ(calls.load(), 2);
}

TEST(TelegramNotifier, StopWithEmptyQueueDoesNotHang) {
    TelegramNotifier n("tok", "chat", 1, 0,
                       [](auto&, auto&, auto&, auto&) { return true; });
    n.stop();
    SUCCEED();
}
