#pragma once
#include "types.hpp"
#include <opencv2/core.hpp>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <vector>

namespace cbw {

// Sends Telegram POST /sendPhoto alerts from a dedicated background thread.
// Non-blocking: send() enqueues; the worker thread handles retries.
// stop() drains the queue and joins the thread before returning.
//
// If bot_token or chat_id is empty, all send() calls are silently skipped.
// The http_post parameter is injectable for unit testing (defaults to libcurl).
class TelegramNotifier {
public:
    // Returns true on success, false on failure (triggers retry).
    using HttpPostFn = std::function<bool(const std::string& bot_token,
                                          const std::string& chat_id,
                                          const std::string& caption,
                                          const std::vector<uint8_t>& jpeg_bytes)>;

    explicit TelegramNotifier(std::string  bot_token,
                               std::string  chat_id,
                               int          max_retries    = 3,
                               int64_t      retry_delay_ms = 2000,
                               HttpPostFn   http_post      = nullptr);
    ~TelegramNotifier();

    // Non-blocking enqueue. No-op when credentials are not configured.
    void send(const AlertEvent& alert, const cv::Mat& snapshot);

    // Drain queue and join worker thread. Safe to call multiple times.
    void stop();

    static std::string format_caption(const AlertEvent& alert);

private:
    struct Payload {
        AlertEvent           alert;
        std::vector<uint8_t> jpeg_bytes;
    };

    std::string bot_token_;
    std::string chat_id_;
    int         max_retries_;
    int64_t     retry_delay_ms_;
    HttpPostFn  http_post_;

    std::thread             worker_;
    std::mutex              mutex_;
    std::condition_variable cv_;
    std::queue<Payload>     queue_;
    bool                    stop_requested_{false};

    void worker_loop();
    void attempt_send(const Payload& payload);

    static bool curl_post(const std::string& bot_token,
                          const std::string& chat_id,
                          const std::string& caption,
                          const std::vector<uint8_t>& jpeg_bytes);
};

} // namespace cbw
