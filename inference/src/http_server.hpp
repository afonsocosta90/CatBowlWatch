#pragma once
#include "types.hpp"
#include <httplib.h>
#include <opencv2/core.hpp>
#include <array>
#include <atomic>
#include <chrono>
#include <mutex>
#include <string>
#include <thread>

namespace cbw {

// GET /status  — JSON bowl states, fps, uptime
// GET /photo   — JPEG of the latest annotated frame
// Thread-safe: inference loop calls update(); HTTP threads call handlers.
class HttpServer {
public:
    HttpServer(int port, int thread_pool_size = 4);
    ~HttpServer();

    // Called from the inference loop; non-blocking
    void update(const std::array<BowlState, 2>& states,
                const cv::Mat& frame,
                bool  registration_pending,
                double fps);

    void start();  // spawns background thread
    void stop();

private:
    int             port_;
    httplib::Server server_;
    std::mutex      mutex_;

    std::array<BowlState, 2> states_;
    cv::Mat                  latest_frame_;
    bool                     registration_pending_ = true;
    double                   fps_                  = 0.0;
    int64_t                  start_ms_;
    std::thread              thread_;

    void handle_status(const httplib::Request&, httplib::Response& res);
    void handle_photo (const httplib::Request&, httplib::Response& res);

    static std::string build_status_json(const std::array<BowlState, 2>& states,
                                         bool registration_pending,
                                         double fps, int64_t uptime_s);
    static int64_t now_ms();
};

} // namespace cbw
