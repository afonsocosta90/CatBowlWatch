#include "http_server.hpp"
#include <opencv2/imgcodecs.hpp>
#include <spdlog/spdlog.h>
#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace cbw {

HttpServer::HttpServer(int port, int thread_pool_size)
    : port_(port), start_ms_(now_ms()) {
    server_.new_task_queue = [thread_pool_size] {
        return new httplib::ThreadPool(thread_pool_size);
    };

    server_.Get("/status", [this](const httplib::Request& req, httplib::Response& res) {
        handle_status(req, res);
    });
    server_.Get("/photo", [this](const httplib::Request& req, httplib::Response& res) {
        handle_photo(req, res);
    });
}

HttpServer::~HttpServer() { stop(); }

int64_t HttpServer::now_ms() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::system_clock::now().time_since_epoch()).count();
}

void HttpServer::update(const std::array<BowlState, 2>& states,
                        const cv::Mat& frame,
                        bool registration_pending,
                        double fps) {
    std::lock_guard<std::mutex> lock(mutex_);
    states_                = states;
    frame.copyTo(latest_frame_);
    registration_pending_  = registration_pending;
    fps_                   = fps;
}

void HttpServer::start() {
    thread_ = std::thread([this] {
        spdlog::info("HttpServer listening on port {}", port_);
        server_.listen("0.0.0.0", port_);
    });
}

void HttpServer::stop() {
    if (server_.is_running()) server_.stop();
    if (thread_.joinable())   thread_.join();
}

void HttpServer::handle_status(const httplib::Request&, httplib::Response& res) {
    std::array<BowlState, 2> snap_states;
    bool   snap_pending;
    double snap_fps;
    int64_t snap_start;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        snap_states  = states_;
        snap_pending = registration_pending_;
        snap_fps     = fps_;
        snap_start   = start_ms_;
    }
    int64_t uptime_s = (now_ms() - snap_start) / 1000;
    res.set_content(build_status_json(snap_states, snap_pending, snap_fps, uptime_s),
                    "application/json");
}

void HttpServer::handle_photo(const httplib::Request&, httplib::Response& res) {
    cv::Mat frame_copy;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (latest_frame_.empty()) {
            res.status = 503;
            res.set_content("{\"error\":\"no frame yet\"}", "application/json");
            return;
        }
        latest_frame_.copyTo(frame_copy);
        // Lock released here — encode outside the lock to avoid stalling inference
    }

    std::vector<uchar> buf;
    if (!cv::imencode(".jpg", frame_copy, buf)) {
        res.status = 500;
        res.set_content("{\"error\":\"encode failed\"}", "application/json");
        return;
    }
    res.set_content(reinterpret_cast<const char*>(buf.data()), buf.size(), "image/jpeg");
}

std::string HttpServer::build_status_json(const std::array<BowlState, 2>& states,
                                          bool registration_pending,
                                          double fps,
                                          int64_t uptime_s) {
    auto fmt_bowl = [](const BowlState& s) -> std::string {
        int64_t empty_for_s = 0;
        if (s.state == "empty" && s.empty_since_ms >= 0) {
            int64_t now = std::chrono::duration_cast<std::chrono::milliseconds>(
                              std::chrono::system_clock::now().time_since_epoch()).count();
            empty_for_s = (now - s.empty_since_ms) / 1000;
        }
        std::ostringstream o;
        o << "{\"id\":\"" << s.bowl_id << "\""
          << ",\"state\":\"" << s.state << "\""
          << ",\"empty_for_s\":" << empty_for_s
          << ",\"confidence\":" << std::fixed << std::setprecision(2) << s.confidence
          << "}";
        return o.str();
    };

    std::ostringstream o;
    o << std::fixed << std::setprecision(1);
    o << "{"
      << "\"bowls\":[" << fmt_bowl(states[0]) << "," << fmt_bowl(states[1]) << "]"
      << ",\"registration_pending\":" << (registration_pending ? "true" : "false")
      << ",\"fps\":" << fps
      << ",\"uptime_s\":" << uptime_s
      << "}";
    return o.str();
}

} // namespace cbw
