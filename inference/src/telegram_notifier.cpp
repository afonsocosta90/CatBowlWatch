#include "telegram_notifier.hpp"
#include <spdlog/spdlog.h>
#include <opencv2/imgcodecs.hpp>
#include <curl/curl.h>
#include <chrono>
#include <ctime>
#include <sstream>
#include <thread>

namespace cbw {

TelegramNotifier::TelegramNotifier(std::string token,
                                    std::string chat_id,
                                    int         max_retries,
                                    int64_t     retry_delay_ms,
                                    HttpPostFn  http_post)
    : bot_token_(std::move(token))
    , chat_id_(std::move(chat_id))
    , max_retries_(max_retries)
    , retry_delay_ms_(retry_delay_ms)
    , http_post_(http_post ? std::move(http_post)
                           : HttpPostFn(&TelegramNotifier::curl_post))
{
    if (!bot_token_.empty() && !chat_id_.empty())
        worker_ = std::thread(&TelegramNotifier::worker_loop, this);
}

TelegramNotifier::~TelegramNotifier() {
    stop();
}

void TelegramNotifier::send(const AlertEvent& alert, const cv::Mat& snapshot) {
    if (bot_token_.empty() || chat_id_.empty()) {
        spdlog::warn("TelegramNotifier: credentials not set — skipping alert for {}",
                     alert.bowl_id);
        return;
    }
    std::vector<uint8_t> jpeg_bytes;
    cv::imencode(".jpg", snapshot, jpeg_bytes, {cv::IMWRITE_JPEG_QUALITY, 85});
    {
        std::lock_guard<std::mutex> lock(mutex_);
        queue_.push({alert, std::move(jpeg_bytes)});
    }
    cv_.notify_one();
}

void TelegramNotifier::stop() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        stop_requested_ = true;
    }
    cv_.notify_all();
    if (worker_.joinable())
        worker_.join();
}

std::string TelegramNotifier::format_caption(const AlertEvent& alert) {
    int64_t empty_for_s = (alert.timestamp_ms - alert.empty_since_ms) / 1000;

    std::time_t t = static_cast<std::time_t>(alert.timestamp_ms / 1000);
    std::tm tm_buf{};
#ifdef _WIN32
    localtime_s(&tm_buf, &t);
#else
    localtime_r(&t, &tm_buf);
#endif
    char time_str[32];
    std::strftime(time_str, sizeof(time_str), "%Y-%m-%d %H:%M:%S", &tm_buf);

    std::ostringstream oss;
    oss << "\U0001F431 " << alert.bowl_id << " is EMPTY\n"
        << "Empty for: " << empty_for_s << " seconds\n"
        << "Time: " << time_str;
    return oss.str();
}

void TelegramNotifier::worker_loop() {
    while (true) {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [this] { return !queue_.empty() || stop_requested_; });

        while (!queue_.empty()) {
            Payload p = std::move(queue_.front());
            queue_.pop();
            lock.unlock();
            attempt_send(p);
            lock.lock();
        }

        if (stop_requested_) break;
    }
}

void TelegramNotifier::attempt_send(const Payload& payload) {
    std::string caption = format_caption(payload.alert);
    for (int attempt = 0; attempt < max_retries_; ++attempt) {
        if (http_post_(bot_token_, chat_id_, caption, payload.jpeg_bytes)) {
            spdlog::info("TelegramNotifier: alert sent for {}", payload.alert.bowl_id);
            return;
        }
        spdlog::warn("TelegramNotifier: attempt {}/{} failed for {}",
                     attempt + 1, max_retries_, payload.alert.bowl_id);
        if (attempt + 1 < max_retries_ && retry_delay_ms_ > 0)
            std::this_thread::sleep_for(std::chrono::milliseconds(retry_delay_ms_));
    }
    spdlog::error("TelegramNotifier: all {} retries exhausted for {}",
                  max_retries_, payload.alert.bowl_id);
}

bool TelegramNotifier::curl_post(const std::string& token,
                                  const std::string& chat_id,
                                  const std::string& caption,
                                  const std::vector<uint8_t>& jpeg_bytes) {
    CURL* curl = curl_easy_init();
    if (!curl) return false;

    const std::string url = "https://api.telegram.org/bot" + token + "/sendPhoto";

    curl_mime* form = curl_mime_init(curl);

    curl_mimepart* part = curl_mime_addpart(form);
    curl_mime_name(part, "chat_id");
    curl_mime_data(part, chat_id.c_str(), CURL_ZERO_TERMINATED);

    part = curl_mime_addpart(form);
    curl_mime_name(part, "photo");
    curl_mime_data(part, reinterpret_cast<const char*>(jpeg_bytes.data()),
                   jpeg_bytes.size());
    curl_mime_filename(part, "snapshot.jpg");
    curl_mime_type(part, "image/jpeg");

    part = curl_mime_addpart(form);
    curl_mime_name(part, "caption");
    curl_mime_data(part, caption.c_str(), CURL_ZERO_TERMINATED);

    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_MIMEPOST, form);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, 10L);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION,
                     +[](char*, size_t size, size_t nmemb, void*) -> size_t {
                         return size * nmemb;
                     });

    CURLcode  res       = curl_easy_perform(curl);
    long      http_code = 0;
    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);

    curl_mime_free(form);
    curl_easy_cleanup(curl);

    if (res != CURLE_OK || http_code != 200) {
        spdlog::debug("TelegramNotifier: curl result={} http={}",
                      static_cast<int>(res), http_code);
        return false;
    }
    return true;
}

} // namespace cbw
