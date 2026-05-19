// CatBowlWatch — Phase 3 toolchain smoke binary.
//
// Reports versions of every third-party library so we can confirm the WSL
// Ubuntu build chain is wired correctly before layering on the real inference
// components. This file will be replaced by the real service entrypoint in a
// subsequent commit once Capture / Preprocessor / InferenceBackend / Tracker /
// DebounceEngine / HttpServer are in place.

#include <onnxruntime_cxx_api.h>
#include <opencv2/core/version.hpp>
#include <spdlog/spdlog.h>
#include <spdlog/version.h>

#include <httplib.h>

#include <cstdio>

int main() {
    std::printf("CatBowlWatch toolchain check\n");
    std::printf("OpenCV: %d.%d.%d\n",
                CV_VERSION_MAJOR, CV_VERSION_MINOR, CV_VERSION_REVISION);
    std::printf("spdlog: %d.%d.%d\n",
                SPDLOG_VER_MAJOR, SPDLOG_VER_MINOR, SPDLOG_VER_PATCH);
    std::printf("ONNX Runtime: %s\n", OrtGetApiBase()->GetVersionString());

    // cpp-httplib has no compile-time version macro — instantiate the type to
    // confirm the header parses and links.
    httplib::Server server;
    (void)server;

    spdlog::info("CatBowlWatch smoke binary linked successfully.");
    std::printf("OK\n");
    return 0;
}
