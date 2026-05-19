# Phase 3 smoke binary — verifies all third-party libs link and report version.
# Real components (Capture, Preprocessor, InferenceBackend, BowlTracker,
# DebounceEngine, HttpServer) land in subsequent commits.

add_executable(catbowlwatch_smoke main.cpp)
target_link_libraries(catbowlwatch_smoke PRIVATE
    catbowlwatch_warnings
    spdlog::spdlog
    httplib::httplib
    onnxruntime
    ${OpenCV_LIBS}
)
target_include_directories(catbowlwatch_smoke PRIVATE ${OpenCV_INCLUDE_DIRS})
