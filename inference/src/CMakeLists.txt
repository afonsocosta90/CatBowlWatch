# Phase 3 component library + service binaries

add_library(catbowlwatch_lib STATIC
    capture.cpp
    preprocessor.cpp
    onnx_backend.cpp
    postprocessor.cpp
    bowl_tracker.cpp
    debounce_engine.cpp
    http_server.cpp
    telegram_notifier.cpp
)

target_link_libraries(catbowlwatch_lib PUBLIC
    catbowlwatch_warnings
    spdlog::spdlog
    httplib::httplib
    onnxruntime
    ${OpenCV_LIBS}
    CURL::libcurl
)

target_include_directories(catbowlwatch_lib PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${OpenCV_INCLUDE_DIRS}
)

# Toolchain smoke binary — verifies lib linkage only; kept for CI
add_executable(catbowlwatch_smoke main.cpp)
target_link_libraries(catbowlwatch_smoke PRIVATE
    catbowlwatch_warnings
    spdlog::spdlog
    httplib::httplib
    onnxruntime
    ${OpenCV_LIBS}
)
target_include_directories(catbowlwatch_smoke PRIVATE ${OpenCV_INCLUDE_DIRS})

# Real inference service binary
add_executable(catbowlwatch service_main.cpp)
target_link_libraries(catbowlwatch PRIVATE catbowlwatch_lib)
