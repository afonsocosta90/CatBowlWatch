# CatBowlWatch C++17 inference service (Phase 3).
#
# Laptop build (default):
#     cmake -B build -DCMAKE_BUILD_TYPE=Release -DWITH_TENSORRT=OFF
#     cmake --build build -j
#     ctest --test-dir build --output-on-failure
#
# Jetson build (Phase 5):
#     cmake -B build -DCMAKE_BUILD_TYPE=Release -DWITH_TENSORRT=ON
#
# External deps:
#   - OpenCV: find_package (system-installed via libopencv-dev)
#   - ONNX Runtime: imported from $ONNXRUNTIME_ROOT (or -DONNXRUNTIME_ROOT=...)
#   - TensorRT: only when WITH_TENSORRT=ON; uses NvInfer.h from system / JetPack
# Vendored via FetchContent (no system install):
#   - spdlog (header-only logging)
#   - cpp-httplib (header-only HTTP server)
#   - GoogleTest (only when BUILD_TESTING=ON, which is the default)

cmake_minimum_required(VERSION 3.16)
project(catbowlwatch LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(WITH_TENSORRT "Build TensorRT backend (requires NvInfer.h, Jetson only)" OFF)
option(BUILD_TESTING "Build unit tests" ON)

# -----------------------------------------------------------------------------
# Compiler warnings
# -----------------------------------------------------------------------------
add_library(catbowlwatch_warnings INTERFACE)
target_compile_options(catbowlwatch_warnings INTERFACE
    -Wall -Wextra -Wpedantic -Wshadow -Wnon-virtual-dtor
)

# -----------------------------------------------------------------------------
# External dependencies
# -----------------------------------------------------------------------------
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs videoio)

# ONNX Runtime: imported from $ONNXRUNTIME_ROOT
if(NOT DEFINED ONNXRUNTIME_ROOT)
    if(DEFINED ENV{ONNXRUNTIME_ROOT})
        set(ONNXRUNTIME_ROOT "$ENV{ONNXRUNTIME_ROOT}")
    endif()
endif()
if(NOT ONNXRUNTIME_ROOT)
    if(APPLE)
        message(FATAL_ERROR
            "ONNXRUNTIME_ROOT not set. Run scripts/setup_macos_dev.sh first, then:\n"
            "    export ONNXRUNTIME_ROOT=~/onnxruntime-osx-arm64-<ver>  # Apple Silicon\n"
            "    export ONNXRUNTIME_ROOT=~/onnxruntime-osx-x86_64-<ver> # Intel Mac")
    else()
        message(FATAL_ERROR
            "ONNXRUNTIME_ROOT not set. Run scripts/setup_wsl_dev.sh first, then:\n"
            "    export ONNXRUNTIME_ROOT=~/onnxruntime-linux-x64-<ver>")
    endif()
endif()

find_path(ONNXRUNTIME_INCLUDE_DIR onnxruntime_cxx_api.h
    HINTS "${ONNXRUNTIME_ROOT}/include"
    REQUIRED)
find_library(ONNXRUNTIME_LIBRARY onnxruntime
    HINTS "${ONNXRUNTIME_ROOT}/lib"
    REQUIRED)

add_library(onnxruntime SHARED IMPORTED)
set_target_properties(onnxruntime PROPERTIES
    IMPORTED_LOCATION "${ONNXRUNTIME_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${ONNXRUNTIME_INCLUDE_DIR}"
)
message(STATUS "ONNX Runtime: ${ONNXRUNTIME_LIBRARY}")

# FetchContent for vendored header-only libs and gtest
include(FetchContent)
set(FETCHCONTENT_QUIET FALSE)

FetchContent_Declare(spdlog
    GIT_REPOSITORY https://github.com/gabime/spdlog.git
    GIT_TAG v1.14.1
    GIT_SHALLOW TRUE)
FetchContent_Declare(cpp_httplib
    GIT_REPOSITORY https://github.com/yhirose/cpp-httplib.git
    GIT_TAG v0.18.1
    GIT_SHALLOW TRUE)

set(SPDLOG_BUILD_EXAMPLE OFF CACHE BOOL "" FORCE)
set(SPDLOG_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(HTTPLIB_REQUIRE_OPENSSL OFF CACHE BOOL "" FORCE)
set(HTTPLIB_REQUIRE_BROTLI OFF CACHE BOOL "" FORCE)
set(HTTPLIB_REQUIRE_ZLIB OFF CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(spdlog cpp_httplib)

# -----------------------------------------------------------------------------
# Subdirectories
# -----------------------------------------------------------------------------
add_subdirectory(src)

if(BUILD_TESTING)
    enable_testing()
    FetchContent_Declare(googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG v1.15.2
        GIT_SHALLOW TRUE)
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)

    add_subdirectory(tests)
endif()
