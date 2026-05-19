#!/usr/bin/env bash
# CatBowlWatch — WSL2 Ubuntu development environment bootstrap (Phase 3).
#
# Run this inside WSL Ubuntu after `wsl --install` completes:
#     cd /mnt/c/Users/<you>/Documents/SwProjects/cat-bowl-watch/CatBowlWatch
#     bash scripts/setup_wsl_dev.sh
#
# Installs the C++ inference service toolchain:
#   - build-essential (gcc/g++), cmake, pkg-config, git, curl
#   - libopencv-dev          (Frame Capture + image handling)
#   - libcurl4-openssl-dev   (Phase 4 Telegram notifier)
#   - ONNX Runtime C++ release tarball, extracted to $HOME
#
# Vendored via CMake FetchContent (no system install):
#   - spdlog        (header-only logging)
#   - cpp-httplib   (header-only HTTP server)
#   - googletest    (test framework)
#
# TensorRT is intentionally NOT installed here — laptop builds use
# -DWITH_TENSORRT=OFF (default). Jetson builds happen on the Jetson with
# JetPack-provided TensorRT.

set -euo pipefail

ORT_VERSION="${ORT_VERSION:-1.20.1}"
ORT_TARBALL="onnxruntime-linux-x64-${ORT_VERSION}.tgz"
ORT_URL="https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/${ORT_TARBALL}"
ORT_DIR="${HOME}/onnxruntime-linux-x64-${ORT_VERSION}"

echo ">> CatBowlWatch dev bootstrap"
echo "   ONNX Runtime: ${ORT_VERSION}"
echo "   ORT install dir: ${ORT_DIR}"
echo

if [ "$(uname)" != "Linux" ]; then
    echo "Error: this script targets Linux (run inside WSL Ubuntu)." >&2
    exit 1
fi

echo ">> apt install build toolchain..."
sudo apt-get update -qq
sudo apt-get install -y \
    build-essential cmake pkg-config git curl ca-certificates \
    libopencv-dev libcurl4-openssl-dev

if [ ! -d "${ORT_DIR}" ]; then
    echo ">> Downloading ONNX Runtime ${ORT_VERSION}..."
    tmp=$(mktemp -d)
    trap "rm -rf ${tmp}" EXIT
    curl -L --fail -o "${tmp}/${ORT_TARBALL}" "${ORT_URL}"
    tar -xzf "${tmp}/${ORT_TARBALL}" -C "${HOME}/"
else
    echo ">> ONNX Runtime already installed at ${ORT_DIR}"
fi

cat <<EOF

================================================================================
Setup complete.

ONNX Runtime is at:  ${ORT_DIR}

Next steps to verify the toolchain:

    export ONNXRUNTIME_ROOT="${ORT_DIR}"
    cd inference
    cmake -B build -DCMAKE_BUILD_TYPE=Release -DWITH_TENSORRT=OFF
    cmake --build build -j\$(nproc)
    ctest --test-dir build --output-on-failure
    ./build/catbowlwatch_smoke

Expected output from the smoke binary:
    CatBowlWatch toolchain check
    OpenCV: 4.x.x
    spdlog: 1.x.x
    ONNX Runtime: ${ORT_VERSION}
    OK

To persist ONNXRUNTIME_ROOT across sessions:
    echo 'export ONNXRUNTIME_ROOT=${ORT_DIR}' >> ~/.bashrc

EOF
