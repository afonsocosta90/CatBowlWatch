#!/usr/bin/env bash
# CatBowlWatch — macOS (Apple Silicon / Intel) development environment bootstrap (Phase 3).
#
# Run this on macOS after cloning the repo:
#     bash scripts/setup_macos_dev.sh
#
# Installs the C++ inference service toolchain:
#   - cmake, pkg-config, git (via Homebrew)
#   - opencv (via Homebrew)
#   - curl (already present on macOS)
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

# Detect architecture
ARCH=$(uname -m)
if [ "${ARCH}" = "arm64" ]; then
    ORT_PLATFORM="osx-arm64"
else
    ORT_PLATFORM="osx-x86_64"
fi

ORT_TARBALL="onnxruntime-${ORT_PLATFORM}-${ORT_VERSION}.tgz"
ORT_URL="https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/${ORT_TARBALL}"
ORT_DIR="${HOME}/onnxruntime-${ORT_PLATFORM}-${ORT_VERSION}"

echo ">> CatBowlWatch macOS dev bootstrap"
echo "   ONNX Runtime: ${ORT_VERSION}"
echo "   Platform: ${ORT_PLATFORM}"
echo "   ORT install dir: ${ORT_DIR}"
echo

if [ "$(uname)" != "Darwin" ]; then
    echo "Error: this script targets macOS (use scripts/setup_wsl_dev.sh on Linux)." >&2
    exit 1
fi

if ! command -v brew &>/dev/null; then
    echo "Error: Homebrew not found. Install from https://brew.sh first." >&2
    exit 1
fi

echo ">> brew install cmake opencv..."
brew install cmake opencv

if [ ! -d "${ORT_DIR}" ]; then
    echo ">> Downloading ONNX Runtime ${ORT_VERSION} for ${ORT_PLATFORM}..."
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
    cmake --build build -j\$(sysctl -n hw.logicalcpu)
    ctest --test-dir build --output-on-failure
    ./build/src/catbowlwatch_smoke

Expected output from the smoke binary:
    CatBowlWatch toolchain check
    OpenCV: 4.x.x
    spdlog: 1.x.x
    ONNX Runtime: ${ORT_VERSION}
    OK

To persist ONNXRUNTIME_ROOT across sessions:
    echo 'export ONNXRUNTIME_ROOT=${ORT_DIR}' >> ~/.zshrc

EOF
