#include "onnx_backend.hpp"
#include <spdlog/spdlog.h>
#include <stdexcept>

namespace cbw {

OnnxBackend::OnnxBackend(const std::string& model_path)
    : env_(ORT_LOGGING_LEVEL_WARNING, "catbowlwatch"),
      session_(env_, model_path.c_str(), Ort::SessionOptions{}) {
    input_name_  = session_.GetInputNameAllocated(0, allocator_).get();
    output_name_ = session_.GetOutputNameAllocated(0, allocator_).get();
    spdlog::info("OnnxBackend loaded: {} (input='{}' output='{}')",
                 model_path, input_name_, output_name_);
}

std::vector<float> OnnxBackend::run(const std::vector<float>& input) {
    if (static_cast<int64_t>(input.size()) != INPUT_SIZE)
        throw std::runtime_error("OnnxBackend::run: unexpected input size");

    constexpr std::array<int64_t, 4> in_shape  = {1, 3, 640, 640};
    constexpr std::array<int64_t, 3> out_shape = {1, 6, 8400};

    auto mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value in_tensor = Ort::Value::CreateTensor<float>(
        mem, const_cast<float*>(input.data()), input.size(),
        in_shape.data(), in_shape.size());

    const char* in_names[]  = {input_name_.c_str()};
    const char* out_names[] = {output_name_.c_str()};

    auto outputs = session_.Run(Ort::RunOptions{nullptr},
                                in_names, &in_tensor, 1,
                                out_names, 1);

    const float* raw = outputs[0].GetTensorData<float>();
    (void)out_shape;
    return std::vector<float>(raw, raw + OUTPUT_SIZE);
}

std::unique_ptr<InferenceBackend> InferenceBackend::create(const std::string& backend_type,
                                                           const std::string& model_path) {
    if (backend_type == "onnx")
        return std::make_unique<OnnxBackend>(model_path);
    throw std::runtime_error("Unknown INFERENCE_BACKEND: " + backend_type +
                             ". Supported: onnx");
}

} // namespace cbw
