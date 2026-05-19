#pragma once
#include "inference_backend.hpp"
#include <onnxruntime_cxx_api.h>
#include <string>

namespace cbw {

class OnnxBackend : public InferenceBackend {
public:
    explicit OnnxBackend(const std::string& model_path);
    std::vector<float> run(const std::vector<float>& input) override;

private:
    Ort::Env                         env_;
    Ort::Session                     session_;
    Ort::AllocatorWithDefaultOptions allocator_;
    std::string                      input_name_;
    std::string                      output_name_;

    static constexpr int64_t INPUT_SIZE  = 1LL * 3 * 640 * 640;
    static constexpr int64_t OUTPUT_SIZE = 1LL * 6 * 8400;
};

} // namespace cbw
