#pragma once
#include <memory>
#include <string>
#include <vector>

namespace cbw {

class InferenceBackend {
public:
    virtual ~InferenceBackend() = default;

    // input:  float32, shape [1,3,640,640] flattened (CHW, batch=1) — 1*3*640*640 values
    // output: float32, shape [1,6,8400] flattened — 1*6*8400 values
    virtual std::vector<float> run(const std::vector<float>& input) = 0;

    // Factory — currently only "onnx" is supported; "tensorrt" requires WITH_TENSORRT=ON
    static std::unique_ptr<InferenceBackend> create(const std::string& backend_type,
                                                    const std::string& model_path);
};

} // namespace cbw
