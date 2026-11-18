#include <torch/extension.h>
#include <vector>

std::vector<torch::Tensor> packed_forward_cuda(
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int64_t);
torch::Tensor packed_backward_cuda(
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int64_t);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &packed_forward_cuda);
    m.def("backward", &packed_backward_cuda);
}
