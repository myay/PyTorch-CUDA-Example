#include <torch/extension.h>
#include <vector>

// CUDA forward declaration
std::vector<torch::Tensor> test_cpp_cuda(
    torch::Tensor input);

#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

std::vector<torch::Tensor> test_cpp(
    torch::Tensor input) {
  CHECK_INPUT(input);
  return test_cpp_cuda(input);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("test", &test_cpp, "TEST CPP");
}
