#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

template <typename scalar_t>
__global__ void test_cpp_cuda_kernel(
    torch::PackedTensorAccessor<scalar_t,1,torch::RestrictPtrTraits,size_t> input) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
    input[c] += 1;
}

std::vector<torch::Tensor> test_cpp_cuda(torch::Tensor input) {
  const int input_size = input.size(0);
  const int threads = 256;
  const int batch_size = 1;
  const dim3 blocks((input_size + threads - 1) / threads, batch_size);

  AT_DISPATCH_FLOATING_TYPES(input.type(), "test_cpp_cuda", ([&] {
    test_cpp_cuda_kernel<scalar_t><<<blocks, threads>>>(
        input.packed_accessor<scalar_t,1,torch::RestrictPtrTraits,size_t>());
  }));

  return {input};
}
