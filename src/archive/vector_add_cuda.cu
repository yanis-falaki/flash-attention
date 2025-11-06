#include <torch/extension.h>
#include <cuda_runtime.h>

namespace cuda_extension 
{
    __global__ void vector_add_kernel(float* out, const float* a, const float* b, int size) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < size) out[idx] = a[idx] + b[idx];
    }

    torch::Tensor vector_add_cuda(torch::Tensor a, torch::Tensor b) {
        auto output = torch::zeros_like(a);
        int threads = 256;
        int blocks = (a.numel() + threads - 1) / threads;
        vector_add_kernel<<<blocks, threads>>>(
            output.data_ptr<float>(),
            a.data_ptr<float>(),
            b.data_ptr<float>(),
            a.numel()
        );
        return output;
    }
}

TORCH_LIBRARY(cuda_extension, m) {
    m.def("vector_add(Tensor a, Tensor b) -> Tensor");
}

TORCH_LIBRARY_IMPL(cuda_extension, CUDA, m) {
    m.impl("vector_add", &cuda_extension::vector_add_cuda);
}

PYBIND11_MODULE(pybind_name, m) {}
