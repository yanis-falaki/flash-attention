#include <torch/extension.h>
#include <cuda_runtime.h>

namespace flash_attn {
    __global__ void flash_attn_kernel(float* out, const float* Q, const float* K, const float* V) {
        return;
    }


    torch::Tensor flash_attn_cuda(torch::Tensor Q, torch::Tensor K, torch::Tensor V) {
        int n = Q.sizes()[0];
        int d_q = Q.sizes()[1];
        int d_k = K.sizes()[1];
        int d_v = V.sizes()[1];

        torch::Tensor output = torch::zeros({n, d_v});
        
        int threads = 256;
        int blocks = 156;
        flash_attn_kernel<<<blocks, threads>>>(
            output.data_ptr<float>(),
            Q.data_ptr<float>(),
            K.data_ptr<float>(),
            V.data_ptr<float>()
        );
        return output;
    }
}

TORCH_LIBRARY(flash_attn, m) {
    m.def("flash_attn(Tensor Q, Tensor K, Tensor V) -> Tensor");
}

TORCH_LIBRARY_IMPL(flash_attn, CUDA, m) {
    m.impl("flash_attn", &flash_attn::flash_attn_cuda);
}

PYBIND11_MODULE(pybind_name, m) {}