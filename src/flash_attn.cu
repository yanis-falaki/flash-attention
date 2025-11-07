#include <torch/extension.h>
#include <cuda_runtime.h>

namespace flash_attn {
    __global__ void flash_attn_kernel(float* O, const float* Q, const float* K, const float* V,
        float* l, float* m, const int d, const int N, const int Bc, const int Br, const int Tc, const int Tr,
        const float softmax_scale) {

        // One tile is of size Bc x d, one tile for each matrix.
        extern __shared__ float sram[];
        int tile_size = Bc * d;
        float* Qi = sram;
        float* Kj = &sram[tile_size];
        float* Vj = &sram[tile_size*2];
        float* Sij = &sram[tile_size*3];

        int bx = blockIdx.x;
        int hx = blockIdx.y;
        int tx = threadIdx.x; // Corresponds to the row in the current head

        // Get start of head in qkv by skipping past batches and heads.
        int global_batch_idx = bx * gridDim.z * blockIdx.x * d;
        int local_head_idx = hx * blockDim.x * d;
        int global_head_idx = global_batch_idx + local_head_idx;

        // l, m offset
        int lm_offset = (bx * gridDim.y * N) + (hx * N) + tx;

        for (int j = 0; j < Tc; ++j) {
            // Cooperatively load Kj, Vj, each thread loads a row
            for (int x = 0; x < d; ++x) {
                Kj[tx*d + x] = K[global_head_idx + (tile_size * j) + (tx * d) + x];
                Vj[tx*d + x] = V[global_head_idx + (tile_size * j) + (tx * d) + x];
            }

            for (int i = 0; i < Tr; ++i) {
                // Load Qi, Oi, li, mi from HBM to on-chip SRAM
                for (int x = 0; x < d; ++x) {
                    Qi[tx*d + x] = Q[global_head_idx + (tile_size * i) + (tx * d) + x];
                }

                // Compute dot product of Qi and Kj^T, keep track of rowmax
                float mij = -INFINITY;
                for (int kt_c = 0; kt_c < Br; ++kt_c) {
                    int dotprod = 0;
                    for (int x = 0; x < d; ++x) {
                        dotprod += Qi[tx*d + x]*Kj[kt_c*d + x];
                    }
                    dotprod = dotprod * softmax_scale;
                    Sij[tx*Bc + kt_c] = dotprod;
                    if (dotprod > mij) mij = dotprod;
                }

                // Compute Pij = exp(Sij - mij) and l_ij = rowsum(Pij)
                float lij = 0;
                for (int x = 0; x < Bc; ++x) {
                    Sij[tx*Bc + x] = __expf(Sij[tx*Bc + x] - mij);
                    lij += Sij[tx*Bc + x];
                }

                // Compute mi_new = max(mi, mij)
                float mi_prev = m[lm_offset + i*Br];
                float mi_new = max(mi_prev, mij);

                // Compute li_new = e^(mi - mi_new)*li + e^(mij - mi_new)*lij
                float expOld = __expf(mi_prev - mi_new);
                float expNew = __expf(mij - mi_new);

                float li_prev = l[lm_offset + i*Br];
                float li_new = expOld*li_prev + expNew*lij;

                // Write Oi = elemwise prod of 1/li_new * li * e^(mi - mij) * Oi + e^(mij - mi_new)*Pij*Vj
                for (int x = 0; x < d; ++x) {
                    float pv = 0; // Pij * Vj
                    for (int y = 0; i < Bc; ++y) {
                        pv += Sij[(Bc * tx) + y] * Vj[(y * d) + x];
                    }
                    O
                }

                // Write mi_new and li_new to HBM
                m[lm_offset + i*Br] = mi_new;
                l[lm_offset + i*Br] = li_new;
            }
        }
    }

    // For now we will assume that d = d_q = d_k = d_v
    torch::Tensor flash_attn_cuda(torch::Tensor Q, torch::Tensor K, torch::Tensor V) {
        int B = Q.sizes()[0];
        int nh = Q.sizes()[1];
        int N = Q.sizes()[2];
        int d = Q.sizes()[3];

        printf("N: %d, d: %d\n", N , d);

        assert((Q.sizes()[0] == K.sizes()[0]) && (Q.sizes()[0] == V.sizes()[0]));
        assert((Q.sizes()[1] == K.sizes()[1]) && (Q.sizes()[1] == V.sizes()[1]));

        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        printf("Number of SMs: %d SMs\n", prop.multiProcessorCount);
        printf("Total shared memory per SM: %d bytes\n", prop.sharedMemPerMultiprocessor);
        printf("Max amount of blocks per SM: %d blocks\n", prop.maxBlocksPerMultiProcessor);
        printf("Total shared memory per block: %d bytes\n", prop.sharedMemPerBlock);

        int M = prop.sharedMemPerMultiprocessor;

        const int Bc = 32;
        const int Br = 32;

        int Tr = ceil((float)N/Br);
        int Tc = ceil((float)N/Bc);

        torch::Tensor O = torch::zeros({N, d});
        torch::Tensor l = torch::zeros({N});
        torch::Tensor m = torch::full({N}, -INFINITY, torch::kFloat);
        torch::Device device(torch::kCUDA);
        l.to(device);
        m.to(device);

        float softmax_scale = 1/sqrt(d);
        
        dim3 blocks = (B, nh);
        dim3 threads = (Bc);

        flash_attn_kernel<<<blocks, threads>>>(
            O.data_ptr<float>(),
            Q.data_ptr<float>(),
            K.data_ptr<float>(),
            V.data_ptr<float>(),
            l.data_ptr<float>(),
            m.data_ptr<float>(),
            d,
            Bc,
            Br,
            Tc,
            Tr,
            softmax_scale
        );

        return O;
    }
}

TORCH_LIBRARY(flash_attn, m) {
    m.def("flash_attn(Tensor Q, Tensor K, Tensor V) -> Tensor");
}

TORCH_LIBRARY_IMPL(flash_attn, CUDA, m) {
    m.impl("flash_attn", &flash_attn::flash_attn_cuda);
}

PYBIND11_MODULE(pybind_name, m) {}