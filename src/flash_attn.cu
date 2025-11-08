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
        int global_batch_idx = bx * gridDim.y * N * d;
        int local_head_idx = hx * N * d;

        // Global_head_idx is the same as the start of lm_offset
        int global_head_idx = global_batch_idx + local_head_idx;

        // In each inner loop lm_offset will be offset by an additional Br*i 
        int lm_offset = (bx * gridDim.y * N) + (hx * N) + tx;

        for (int j = 0; j < Tc; ++j) {
            // Cooperatively load Kj, Vj, each thread loads a row
            // No need for an else branch as rows out of bounds never get used in calcs.
            if (Bc*j + tx < N)
                for (int x = 0; x < d; ++x) {
                    int Kij_offset = global_head_idx + (tile_size * j) + (tx * d) + x;
                    Kj[tx*d + x] = K[Kij_offset];
                    Vj[tx*d + x] = V[Kij_offset];
                }

            // Ensure all K^T columns are loaded
            __syncthreads();

            // This is used for loops over Kj^T later to make sure we're not accessing OOB elements
            int valid_kjt_bounds = min(Bc, N - Bc*j);

            for (int i = 0; i < Tr; ++i) {
                // If we get to a row that doesn't exist, simply don't compute it.
                if (i * Br + tx >= N)
                    continue;

                // Load Qi, li, mi from HBM to on-chip SRAM (Paper says SRAM, but it could be loaded locally)
                for (int x = 0; x < d; ++x) {
                    Qi[tx*d + x] = Q[global_head_idx + (tile_size * i) + (tx * d) + x];
                }
                // No need to sync Qi as each row in Qi is accessed by a single thread

                float li = l[lm_offset + Br*i];
                float mi = m[lm_offset + Br*i];

                // Compute dot product of Qi and Kj^T, keep track of rowmax
                float mij = -INFINITY;
                for (int kt_c = 0; kt_c < valid_kjt_bounds; ++kt_c) {
                    float dotprod = 0.0f;
                    for (int x = 0; x < d; ++x) {
                        dotprod += Qi[tx*d + x]*Kj[kt_c*d + x];
                    }
                    dotprod = dotprod * softmax_scale;
                    Sij[tx*Bc + kt_c] = dotprod;
                    if (dotprod > mij) mij = dotprod;
                }

                // Compute Pij = exp(Sij - mij) and l_ij = rowsum(Pij)
                float lij = 0;
                for (int x = 0; x < valid_kjt_bounds; ++x) {
                    Sij[tx*Bc + x] = __expf(Sij[tx*Bc + x] - mij);
                    lij += Sij[tx*Bc + x];
                }

                // Compute mi_new = max(mi, mij)
                float mi_new = max(mi, mij);

                // Compute li_new = e^(mi - mi_new)*li + e^(mij - mi_new)*lij
                float expOld = __expf(mi - mi_new);
                float expNew = __expf(mij - mi_new);

                float li_new = expOld*li + expNew*lij;

                // Write Oi = elemwise prod of 1/li_new * li * e^(mi - mij) * Oi + e^(mij - mi_new)*Pij*Vj
                for (int x = 0; x < d; ++x) {
                    float pvij = 0; // Pij * Vij
                    for (int y = 0; y < valid_kjt_bounds; ++y) {
                        pvij += Sij[(Bc * tx) + y] * Vj[(y * d) + x];
                    }
                    int O_offset = global_head_idx + (tile_size * i) + (tx * d) + x;
                    O[O_offset] = (1.0f/li_new) * (li*expOld*O[O_offset] + expNew*pvij);
                }

                // Write mi_new and li_new to HBM
                m[lm_offset + i*Br] = mi_new;
                l[lm_offset + i*Br] = li_new;
            }
        }
    }

    // Incoming tensor: B x H x N x D
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

        // Currently Bc and Br must be identical
        const int Bc = 32;
        const int Br = 32;

        int Tr = ceil((float)N/Br);
        int Tc = ceil((float)N/Bc);

        torch::Device device(torch::kCUDA);
        torch::Tensor O = torch::zeros({B, nh, N, d}).to(device);
        torch::Tensor l = torch::zeros({B, nh, N}).to(device);
        torch::Tensor m = torch::full({B, nh, N}, -INFINITY, torch::kFloat).to(device);

        float softmax_scale = 1.0/sqrt((float)d);
        
        dim3 blocks = (B, nh);
        dim3 threads = (Bc);

        int SRAM_size = ((Bc*d * 3) + (Br*Bc)) * sizeof(float);

        flash_attn_kernel<<<blocks, threads, SRAM_size>>>(
            O.data_ptr<float>(),
            Q.data_ptr<float>(),
            K.data_ptr<float>(),
            V.data_ptr<float>(),
            l.data_ptr<float>(),
            m.data_ptr<float>(),
            d,
            N,
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