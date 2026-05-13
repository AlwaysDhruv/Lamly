#ifndef ADD_CUH
#define ADD_CUH

#include <vector>
#include <cuda_runtime.h>
#include <curand_kernel.h>

namespace Tensor
{
    __global__ void add_kernel(float *A, float *B, float *C, int N)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < N)
            C[i] = A[i] + B[i];
    }

    __global__ void random_kernel(float *data, curandState *states, int N)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;

        if (i < N)
        {
            curand_init(1234, i, 0, &states[i]);
            data[i] = curand_uniform(&states[i]);
        }
    }

    __global__ void matmul_kernel(float *A, float *B, float *C,
                                  int rowsA, int colsA, int colsB)
    {
        int row = blockIdx.y * blockDim.y + threadIdx.y;
        int col = blockIdx.x * blockDim.x + threadIdx.x;

        if (row < rowsA && col < colsB)
        {
            float sum = 0.0f;
            for (int k = 0; k < colsA; k++)
                sum += A[row * colsA + k] * B[k * colsB + col];

            C[row * colsB + col] = sum;
        }
    }

    __global__ void random3d_kernel(int *data, curandState *states,
                                   int d1, int d2, int d3, float dropout_rate)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;

        int N = d1 * d2 * d3;

        if (i < N)
        {
            curand_init(1234, i, 0, &states[i]);

            // generate 0 or 1
            float r = curand_uniform(&states[i]);
            data[i] = (r > dropout_rate) ? 1 : 0;
        }
    }

    __global__ void dropout_kernel(float *v1, int *v2, float *out,
                                   int N, float prob)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;

        if (i < N)
        {
            out[i] = (v1[i] * v2[i]) / prob;
        }
    }

    __global__ void layernorm_kernel(
        float *v,
        const float *gamma,
        const float *beta,
        int batch_size,
        int seq_len,
        int embed_size)
    {
        int idx =
            blockIdx.x * blockDim.x + threadIdx.x;

        int total_tokens =
            batch_size * seq_len;

        if (idx >= total_tokens || embed_size <= 0)
            return;

        int offset = idx * embed_size;

        // mean
        float mean = 0.0f;

        for (int k = 0; k < embed_size; k++)
        {
            mean += v[offset + k];
        }

        mean /= (float)embed_size;

        // variance
        float variance = 0.0f;

        for (int k = 0; k < embed_size; k++)
        {
            float diff =
                v[offset + k] - mean;

            variance += diff * diff;
        }

        variance /= (float)embed_size;

        // numerical stability
        float inv_std =
            rsqrtf(variance + 1e-5f);

        // normalize + affine
        for (int k = 0; k < embed_size; k++)
        {
            float normalized =
                (v[offset + k] - mean) * inv_std;

            v[offset + k] =
                gamma[k] * normalized + beta[k];
        }
    }

    __global__ void dot_kernel(
        float *A,
        float *B,
        float *C,
        int rowsA,
        int colsA,
        int colsB)
    {
        int row = blockIdx.y * blockDim.y + threadIdx.y;
        int col = blockIdx.x * blockDim.x + threadIdx.x;

        if (row < rowsA && col < colsB)
        {
            float sum = 0.0f;

            for (int k = 0; k < colsA; k++)
            {
                sum += A[row * colsA + k]
                     * B[k * colsB + col];
            }

            C[row * colsB + col] = sum;
        }
    }

    __global__ void matadd_kernel(
        float *A,
        float *B,
        float *C,
        int N)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;

        if (i < N)
        {
            C[i] = A[i] + B[i];
        }
    }

    inline void check_cuda(cudaError_t err)
    {
        if (err != cudaSuccess)
        {
            std::cout
                << "CUDA Error: "
                << cudaGetErrorString(err)
                << std::endl;

            exit(EXIT_FAILURE);
        }
    }

    inline void layer_norm(
        std::vector<std::vector<std::vector<float>>>& v,
        std::vector<float>& gamma,
        std::vector<float>& beta)
    {
        int batch_size = v.size();

        if (batch_size == 0)
            return;

        int seq_len = v[0].size();

        if (seq_len == 0)
            return;

        int embed_size = v[0][0].size();

        if (embed_size == 0)
            return;

        if (gamma.size() != embed_size ||
            beta.size() != embed_size)
        {
            std::cout
                << "gamma/beta size mismatch"
                << std::endl;

            return;
        }

        int total =
            batch_size * seq_len * embed_size;

        // flatten
        std::vector<float> flat(total);

        for (int i = 0; i < batch_size; i++)
        {
            for (int j = 0; j < seq_len; j++)
            {
                for (int k = 0; k < embed_size; k++)
                {
                    int idx =
                        (i * seq_len + j) * embed_size + k;

                    flat[idx] = v[i][j][k];
                }
            }
        }

        float *d_v = nullptr;
        float *d_gamma = nullptr;
        float *d_beta = nullptr;

        check_cuda(
            cudaMalloc(&d_v,
                       total * sizeof(float)));

        check_cuda(
            cudaMalloc(&d_gamma,
                       embed_size * sizeof(float)));

        check_cuda(
            cudaMalloc(&d_beta,
                       embed_size * sizeof(float)));

        check_cuda(
            cudaMemcpy(
                d_v,
                flat.data(),
                total * sizeof(float),
                cudaMemcpyHostToDevice));

        check_cuda(
            cudaMemcpy(
                d_gamma,
                gamma.data(),
                embed_size * sizeof(float),
                cudaMemcpyHostToDevice));

        check_cuda(
            cudaMemcpy(
                d_beta,
                beta.data(),
                embed_size * sizeof(float),
                cudaMemcpyHostToDevice));

        int total_tokens =
            batch_size * seq_len;

        int threads = 256;

        int blocks =
            (total_tokens + threads - 1) / threads;

        layernorm_kernel<<<blocks, threads>>>(
            d_v,
            d_gamma,
            d_beta,
            batch_size,
            seq_len,
            embed_size);

        check_cuda(cudaGetLastError());
        check_cuda(cudaDeviceSynchronize());

        check_cuda(
            cudaMemcpy(
                flat.data(),
                d_v,
                total * sizeof(float),
                cudaMemcpyDeviceToHost));

        cudaFree(d_v);
        cudaFree(d_gamma);
        cudaFree(d_beta);

        // reshape back
        for (int i = 0; i < batch_size; i++)
        {
            for (int j = 0; j < seq_len; j++)
            {
                for (int k = 0; k < embed_size; k++)
                {
                    int idx =
                        (i * seq_len + j) * embed_size + k;

                    v[i][j][k] = flat[idx];
                }
            }
        }
    }

    inline std::vector<std::vector<float>>
    random(int N1, int N2)
    {
        std::vector<std::vector<float>> matrix(
            N1,
            std::vector<float>(N2));

        // host memory
        std::vector<float> h_data(N1 * N2);

        // fill random values
        for (int i = 0; i < N1 * N2; i++)
        {
            h_data[i] =
                static_cast<float>(rand()) / RAND_MAX;
        }

        // device memory
        float *d_data = nullptr;

        cudaError_t err;

        err = cudaMalloc(
            &d_data,
            N1 * N2 * sizeof(float));

        if (err != cudaSuccess)
        {
            std::cout
                << "cudaMalloc failed: "
                << cudaGetErrorString(err)
                << std::endl;

            exit(EXIT_FAILURE);
        }

        err = cudaMemcpy(
            d_data,
            h_data.data(),
            N1 * N2 * sizeof(float),
            cudaMemcpyHostToDevice);

        if (err != cudaSuccess)
        {
            std::cout
                << "cudaMemcpy H2D failed: "
                << cudaGetErrorString(err)
                << std::endl;

            exit(EXIT_FAILURE);
        }

        // copy back
        err = cudaMemcpy(
            h_data.data(),
            d_data,
            N1 * N2 * sizeof(float),
            cudaMemcpyDeviceToHost);

        if (err != cudaSuccess)
        {
            std::cout
                << "cudaMemcpy D2H failed: "
                << cudaGetErrorString(err)
                << std::endl;

            exit(EXIT_FAILURE);
        }

        cudaFree(d_data);

        // reshape correctly
        for (int i = 0; i < N1; i++)
        {
            for (int j = 0; j < N2; j++)
            {
                int index = i * N2 + j;

                matrix[i][j] = h_data[index];
            }
        }

        return matrix;
    }

    inline std::vector<float> matadd(const std::vector<float>& A,
                                  const std::vector<float>& B)
    {
        int N = A.size();

        float *d_A, *d_B, *d_C;

        cudaMalloc(&d_A, N*sizeof(float));
        cudaMalloc(&d_B, N*sizeof(float));
        cudaMalloc(&d_C, N*sizeof(float));

        cudaMemcpy(d_A, A.data(), N*sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, B.data(), N*sizeof(float), cudaMemcpyHostToDevice);

        int threads = 256;
        int blocks = (N + threads - 1) / threads;

        add_kernel<<<blocks, threads>>>(d_A, d_B, d_C, N);

        std::vector<float> C(N);
        cudaMemcpy(C.data(), d_C, N*sizeof(float), cudaMemcpyDeviceToHost);

        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);

        return C;
    }
 
    inline std::vector<std::vector<float>> matmul(
        const std::vector<std::vector<float>>& A,
        const std::vector<std::vector<float>>& B)
    {
        int rowsA = A.size();
        int colsA = A[0].size();
        int rowsB = B.size();
        int colsB = B[0].size();

        if (colsA != rowsB)
            throw std::runtime_error("Matrix size mismatch");

        // Flatten A and B
        std::vector<float> flatA(rowsA * colsA);
        std::vector<float> flatB(rowsB * colsB);

        for (int i = 0; i < rowsA; i++)
            for (int j = 0; j < colsA; j++)
                flatA[i * colsA + j] = A[i][j];

        for (int i = 0; i < rowsB; i++)
            for (int j = 0; j < colsB; j++)
                flatB[i * colsB + j] = B[i][j];

        // --- GPU part ---
        float *d_A, *d_B, *d_C;

        cudaMalloc(&d_A, flatA.size() * sizeof(float));
        cudaMalloc(&d_B, flatB.size() * sizeof(float));
        cudaMalloc(&d_C, rowsA * colsB * sizeof(float));

        cudaMemcpy(d_A, flatA.data(), flatA.size()*sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, flatB.data(), flatB.size()*sizeof(float), cudaMemcpyHostToDevice);

        dim3 threads(16, 16);
        dim3 blocks((colsB + 15)/16, (rowsA + 15)/16);

        matmul_kernel<<<blocks, threads>>>(d_A, d_B, d_C, rowsA, colsA, colsB);

        std::vector<float> flatC(rowsA * colsB);
        cudaMemcpy(flatC.data(), d_C, flatC.size()*sizeof(float), cudaMemcpyDeviceToHost);

        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);

        // Convert back to 2D
        std::vector<std::vector<float>> C(rowsA, std::vector<float>(colsB));

        for (int i = 0; i < rowsA; i++)
            for (int j = 0; j < colsB; j++)
                C[i][j] = flatC[i * colsB + j];

        return C;
    }

    inline std::vector<std::vector<std::vector<int>>>
    dropout_mask(int d1, int d2, int d3, float dropout_rate)
    {
        int N = d1 * d2 * d3;

        int *d_data;
        curandState *d_states;

        cudaMalloc(&d_data, N * sizeof(int));
        cudaMalloc(&d_states, N * sizeof(curandState));

        int threads = 256;
        int blocks = (N + threads - 1) / threads;

        random3d_kernel<<<blocks, threads>>>(d_data, d_states, d1, d2, d3, dropout_rate);

        std::vector<int> flat(N);
        cudaMemcpy(flat.data(), d_data, N * sizeof(int), cudaMemcpyDeviceToHost);

        cudaFree(d_data);
        cudaFree(d_states);

        // reshape to 3D
        std::vector<std::vector<std::vector<int>>> result(
            d1, std::vector<std::vector<int>>(d2, std::vector<int>(d3)));

        for (int i = 0; i < d1; i++)
            for (int j = 0; j < d2; j++)
                for (int k = 0; k < d3; k++)
                {
                    int idx = (i * d2 + j) * d3 + k;
                    result[i][j][k] = flat[idx];
                }

        return result;
    }
    
    inline std::vector<std::vector<std::vector<float>>> dropout(
        const std::vector<std::vector<std::vector<float>>>& v1,
        const std::vector<std::vector<std::vector<int>>>& v2,
        float prob)
    {
        int d1 = v1.size();
        int d2 = v1[0].size();
        int d3 = v1[0][0].size();

        int N = d1 * d2 * d3;

        // flatten
        std::vector<float> flat1(N);
        std::vector<int> flat2(N);

        for (int i = 0; i < d1; i++)
            for (int j = 0; j < d2; j++)
                for (int k = 0; k < d3; k++)
                {
                    int idx = (i * d2 + j) * d3 + k;
                    flat1[idx] = v1[i][j][k];
                    flat2[idx] = v2[i][j][k];
                }

        // device memory
        float *d_v1, *d_out;
        int *d_v2;

        cudaMalloc(&d_v1, N * sizeof(float));
        cudaMalloc(&d_v2, N * sizeof(int));
        cudaMalloc(&d_out, N * sizeof(float));

        cudaMemcpy(d_v1, flat1.data(), N*sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_v2, flat2.data(), N*sizeof(int), cudaMemcpyHostToDevice);

        int threads = 256;
        int blocks = (N + threads - 1) / threads;

        dropout_kernel<<<blocks, threads>>>(d_v1, d_v2, d_out, N, prob);

        std::vector<float> flat_out(N);
        cudaMemcpy(flat_out.data(), d_out, N*sizeof(float), cudaMemcpyDeviceToHost);

        cudaFree(d_v1);
        cudaFree(d_v2);
        cudaFree(d_out);

        // reshape back to 3D
        std::vector<std::vector<std::vector<float>>> ans(
            d1, std::vector<std::vector<float>>(d2, std::vector<float>(d3)));

        for (int i = 0; i < d1; i++)
            for (int j = 0; j < d2; j++)
                for (int k = 0; k < d3; k++)
                {
                    int idx = (i * d2 + j) * d3 + k;
                    ans[i][j][k] = flat_out[idx];
                }

        return ans;
    }
    
    vector<vector<vector<vector<float>>>> head_spliting(vector<vector<vector<float>>>& v1, int head_size)
    {
        int num_seq = v1.size();
        int seq_len = v1[0].size();
        int embed_size = v1[0][0].size();
        int head_dim = embed_size / head_size;

        vector<vector<vector<vector<float>>>> h;
        h.reserve(num_seq);

        for (int i = 0; i < num_seq; ++i)
        {
            vector<vector<vector<float>>> temp2;
            temp2.reserve(seq_len);
            for (int j = 0; j < seq_len; ++j)
            {
                vector<vector<float>> temp1;
                temp1.reserve(embed_size);      
                for (int k = 0; k < embed_size; k+=head_dim)
                {
                    vector<float> temp;
                    temp.reserve(head_dim);

                    for (int l = k; l < k + head_dim; ++l) temp.push_back(v1[i][j][l]);
                    temp1.push_back(temp);
                }
                temp2.push_back(temp1);
            }
            h.push_back(temp2);
        }
        return h;
    }
    
    vector<vector<vector<float>>> transpose(vector<vector<vector<float>>>& v)
    {
        int dim1 = v.size();
        int dim2 = v[0].size();
        int dim3 = v[0][0].size();

        vector<vector<vector<float>>> trans;
        trans.reserve(dim2);

        for (int i = 0; i < dim2; ++i)
        {
            vector<vector<float>> temp;
            temp.reserve(dim1);

            for (int j = 0; j < dim1; ++j)
            {
                vector<float> temp1;
                temp1.reserve(dim3);

                for (int k = 0; k < dim3; ++k) temp1.push_back(v[j][i][k]);
                
                temp.push_back(temp1);
            }
            trans.push_back(temp);
        }        
        return trans;
    }

    vector<vector<float>> transpose(vector<vector<float>>& v)
    {
        int dim1 = v.size();
        int dim2 = v[0].size();

        vector<vector<float>> trans;
        trans.reserve(dim1);

        for (int i = 0; i < dim2; ++i)
        {
            vector<float> temp;
            temp.reserve(dim2);
            for (int j = 0; j < dim1; ++j) temp.push_back(v[j][i]);
            trans.push_back(temp);
        }
        
        return trans;
    }
        
    void casual_mask(vector<vector<float>>& attension_score, float scale)
    {
        for (size_t i = 0; i < attension_score.size(); ++i)
        {
            for (size_t k = 0; k < attension_score[0].size(); ++k)
            {
                if (k > i) attension_score[i][k] = -1e9f;
                else attension_score[i][k] *= scale;
            }
        }
    }

    inline std::vector<std::vector<float>> dot_product(
        const std::vector<std::vector<float>>& A,
        const std::vector<std::vector<float>>& B)
    {
        int rowsA = A.size();
        int colsA = A[0].size();

        int rowsB = B.size();
        int colsB = B[0].size();

        if (colsA != rowsB)
            throw std::runtime_error("Shape mismatch");

        // flatten
        std::vector<float> flatA(rowsA * colsA);
        std::vector<float> flatB(rowsB * colsB);

        for (int i = 0; i < rowsA; i++)
            for (int j = 0; j < colsA; j++)
                flatA[i * colsA + j] = A[i][j];

        for (int i = 0; i < rowsB; i++)
            for (int j = 0; j < colsB; j++)
                flatB[i * colsB + j] = B[i][j];

        // GPU memory
        float *d_A, *d_B, *d_C;

        cudaMalloc(&d_A, flatA.size() * sizeof(float));
        cudaMalloc(&d_B, flatB.size() * sizeof(float));
        cudaMalloc(&d_C, rowsA * colsB * sizeof(float));

        cudaMemcpy(d_A, flatA.data(),
                   flatA.size() * sizeof(float),
                   cudaMemcpyHostToDevice);

        cudaMemcpy(d_B, flatB.data(),
                   flatB.size() * sizeof(float),
                   cudaMemcpyHostToDevice);

        dim3 threads(16, 16);

        dim3 blocks(
            (colsB + 15) / 16,
            (rowsA + 15) / 16
        );

        dot_kernel<<<blocks, threads>>>(
            d_A, d_B, d_C,
            rowsA, colsA, colsB
        );

        // copy back
        std::vector<float> flatC(rowsA * colsB);

        cudaMemcpy(flatC.data(),
                   d_C,
                   flatC.size() * sizeof(float),
                   cudaMemcpyDeviceToHost);

        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);

        // reshape to 2D
        std::vector<std::vector<float>> C(
            rowsA,
            std::vector<float>(colsB)
        );

        for (int i = 0; i < rowsA; i++)
            for (int j = 0; j < colsB; j++)
                C[i][j] = flatC[i * colsB + j];

        return C;
    }

    vector<float> merge_head(vector<vector<float>>& v)
    {
        int cols = v.size();
        int rows = v[0].size();

        vector<float> ans;
        ans.reserve(cols * rows);
        for (int i = 0; i < cols; ++i) for (int j = 0; j < rows; ++j) ans.push_back(v[i][j]);
        return ans;
    }

    inline std::vector<std::vector<std::vector<float>>> matadd(
        std::vector<std::vector<std::vector<float>>>& v1,
        std::vector<std::vector<std::vector<float>>>& v2)
    {
        int dim1 = v1.size();
        int dim2 = v1[0].size();
        int dim3 = v1[0][0].size();

        int N = dim1 * dim2 * dim3;

        // flatten
        std::vector<float> flatA(N);
        std::vector<float> flatB(N);

        for (int i = 0; i < dim1; i++)
            for (int j = 0; j < dim2; j++)
                for (int k = 0; k < dim3; k++)
                {
                    int idx = (i * dim2 + j) * dim3 + k;

                    flatA[idx] = v1[i][j][k];
                    flatB[idx] = v2[i][j][k];
                }

        // GPU memory
        float *d_A, *d_B, *d_C;

        cudaMalloc(&d_A, N * sizeof(float));
        cudaMalloc(&d_B, N * sizeof(float));
        cudaMalloc(&d_C, N * sizeof(float));

        cudaMemcpy(d_A,
                   flatA.data(),
                   N * sizeof(float),
                   cudaMemcpyHostToDevice);

        cudaMemcpy(d_B,
                   flatB.data(),
                   N * sizeof(float),
                   cudaMemcpyHostToDevice);

        int threads = 256;
        int blocks = (N + threads - 1) / threads;

        matadd_kernel<<<blocks, threads>>>(d_A, d_B, d_C, N);

        // copy back
        std::vector<float> flatC(N);

        cudaMemcpy(flatC.data(),
                   d_C,
                   N * sizeof(float),
                   cudaMemcpyDeviceToHost);

        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);

        // reshape back
        std::vector<std::vector<std::vector<float>>> ans(
            dim1,
            std::vector<std::vector<float>>(
                dim2,
                std::vector<float>(dim3)
            )
        );

        for (int i = 0; i < dim1; i++)
            for (int j = 0; j < dim2; j++)
                for (int k = 0; k < dim3; k++)
                {
                    int idx = (i * dim2 + j) * dim3 + k;
                    ans[i][j][k] = flatC[idx];
                }

        return ans;
    }

}

#endif