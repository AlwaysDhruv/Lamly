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

    __global__ void dropout_mask_kernel(
        float *mask,
        curandState *states,
        int rows,
        int cols,
        float dropout_rate)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;

        int N = rows * cols;

        if (i < N)
        {
            curand_init(1234, i, 0, &states[i]);

            float r = curand_uniform(&states[i]);

            mask[i] =
                (r > dropout_rate)
                ? 1.0f
                : 0.0f;
        }
    }

    __global__ void dropout_kernel(
        float *A,
        float *Mask,
        float *Out,
        int N,
        float prob)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;

        if (i < N)
        {
            Out[i] = (A[i] * Mask[i]) / prob;
        }
    }

    __global__ void layernorm_kernel(
        float *data,
        float *gamma,
        float *beta,
        int seq_len,
        int embed_size)
    {
        int row = blockIdx.x * blockDim.x + threadIdx.x;

        if (row < seq_len)
        {
            int offset = row * embed_size;

            // mean
            float mean = 0.0f;

            for (int k = 0; k < embed_size; k++)
                mean += data[offset + k];

            mean /= embed_size;

            // variance
            float variance = 0.0f;

            for (int k = 0; k < embed_size; k++)
            {
                float diff = data[offset + k] - mean;
                variance += diff * diff;
            }

            variance /= embed_size;

            float std = sqrtf(variance + 1e-5f);

            // normalize + affine
            for (int k = 0; k < embed_size; k++)
            {
                data[offset + k] =
                    gamma[k] *
                    ((data[offset + k] - mean) / std)
                    + beta[k];
            }
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
        std::vector<std::vector<float>>& v,
        std::vector<float>& gamma,
        std::vector<float>& beta)
    {
        int seq_len = v.size();
        int embed_size = v[0].size();

        int N = seq_len * embed_size;

        // flatten
        std::vector<float> flat(N);

        for (int i = 0; i < seq_len; i++)
            for (int j = 0; j < embed_size; j++)
                flat[i * embed_size + j] = v[i][j];

        // device memory
        float *d_data, *d_gamma, *d_beta;

        cudaMalloc(&d_data, N * sizeof(float));
        cudaMalloc(&d_gamma, embed_size * sizeof(float));
        cudaMalloc(&d_beta, embed_size * sizeof(float));

        cudaMemcpy(d_data,
                   flat.data(),
                   N * sizeof(float),
                   cudaMemcpyHostToDevice);

        cudaMemcpy(d_gamma,
                   gamma.data(),
                   embed_size * sizeof(float),
                   cudaMemcpyHostToDevice);

        cudaMemcpy(d_beta,
                   beta.data(),
                   embed_size * sizeof(float),
                   cudaMemcpyHostToDevice);

        int threads = 256;
        int blocks = (seq_len + threads - 1) / threads;

        layernorm_kernel<<<blocks, threads>>>(
            d_data,
            d_gamma,
            d_beta,
            seq_len,
            embed_size
        );

        cudaMemcpy(flat.data(),
                   d_data,
                   N * sizeof(float),
                   cudaMemcpyDeviceToHost);

        cudaFree(d_data);
        cudaFree(d_gamma);
        cudaFree(d_beta);

        // reshape back
        for (int i = 0; i < seq_len; i++)
            for (int j = 0; j < embed_size; j++)
                v[i][j] = flat[i * embed_size + j];
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

    inline std::vector<std::vector<float>>
    dropout_mask(
        int n1,
        int n2,
        float dropout_rate)
    {
        int N = n1 * n2;

        float *d_mask;
        curandState *d_states;

        cudaMalloc(&d_mask, N * sizeof(float));
        cudaMalloc(&d_states, N * sizeof(curandState));

        int threads = 256;
        int blocks = (N + threads - 1) / threads;

        dropout_mask_kernel<<<blocks, threads>>>(
            d_mask,
            d_states,
            n1,
            n2,
            dropout_rate
        );

        // copy back
        std::vector<float> flat(N);

        cudaMemcpy(flat.data(),
                   d_mask,
                   N * sizeof(float),
                   cudaMemcpyDeviceToHost);

        cudaFree(d_mask);
        cudaFree(d_states);

        // reshape
        std::vector<std::vector<float>> mask(
            n1,
            std::vector<float>(n2)
        );

        for (int i = 0; i < n1; i++)
            for (int j = 0; j < n2; j++)
                mask[i][j] = flat[i * n2 + j];

        return mask;
    }

    inline std::vector<std::vector<float>>
    dropout(
        std::vector<std::vector<float>>& v1,
        std::vector<std::vector<float>>& v2,
        float prob)
    {
        int rows = v1.size();
        int cols = v1[0].size();

        int N = rows * cols;

        // flatten
        std::vector<float> flatA(N);
        std::vector<float> flatMask(N);

        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
            {
                int idx = i * cols + j;

                flatA[idx] = v1[i][j];
                flatMask[idx] = v2[i][j];
            }

        // device memory
        float *d_A, *d_Mask, *d_Out;

        cudaMalloc(&d_A, N * sizeof(float));
        cudaMalloc(&d_Mask, N * sizeof(float));
        cudaMalloc(&d_Out, N * sizeof(float));

        cudaMemcpy(d_A,
                   flatA.data(),
                   N * sizeof(float),
                   cudaMemcpyHostToDevice);

        cudaMemcpy(d_Mask,
                   flatMask.data(),
                   N * sizeof(float),
                   cudaMemcpyHostToDevice);

        int threads = 256;
        int blocks = (N + threads - 1) / threads;

        dropout_kernel<<<blocks, threads>>>(
            d_A,
            d_Mask,
            d_Out,
            N,
            prob
        );

        // copy back
        std::vector<float> flatOut(N);

        cudaMemcpy(flatOut.data(),
                   d_Out,
                   N * sizeof(float),
                   cudaMemcpyDeviceToHost);

        cudaFree(d_A);
        cudaFree(d_Mask);
        cudaFree(d_Out);

        // reshape
        std::vector<std::vector<float>> ans(
            rows,
            std::vector<float>(cols)
        );

        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                ans[i][j] = flatOut[i * cols + j];

        return ans;
    }

    vector<vector<vector<float>>> head_spliting(vector<vector<float>>& v1, int head_size)
    {
        int seq_len = v1.size();
        int embed_size = v1[0].size();
        int head_dim = embed_size / head_size;

        vector<vector<vector<float>>> h;
        h.reserve(seq_len);

        for (int j = 0; j < seq_len; ++j)
        {
            vector<vector<float>> temp1;
            temp1.reserve(embed_size);      

            for (int k = 0; k < embed_size; k+=head_dim)
            {
                vector<float> temp;
                temp.reserve(head_dim);

                for (int l = k; l < k + head_dim; ++l) temp.push_back(v1[j][l]);
                temp1.push_back(temp);
            }
            h.push_back(temp1);
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

    vector<vector<float>> transpose(const vector<vector<float>>& v)
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
    
    vector<vector<float>> transpose2(vector<vector<float>>& v)
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
    inline std::vector<std::vector<float>>
    matadd(
        std::vector<std::vector<float>>& v1,
        std::vector<std::vector<float>>& v2)
    {
        int rows = v1.size();
        int cols = v1[0].size();

        int N = rows * cols;

        // flatten
        std::vector<float> flatA(N);
        std::vector<float> flatB(N);

        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
            {
                int idx = i * cols + j;

                flatA[idx] = v1[i][j];
                flatB[idx] = v2[i][j];
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

        matadd_kernel<<<blocks, threads>>>(
            d_A,
            d_B,
            d_C,
            N
        );

        // copy back
        std::vector<float> flatC(N);

        cudaMemcpy(flatC.data(),
                d_C,
                N * sizeof(float),
                cudaMemcpyDeviceToHost);

        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);

        // reshape
        std::vector<std::vector<float>> ans(
            rows,
            std::vector<float>(cols)
        );

        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                ans[i][j] = flatC[i * cols + j];

        return ans;
    }

}

#endif