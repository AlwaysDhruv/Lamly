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

    __global__ void elementwise_mul_broadcast_kernel(
        float *A,
        float *B,
        float *C,
        int rows,
        int cols)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        int N = rows * cols;

        if (idx < N)
        {
            int col = idx % cols;

            C[idx] = A[idx] * B[col];
        }
    }

    __global__ void random_kernel(
        float *data,
        curandState *states,
        int N)
    {
        int idx =
            blockIdx.x * blockDim.x + threadIdx.x;

        if (idx < N)
        {
            curand_init(
                1234,
                idx,
                0,
                &states[idx]);

            data[idx] =
                curand_uniform(&states[idx]);
        }
    }

    __global__ void sum2d_kernel(
        float *data,
        float *out,
        int rows,
        int cols)
    {
        int col =
            blockIdx.x * blockDim.x + threadIdx.x;

        if (col < cols)
        {
            float sum = 0.0f;

            for (int row = 0; row < rows; row++)
            {
                sum += data[row * cols + col];
            }

            out[col] = sum;
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
    __global__ void elementwise_mul_kernel(
        float *A,
        float *B,
        float *C,
        int N)
    {
        int idx =
            blockIdx.x * blockDim.x + threadIdx.x;

        if (idx < N)
        {
            C[idx] =
                A[idx] * B[idx];
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

    __global__ void add_kernel(
        float *data,
        float *out,
        int batch_size,
        int seq_len,
        int embed_size)
    {
        int j = blockIdx.x * blockDim.x + threadIdx.x;

        if (j < embed_size)
        {
            float sum = 0.0f;

            for (int i = 0; i < batch_size; i++)
            {
                for (int k = 0; k < seq_len; k++)
                {
                    int idx =
                        (i * seq_len + k) * embed_size + j;

                    sum += data[idx];
                }
            }

            out[j] = sum;
        }
    }

    __global__ void matmul_e_kernel(
        float *A,
        float *B,
        float *C,
        int N)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;

        if (i < N)
        {
            C[i] = A[i] * B[i];
        }
    }

    __global__ void normalized_gradient_kernel(
        float *A,
        float *B,
        float *C,
        int batch_size,
        int seq_len,
        int embed_size)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        int N = batch_size * seq_len * embed_size;

        if (idx < N)
        {
            int k = idx % embed_size;

            C[idx] = A[idx] * B[k];
        }
    }

    __global__ void variance_gradient_kernel(
        float *dx,
        float *X,
        float *mean,
        float *var,
        float *out,
        int batch_size,
        int seq_len,
        int embed_size)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        int total = batch_size * seq_len;

        if (idx < total)
        {
            int i = idx / seq_len;
            int j = idx % seq_len;

            int offset =
                (i * seq_len + j) * embed_size;

            float inv_std_cubed =
                rsqrtf(var[idx] + 1e-5f);

            inv_std_cubed =
                inv_std_cubed *
                inv_std_cubed *
                inv_std_cubed;

            float sum = 0.0f;

            for (int k = 0; k < embed_size; k++)
            {
                sum +=
                    dx[offset + k] *
                    (X[offset + k] - mean[idx]) *
                    (-0.5f) *
                    inv_std_cubed;
            }

            out[idx] = sum;
        }
    }

    __global__ void mean_gradient_kernel(
        float *dx,
        float *X_meaned,
        float *dvar,
        float *std,
        float *out,
        int batch_size,
        int seq_len,
        int embed_size)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        int total = batch_size * seq_len;

        if (idx < total)
        {
            int offset =
                idx * embed_size;

            float value =
                -1.0f / std[idx];

            float sum1 = 0.0f;
            float sum2 = 0.0f;

            for (int k = 0; k < embed_size; k++)
            {
                sum1 +=
                    dx[offset + k] * value;

                sum2 +=
                    (-2.0f * X_meaned[offset + k]);
            }

            float mean_term =
                (dvar[idx] * sum2) / embed_size;

            out[idx] = sum1 + mean_term;
        }
    }

    __global__ void input_gradient_kernel(
        float *dx_hat,
        float *X_meaned,
        float *dvar,
        float *dmean,
        float *std,
        float *out,
        int batch_size,
        int seq_len,
        int embed_size)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        int N =
            batch_size *
            seq_len *
            embed_size;

        if (idx < N)
        {
            int token_idx =
                idx / embed_size;

            int k =
                idx % embed_size;

            out[idx] =
                (dx_hat[idx] / std[token_idx])
                +
                (
                    dvar[token_idx]
                    *
                    2.0f
                    *
                    X_meaned[idx]
                    /
                    embed_size
                )
                +
                (
                    dmean[token_idx]
                    /
                    embed_size
                );
        }
    }

    __global__ void layernorm_kernel(
        float *value,
        float *gamma,
        float *beta,
        float *mean_out,
        float *var_out,
        float *std_out,
        float *xnorm_out,
        int seq_len,
        int embed_size)
    {
        int row =
            blockIdx.x * blockDim.x + threadIdx.x;

        if (row < seq_len)
        {
            int offset =
                row * embed_size;

            // mean
            float mean = 0.0f;

            for (int k = 0; k < embed_size; k++)
                mean += value[offset + k];

            mean /= embed_size;

            mean_out[row] = mean;

            // variance
            float variance = 0.0f;

            for (int k = 0; k < embed_size; k++)
            {
                float diff =
                    value[offset + k] - mean;

                variance += diff * diff;
            }

            variance /= embed_size;

            var_out[row] = variance;

            // std
            float std =
                sqrtf(variance + 1e-5f);

            std_out[row] = std;

            // normalize + affine
            for (int k = 0; k < embed_size; k++)
            {
                float norm =
                    (value[offset + k] - mean) / std;

                xnorm_out[offset + k] = norm;

                value[offset + k] =
                    gamma[k] * norm + beta[k];
            }
        }
    }

    __global__ void causal_mask_kernel(
        float *attention,
        float *scaled,
        float *masked,
        int seq_len,
        float scale)
    {
        int row =
            blockIdx.y * blockDim.y + threadIdx.y;

        int col =
            blockIdx.x * blockDim.x + threadIdx.x;

        if (row < seq_len && col < seq_len)
        {
            int idx =
                row * seq_len + col;

            if (col > row)
            {
                attention[idx] = -1e9f;
                scaled[idx] = -1e9f;
                masked[idx] = -1e9f;
            }
            else
            {
                float value =
                    attention[idx] * scale;

                attention[idx] = value;
                scaled[idx] = value;
                masked[idx] = value;
            }
        }
    }

    __global__ void embed_pos_backward_kernel(
        float *dinput,
        float *dembed,
        float *dpos,
        long long *TX,
        int seq_len,
        int embed_size)
    {
        int idx =
            blockIdx.x * blockDim.x + threadIdx.x;

        int N = seq_len * embed_size;

        if (idx < N)
        {
            int token_pos =
                idx / embed_size;

            int em =
                idx % embed_size;

            int token_id =
                TX[token_pos];

            float grad =
                dinput[idx];

            atomicAdd(
                &dembed[token_id * embed_size + em],
                grad);

            atomicAdd(
                &dpos[token_pos * embed_size + em],
                grad);
        }
    }

    __global__ void sgd_kernel(
        float *weight,
        float *gradient,
        float lr,
        int N)
    {
        int idx =
            blockIdx.x * blockDim.x + threadIdx.x;

        if (idx < N)
        {
            weight[idx] -= lr * gradient[idx];
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

    inline std::vector<std::vector<std::vector<float>>>
    random(int N1, int N2, int N3)
    {
        int N = N1 * N2 * N3;

        // device memory
        float *d_data;
        curandState *d_states;

        cudaMalloc(
            &d_data,
            N * sizeof(float));

        cudaMalloc(
            &d_states,
            N * sizeof(curandState));

        int threads = 256;
        int blocks =
            (N + threads - 1) / threads;

        random_kernel<<<blocks, threads>>>(
            d_data,
            d_states,
            N);

        cudaDeviceSynchronize();

        // copy back
        std::vector<float> h_data(N);

        cudaMemcpy(
            h_data.data(),
            d_data,
            N * sizeof(float),
            cudaMemcpyDeviceToHost);

        cudaFree(d_data);
        cudaFree(d_states);

        // reshape
        std::vector<std::vector<std::vector<float>>> tensor(
            N1,
            std::vector<std::vector<float>>(
                N2,
                std::vector<float>(N3)));

        for (int i = 0; i < N1; i++)
        {
            for (int j = 0; j < N2; j++)
            {
                for (int k = 0; k < N3; k++)
                {
                    int idx =
                        (i * N2 + j) * N3 + k;

                    tensor[i][j][k] =
                        h_data[idx];
                }
            }
        }

        return tensor;
    }

    inline std::vector<std::vector<float>>
    random(int rows, int cols)
    {
        int N = rows * cols;

        float *d_data;
        curandState *d_states;

        cudaMalloc(
            &d_data,
            N * sizeof(float));

        cudaMalloc(
            &d_states,
            N * sizeof(curandState));

        int threads = 256;
        int blocks =
            (N + threads - 1) / threads;

        random_kernel<<<blocks, threads>>>(
            d_data,
            d_states,
            N);

        cudaDeviceSynchronize();

        std::vector<float> h_data(N);

        cudaMemcpy(
            h_data.data(),
            d_data,
            N * sizeof(float),
            cudaMemcpyDeviceToHost);

        cudaFree(d_data);
        cudaFree(d_states);

        // reshape
        std::vector<std::vector<float>> matrix(
            rows,
            std::vector<float>(cols));

        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                matrix[i][j] =
                    h_data[i * cols + j];
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

    inline std::vector<float> sum(
        std::vector<std::vector<std::vector<float>>>& v)
    {
        int batch_size = v.size();
        int seq_len = v[0].size();
        int embed_size = v[0][0].size();

        int N = batch_size * seq_len * embed_size;

        // flatten
        std::vector<float> flat(N);

        for (int i = 0; i < batch_size; i++)
            for (int k = 0; k < seq_len; k++)
                for (int j = 0; j < embed_size; j++)
                {
                    int idx =
                        (i * seq_len + k) * embed_size + j;

                    flat[idx] = v[i][k][j];
                }

        // output
        std::vector<float> ans(embed_size);

        // device memory
        float *d_data;
        float *d_out;

        cudaMalloc(&d_data, N * sizeof(float));
        cudaMalloc(&d_out, embed_size * sizeof(float));

        cudaMemcpy(d_data,
                   flat.data(),
                   N * sizeof(float),
                   cudaMemcpyHostToDevice);

        int threads = 256;
        int blocks = (embed_size + threads - 1) / threads;

        add_kernel<<<blocks, threads>>>(
            d_data,
            d_out,
            batch_size,
            seq_len,
            embed_size
        );

        cudaMemcpy(ans.data(),
                   d_out,
                   embed_size * sizeof(float),
                   cudaMemcpyDeviceToHost);

        cudaFree(d_data);
        cudaFree(d_out);

        return ans;
    }

    inline std::vector<std::vector<std::vector<float>>>
    matmul_e(
        std::vector<std::vector<std::vector<float>>>& v1,
        std::vector<std::vector<std::vector<float>>>& v2)
    {
        int batch_size = v1.size();
        int seq_len = v1[0].size();
        int embed_size = v1[0][0].size();

        int N = batch_size * seq_len * embed_size;

        // flatten
        std::vector<float> flatA(N);
        std::vector<float> flatB(N);

        for (int i = 0; i < batch_size; i++)
            for (int j = 0; j < seq_len; j++)
                for (int k = 0; k < embed_size; k++)
                {
                    int idx =
                        (i * seq_len + j) * embed_size + k;

                    flatA[idx] = v1[i][j][k];
                    flatB[idx] = v2[i][j][k];
                }

        // device memory
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

        matmul_e_kernel<<<blocks, threads>>>(
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
        std::vector<std::vector<std::vector<float>>> temp(
            batch_size,
            std::vector<std::vector<float>>(
                seq_len,
                std::vector<float>(embed_size)
            )
        );

        for (int i = 0; i < batch_size; i++)
            for (int j = 0; j < seq_len; j++)
                for (int k = 0; k < embed_size; k++)
                {
                    int idx =
                        (i * seq_len + j) * embed_size + k;

                    temp[i][j][k] = flatC[idx];
                }

        return temp;
    }

    inline std::vector<std::vector<std::vector<float>>>
    normalized_gradient(
        std::vector<std::vector<std::vector<float>>>& v1,
        std::vector<float>& v2)
    {
        int batch_size = v1.size();
        int seq_len = v1[0].size();
        int embed_size = v1[0][0].size();

        int N = batch_size * seq_len * embed_size;

        // flatten
        std::vector<float> flatA(N);

        for (int i = 0; i < batch_size; i++)
            for (int j = 0; j < seq_len; j++)
                for (int k = 0; k < embed_size; k++)
                {
                    int idx =
                        (i * seq_len + j) * embed_size + k;

                    flatA[idx] = v1[i][j][k];
                }

        // output
        std::vector<float> flatC(N);

        // device memory
        float *d_A, *d_B, *d_C;

        cudaMalloc(&d_A, N * sizeof(float));
        cudaMalloc(&d_B, embed_size * sizeof(float));
        cudaMalloc(&d_C, N * sizeof(float));

        cudaMemcpy(d_A,
                   flatA.data(),
                   N * sizeof(float),
                   cudaMemcpyHostToDevice);

        cudaMemcpy(d_B,
                   v2.data(),
                   embed_size * sizeof(float),
                   cudaMemcpyHostToDevice);

        int threads = 256;
        int blocks = (N + threads - 1) / threads;

        normalized_gradient_kernel<<<blocks, threads>>>(
            d_A,
            d_B,
            d_C,
            batch_size,
            seq_len,
            embed_size
        );

        cudaMemcpy(flatC.data(),
                   d_C,
                   N * sizeof(float),
                   cudaMemcpyDeviceToHost);

        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);

        // reshape
        std::vector<std::vector<std::vector<float>>> temp(
            batch_size,
            std::vector<std::vector<float>>(
                seq_len,
                std::vector<float>(embed_size)
            )
        );

        for (int i = 0; i < batch_size; i++)
            for (int j = 0; j < seq_len; j++)
                for (int k = 0; k < embed_size; k++)
                {
                    int idx =
                        (i * seq_len + j) * embed_size + k;

                    temp[i][j][k] = flatC[idx];
                }

        return temp;
    }

    inline std::vector<std::vector<float>>
    variance_gradient(
        std::vector<std::vector<std::vector<float>>>& dx,
        std::vector<std::vector<std::vector<float>>>& X,
        std::vector<std::vector<float>>& mean,
        std::vector<std::vector<float>>& var)
    {
        int batch_size = dx.size();
        int seq_len = dx[0].size();
        int embed_size = dx[0][0].size();

        int N = batch_size * seq_len * embed_size;
        int M = batch_size * seq_len;

        // flatten
        std::vector<float> flat_dx(N);
        std::vector<float> flat_X(N);

        std::vector<float> flat_mean(M);
        std::vector<float> flat_var(M);

        for (int i = 0; i < batch_size; i++)
        {
            for (int j = 0; j < seq_len; j++)
            {
                int idx2 = i * seq_len + j;

                flat_mean[idx2] = mean[i][j];
                flat_var[idx2] = var[i][j];

                for (int k = 0; k < embed_size; k++)
                {
                    int idx3 =
                        (i * seq_len + j) * embed_size + k;

                    flat_dx[idx3] = dx[i][j][k];
                    flat_X[idx3] = X[i][j][k];
                }
            }
        }

        // output
        std::vector<float> flat_out(M);

        // device memory
        float *d_dx, *d_X;
        float *d_mean, *d_var;
        float *d_out;

        cudaMalloc(&d_dx, N * sizeof(float));
        cudaMalloc(&d_X, N * sizeof(float));

        cudaMalloc(&d_mean, M * sizeof(float));
        cudaMalloc(&d_var, M * sizeof(float));

        cudaMalloc(&d_out, M * sizeof(float));

        cudaMemcpy(d_dx,
                   flat_dx.data(),
                   N * sizeof(float),
                   cudaMemcpyHostToDevice);

        cudaMemcpy(d_X,
                   flat_X.data(),
                   N * sizeof(float),
                   cudaMemcpyHostToDevice);

        cudaMemcpy(d_mean,
                   flat_mean.data(),
                   M * sizeof(float),
                   cudaMemcpyHostToDevice);

        cudaMemcpy(d_var,
                   flat_var.data(),
                   M * sizeof(float),
                   cudaMemcpyHostToDevice);

        int threads = 256;
        int blocks = (M + threads - 1) / threads;

        variance_gradient_kernel<<<blocks, threads>>>(
            d_dx,
            d_X,
            d_mean,
            d_var,
            d_out,
            batch_size,
            seq_len,
            embed_size
        );

        cudaMemcpy(flat_out.data(),
                   d_out,
                   M * sizeof(float),
                   cudaMemcpyDeviceToHost);

        cudaFree(d_dx);
        cudaFree(d_X);

        cudaFree(d_mean);
        cudaFree(d_var);

        cudaFree(d_out);

        // reshape
        std::vector<std::vector<float>> ans(
            batch_size,
            std::vector<float>(seq_len)
        );

        for (int i = 0; i < batch_size; i++)
            for (int j = 0; j < seq_len; j++)
            {
                int idx = i * seq_len + j;
                ans[i][j] = flat_out[idx];
            }

        return ans;
    }

    inline std::vector<std::vector<float>>
    mean_gradient(
        std::vector<std::vector<std::vector<float>>>& dx,
        std::vector<std::vector<std::vector<float>>>& X_meaned,
        std::vector<std::vector<float>>& dvar,
        std::vector<std::vector<float>>& std)
    {
        int batch_size = dx.size();
        int seq_len = dx[0].size();
        int embed_size = dx[0][0].size();

        int N = batch_size * seq_len * embed_size;
        int M = batch_size * seq_len;

        // flatten
        std::vector<float> flat_dx(N);
        std::vector<float> flat_Xm(N);

        std::vector<float> flat_dvar(M);
        std::vector<float> flat_std(M);

        for (int i = 0; i < batch_size; i++)
        {
            for (int j = 0; j < seq_len; j++)
            {
                int idx2 = i * seq_len + j;

                flat_dvar[idx2] = dvar[i][j];
                flat_std[idx2] = std[i][j];

                for (int k = 0; k < embed_size; k++)
                {
                    int idx3 =
                        (i * seq_len + j) * embed_size + k;

                    flat_dx[idx3] = dx[i][j][k];
                    flat_Xm[idx3] = X_meaned[i][j][k];
                }
            }
        }

        // output
        std::vector<float> flat_out(M);

        // device memory
        float *d_dx;
        float *d_Xm;

        float *d_dvar;
        float *d_std;

        float *d_out;

        cudaMalloc(&d_dx, N * sizeof(float));
        cudaMalloc(&d_Xm, N * sizeof(float));

        cudaMalloc(&d_dvar, M * sizeof(float));
        cudaMalloc(&d_std, M * sizeof(float));

        cudaMalloc(&d_out, M * sizeof(float));

        cudaMemcpy(d_dx,
                   flat_dx.data(),
                   N * sizeof(float),
                   cudaMemcpyHostToDevice);

        cudaMemcpy(d_Xm,
                   flat_Xm.data(),
                   N * sizeof(float),
                   cudaMemcpyHostToDevice);

        cudaMemcpy(d_dvar,
                   flat_dvar.data(),
                   M * sizeof(float),
                   cudaMemcpyHostToDevice);

        cudaMemcpy(d_std,
                   flat_std.data(),
                   M * sizeof(float),
                   cudaMemcpyHostToDevice);

        int threads = 256;
        int blocks = (M + threads - 1) / threads;

        mean_gradient_kernel<<<blocks, threads>>>(
            d_dx,
            d_Xm,
            d_dvar,
            d_std,
            d_out,
            batch_size,
            seq_len,
            embed_size
        );

        cudaMemcpy(flat_out.data(),
                   d_out,
                   M * sizeof(float),
                   cudaMemcpyDeviceToHost);

        cudaFree(d_dx);
        cudaFree(d_Xm);

        cudaFree(d_dvar);
        cudaFree(d_std);

        cudaFree(d_out);

        // reshape
        std::vector<std::vector<float>> ans(
            batch_size,
            std::vector<float>(seq_len)
        );

        for (int i = 0; i < batch_size; i++)
            for (int j = 0; j < seq_len; j++)
            {
                int idx = i * seq_len + j;
                ans[i][j] = flat_out[idx];
            }

        return ans;
    }

    inline std::vector<std::vector<std::vector<float>>>
    input_gradient(
        std::vector<std::vector<std::vector<float>>>& dx_hat,
        std::vector<std::vector<std::vector<float>>>& X_meaned,
        std::vector<std::vector<float>>& dvar,
        std::vector<std::vector<float>>& dmean,
        std::vector<std::vector<float>>& std)
    {
        int batch_size = dx_hat.size();
        int seq_len = dx_hat[0].size();
        int embed_size = dx_hat[0][0].size();

        int N =
            batch_size *
            seq_len *
            embed_size;

        int M =
            batch_size *
            seq_len;

        // flatten
        std::vector<float> flat_dxhat(N);
        std::vector<float> flat_Xm(N);

        std::vector<float> flat_dvar(M);
        std::vector<float> flat_dmean(M);
        std::vector<float> flat_std(M);

        for (int i = 0; i < batch_size; i++)
        {
            for (int j = 0; j < seq_len; j++)
            {
                int idx2 =
                    i * seq_len + j;

                flat_dvar[idx2] =
                    dvar[i][j];

                flat_dmean[idx2] =
                    dmean[i][j];

                flat_std[idx2] =
                    std[i][j];

                for (int k = 0; k < embed_size; k++)
                {
                    int idx3 =
                        (i * seq_len + j)
                        * embed_size + k;

                    flat_dxhat[idx3] =
                        dx_hat[i][j][k];

                    flat_Xm[idx3] =
                        X_meaned[i][j][k];
                }
            }
        }

        // output
        std::vector<float> flat_out(N);

        // device memory
        float *d_dxhat;
        float *d_Xm;

        float *d_dvar;
        float *d_dmean;
        float *d_std;

        float *d_out;

        cudaMalloc(&d_dxhat, N * sizeof(float));
        cudaMalloc(&d_Xm, N * sizeof(float));

        cudaMalloc(&d_dvar, M * sizeof(float));
        cudaMalloc(&d_dmean, M * sizeof(float));
        cudaMalloc(&d_std, M * sizeof(float));

        cudaMalloc(&d_out, N * sizeof(float));

        cudaMemcpy(d_dxhat,
                   flat_dxhat.data(),
                   N * sizeof(float),
                   cudaMemcpyHostToDevice);

        cudaMemcpy(d_Xm,
                   flat_Xm.data(),
                   N * sizeof(float),
                   cudaMemcpyHostToDevice);

        cudaMemcpy(d_dvar,
                   flat_dvar.data(),
                   M * sizeof(float),
                   cudaMemcpyHostToDevice);

        cudaMemcpy(d_dmean,
                   flat_dmean.data(),
                   M * sizeof(float),
                   cudaMemcpyHostToDevice);

        cudaMemcpy(d_std,
                   flat_std.data(),
                   M * sizeof(float),
                   cudaMemcpyHostToDevice);

        int threads = 256;
        int blocks =
            (N + threads - 1) / threads;

        input_gradient_kernel<<<blocks, threads>>>(
            d_dxhat,
            d_Xm,
            d_dvar,
            d_dmean,
            d_std,
            d_out,
            batch_size,
            seq_len,
            embed_size
        );

        cudaMemcpy(flat_out.data(),
                   d_out,
                   N * sizeof(float),
                   cudaMemcpyDeviceToHost);

        cudaFree(d_dxhat);
        cudaFree(d_Xm);

        cudaFree(d_dvar);
        cudaFree(d_dmean);
        cudaFree(d_std);

        cudaFree(d_out);

        // reshape
        std::vector<std::vector<std::vector<float>>> ans(
            batch_size,
            std::vector<std::vector<float>>(
                seq_len,
                std::vector<float>(embed_size)
            )
        );

        for (int i = 0; i < batch_size; i++)
            for (int j = 0; j < seq_len; j++)
                for (int k = 0; k < embed_size; k++)
                {
                    int idx =
                        (i * seq_len + j)
                        * embed_size + k;

                    ans[i][j][k] =
                        flat_out[idx];
                }

        return ans;
    }

        inline void layer_norm(
        std::vector<std::vector<float>>& value,
        std::vector<float>& gamma,
        std::vector<float>& beta,

        std::vector<float>& m,
        std::vector<float>& v,
        std::vector<float>& s,

        std::vector<std::vector<float>>& X_norm)
    {
        int seq_len = value.size();
        int embed_size = value[0].size();

        int N = seq_len * embed_size;

        // flatten input
        std::vector<float> flat(N);

        for (int i = 0; i < seq_len; i++)
            for (int j = 0; j < embed_size; j++)
                flat[i * embed_size + j] =
                    value[i][j];

        // resize outputs
        m.resize(seq_len);
        v.resize(seq_len);
        s.resize(seq_len);

        std::vector<float> flat_xnorm(N);

        // device memory
        float *d_value;

        float *d_gamma;
        float *d_beta;

        float *d_mean;
        float *d_var;
        float *d_std;

        float *d_xnorm;

        cudaMalloc(&d_value, N * sizeof(float));

        cudaMalloc(&d_gamma,
                   embed_size * sizeof(float));

        cudaMalloc(&d_beta,
                   embed_size * sizeof(float));

        cudaMalloc(&d_mean,
                   seq_len * sizeof(float));

        cudaMalloc(&d_var,
                   seq_len * sizeof(float));

        cudaMalloc(&d_std,
                   seq_len * sizeof(float));

        cudaMalloc(&d_xnorm,
                   N * sizeof(float));

        cudaMemcpy(d_value,
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

        int blocks =
            (seq_len + threads - 1) / threads;

        layernorm_kernel<<<blocks, threads>>>(
            d_value,
            d_gamma,
            d_beta,
            d_mean,
            d_var,
            d_std,
            d_xnorm,
            seq_len,
            embed_size
        );

        // copy back
        cudaMemcpy(flat.data(),
                   d_value,
                   N * sizeof(float),
                   cudaMemcpyDeviceToHost);

        cudaMemcpy(m.data(),
                   d_mean,
                   seq_len * sizeof(float),
                   cudaMemcpyDeviceToHost);

        cudaMemcpy(v.data(),
                   d_var,
                   seq_len * sizeof(float),
                   cudaMemcpyDeviceToHost);

        cudaMemcpy(s.data(),
                   d_std,
                   seq_len * sizeof(float),
                   cudaMemcpyDeviceToHost);

        cudaMemcpy(flat_xnorm.data(),
                   d_xnorm,
                   N * sizeof(float),
                   cudaMemcpyDeviceToHost);

        // free
        cudaFree(d_value);

        cudaFree(d_gamma);
        cudaFree(d_beta);

        cudaFree(d_mean);
        cudaFree(d_var);
        cudaFree(d_std);

        cudaFree(d_xnorm);

        // reshape outputs
        X_norm.assign(
            seq_len,
            std::vector<float>(embed_size)
        );

        for (int i = 0; i < seq_len; i++)
        {
            for (int j = 0; j < embed_size; j++)
            {
                int idx =
                    i * embed_size + j;

                X_norm[i][j] =
                    flat_xnorm[idx];

                value[i][j] =
                    flat[idx];
            }
        }
    }

    inline void causal_mask(
        std::vector<std::vector<float>>& attention_score,
        float scale,
        std::vector<std::vector<float>>& scaled,
        std::vector<std::vector<float>>& masked)
    {
        int seq_len =
            attention_score.size();

        int N =
            seq_len * seq_len;

        // flatten
        std::vector<float> flat_attention(N);

        for (int i = 0; i < seq_len; i++)
            for (int j = 0; j < seq_len; j++)
                flat_attention[i * seq_len + j] =
                    attention_score[i][j];

        // outputs
        std::vector<float> flat_scaled(N);
        std::vector<float> flat_masked(N);

        // device memory
        float *d_attention;
        float *d_scaled;
        float *d_masked;

        cudaMalloc(&d_attention,
                   N * sizeof(float));

        cudaMalloc(&d_scaled,
                   N * sizeof(float));

        cudaMalloc(&d_masked,
                   N * sizeof(float));

        cudaMemcpy(d_attention,
                   flat_attention.data(),
                   N * sizeof(float),
                   cudaMemcpyHostToDevice);

        dim3 threads(16, 16);

        dim3 blocks(
            (seq_len + 15) / 16,
            (seq_len + 15) / 16
        );

        causal_mask_kernel<<<blocks, threads>>>(
            d_attention,
            d_scaled,
            d_masked,
            seq_len,
            scale
        );

        // copy back
        cudaMemcpy(flat_attention.data(),
                   d_attention,
                   N * sizeof(float),
                   cudaMemcpyDeviceToHost);

        cudaMemcpy(flat_scaled.data(),
                   d_scaled,
                   N * sizeof(float),
                   cudaMemcpyDeviceToHost);

        cudaMemcpy(flat_masked.data(),
                   d_masked,
                   N * sizeof(float),
                   cudaMemcpyDeviceToHost);

        // free
        cudaFree(d_attention);
        cudaFree(d_scaled);
        cudaFree(d_masked);

        // reshape
        scaled.assign(
            seq_len,
            std::vector<float>(seq_len)
        );

        masked.assign(
            seq_len,
            std::vector<float>(seq_len)
        );

        for (int i = 0; i < seq_len; i++)
        {
            for (int j = 0; j < seq_len; j++)
            {
                int idx =
                    i * seq_len + j;

                attention_score[i][j] =
                    flat_attention[idx];

                scaled[i][j] =
                    flat_scaled[idx];

                masked[i][j] =
                    flat_masked[idx];
            }
        }
    }

    inline std::vector<std::vector<float>>
    elementwise_mul(
        std::vector<std::vector<float>>& v1,
        std::vector<std::vector<float>>& v2)
    {
        int rows =
            v1.size();

        int cols =
            v1[0].size();

        int N =
            rows * cols;

        // flatten
        std::vector<float> flatA(N);
        std::vector<float> flatB(N);

        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                int idx =
                    i * cols + j;

                flatA[idx] =
                    v1[i][j];

                flatB[idx] =
                    v2[i][j];
            }
        }

        // output
        std::vector<float> flatC(N);

        // device memory
        float *d_A;
        float *d_B;
        float *d_C;

        cudaMalloc(&d_A,
                N * sizeof(float));

        cudaMalloc(&d_B,
                N * sizeof(float));

        cudaMalloc(&d_C,
                N * sizeof(float));

        cudaMemcpy(d_A,
                flatA.data(),
                N * sizeof(float),
                cudaMemcpyHostToDevice);

        cudaMemcpy(d_B,
                flatB.data(),
                N * sizeof(float),
                cudaMemcpyHostToDevice);

        int threads = 256;

        int blocks =
            (N + threads - 1) / threads;

        elementwise_mul_kernel<<<blocks, threads>>>(
            d_A,
            d_B,
            d_C,
            N
        );

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
        {
            for (int j = 0; j < cols; j++)
            {
                int idx =
                    i * cols + j;

                ans[i][j] =
                    flatC[idx];
            }
        }

        return ans;
    }

    inline std::vector<std::vector<float>>
    elementwise_mul(
        std::vector<std::vector<float>>& v1,
        std::vector<float>& v2)
    {
        int rows = v1.size();
        int cols = v1[0].size();

        int N = rows * cols;

        // flatten matrix
        std::vector<float> flatA(N);

        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                flatA[i * cols + j] = v1[i][j];

        std::vector<float> flatC(N);

        float *d_A;
        float *d_B;
        float *d_C;

        cudaMalloc(&d_A, N * sizeof(float));
        cudaMalloc(&d_B, cols * sizeof(float));
        cudaMalloc(&d_C, N * sizeof(float));

        cudaMemcpy(
            d_A,
            flatA.data(),
            N * sizeof(float),
            cudaMemcpyHostToDevice);

        cudaMemcpy(
            d_B,
            v2.data(),
            cols * sizeof(float),
            cudaMemcpyHostToDevice);

        int threads = 256;
        int blocks = (N + threads - 1) / threads;

        elementwise_mul_broadcast_kernel<<<blocks, threads>>>(
            d_A,
            d_B,
            d_C,
            rows,
            cols);

        cudaMemcpy(
            flatC.data(),
            d_C,
            N * sizeof(float),
            cudaMemcpyDeviceToHost);

        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);

        // reshape
        std::vector<std::vector<float>> ans(
            rows,
            std::vector<float>(cols));

        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                ans[i][j] = flatC[i * cols + j];

        return ans;
    }

    inline std::vector<float> sum(
        std::vector<std::vector<float>>& v)
    {
        int rows = v.size();
        int cols = v[0].size();

        int N = rows * cols;

        // flatten
        std::vector<float> flat(N);

        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                flat[i * cols + j] =
                    v[i][j];
            }
        }

        std::vector<float> ans(cols);

        float *d_data;
        float *d_out;

        cudaMalloc(&d_data,
                   N * sizeof(float));

        cudaMalloc(&d_out,
                   cols * sizeof(float));

        cudaMemcpy(d_data,
                   flat.data(),
                   N * sizeof(float),
                   cudaMemcpyHostToDevice);

        int threads = 256;
        int blocks =
            (cols + threads - 1) / threads;

        sum2d_kernel<<<blocks, threads>>>(
            d_data,
            d_out,
            rows,
            cols);

        cudaMemcpy(ans.data(),
                   d_out,
                   cols * sizeof(float),
                   cudaMemcpyDeviceToHost);

        cudaFree(d_data);
        cudaFree(d_out);

        return ans;
    }

    vector<vector<vector<float>>> split_heads(
        const vector<vector<float>>& x,
        int num_heads
    )
    {
        int seq_len = x.size();
        int embed_size = x[0].size();

        int head_dim = embed_size / num_heads;

        vector<vector<vector<float>>> out(
            num_heads,
            vector<vector<float>>(
                seq_len,
                vector<float>(head_dim)
            )
        );

        for (int t = 0; t < seq_len; t++)
        {
            for (int h = 0; h < num_heads; h++)
            {
                for (int d = 0; d < head_dim; d++)
                {
                    out[h][t][d] =
                        x[t][h * head_dim + d];
                }
            }
        }

        return out;
    }

    vector<vector<float>> merge_heads(
        const vector<vector<vector<float>>>& x
    )
    {
        int heads = x.size();
        int seq_len = x[0].size();
        int head_dim = x[0][0].size();

        int embed_size =
            heads * head_dim;

        vector<vector<float>> out(
            seq_len,
            vector<float>(
                embed_size,
                0.0f
            )
        );

        for (int token = 0;
             token < seq_len;
             ++token)
        {
            for (int head = 0;
                 head < heads;
                 ++head)
            {
                for (int dim = 0;
                     dim < head_dim;
                     ++dim)
                {
                    out[token]
                       [head * head_dim + dim]
                    =
                    x[head]
                     [token]
                     [dim];
                }
            }
        }

        return out;
    }

    inline void embed_pos_backward(
        std::vector<std::vector<float>>& dinput,
        std::vector<std::vector<float>>& dembed_mat,
        std::vector<std::vector<float>>& dpos_mat,
        std::vector<long long>& TX)
    {
        int seq_len = dinput.size();
        int embed_size = dinput[0].size();

        int vocab_size = dembed_mat.size();

        int N = seq_len * embed_size;

        // flatten
        std::vector<float> flat_input(N);

        for (int i = 0; i < seq_len; i++)
            for (int j = 0; j < embed_size; j++)
                flat_input[i * embed_size + j]
                    = dinput[i][j];

        std::vector<float> flat_embed(
            vocab_size * embed_size);

        std::vector<float> flat_pos(
            seq_len * embed_size);

        for (int i = 0; i < vocab_size; i++)
            for (int j = 0; j < embed_size; j++)
                flat_embed[i * embed_size + j]
                    = dembed_mat[i][j];

        for (int i = 0; i < seq_len; i++)
            for (int j = 0; j < embed_size; j++)
                flat_pos[i * embed_size + j]
                    = dpos_mat[i][j];

        float *d_input;
        float *d_embed;
        float *d_pos;
        long long *d_TX;

        cudaMalloc(&d_input,
                   N * sizeof(float));

        cudaMalloc(&d_embed,
                   flat_embed.size() *
                   sizeof(float));

        cudaMalloc(&d_pos,
                   flat_pos.size() *
                   sizeof(float));

        cudaMalloc(&d_TX,
                   seq_len *
                   sizeof(long long));

        cudaMemcpy(
            d_input,
            flat_input.data(),
            N * sizeof(float),
            cudaMemcpyHostToDevice);

        cudaMemcpy(
            d_embed,
            flat_embed.data(),
            flat_embed.size() * sizeof(float),
            cudaMemcpyHostToDevice);

        cudaMemcpy(
            d_pos,
            flat_pos.data(),
            flat_pos.size() * sizeof(float),
            cudaMemcpyHostToDevice);

        cudaMemcpy(
            d_TX,
            TX.data(),
            seq_len * sizeof(long long),
            cudaMemcpyHostToDevice);

        int threads = 256;
        int blocks =
            (N + threads - 1) / threads;

        embed_pos_backward_kernel<<<blocks, threads>>>(
            d_input,
            d_embed,
            d_pos,
            d_TX,
            seq_len,
            embed_size);

        cudaMemcpy(
            flat_embed.data(),
            d_embed,
            flat_embed.size() * sizeof(float),
            cudaMemcpyDeviceToHost);

        cudaMemcpy(
            flat_pos.data(),
            d_pos,
            flat_pos.size() * sizeof(float),
            cudaMemcpyDeviceToHost);

        cudaFree(d_input);
        cudaFree(d_embed);
        cudaFree(d_pos);
        cudaFree(d_TX);

        // reshape back
        for (int i = 0; i < vocab_size; i++)
            for (int j = 0; j < embed_size; j++)
                dembed_mat[i][j] =
                    flat_embed[i * embed_size + j];

        for (int i = 0; i < seq_len; i++)
            for (int j = 0; j < embed_size; j++)
                dpos_mat[i][j] =
                    flat_pos[i * embed_size + j];
    }

    inline void SGD(
        std::vector<float>& weight,
        std::vector<float>& gradient,
        float lr)
    {
        int N = weight.size();

        float *d_weight;
        float *d_gradient;

        cudaMalloc(&d_weight, N * sizeof(float));
        cudaMalloc(&d_gradient, N * sizeof(float));

        cudaMemcpy(
            d_weight,
            weight.data(),
            N * sizeof(float),
            cudaMemcpyHostToDevice);

        cudaMemcpy(
            d_gradient,
            gradient.data(),
            N * sizeof(float),
            cudaMemcpyHostToDevice);

        int threads = 256;
        int blocks = (N + threads - 1) / threads;

        sgd_kernel<<<blocks, threads>>>(
            d_weight,
            d_gradient,
            lr,
            N);

        cudaMemcpy(
            weight.data(),
            d_weight,
            N * sizeof(float),
            cudaMemcpyDeviceToHost);

        cudaFree(d_weight);
        cudaFree(d_gradient);
    }

    inline void SGD(
        std::vector<std::vector<float>>& weight,
        std::vector<std::vector<float>>& gradient,
        float lr)
    {
        int rows = weight.size();
        int cols = weight[0].size();

        int N = rows * cols;

        std::vector<float> flat_w(N);
        std::vector<float> flat_g(N);

        for(int i=0;i<rows;i++)
        {
            for(int j=0;j<cols;j++)
            {
                int idx = i * cols + j;

                flat_w[idx] = weight[i][j];
                flat_g[idx] = gradient[i][j];
            }
        }

        float *d_w;
        float *d_g;

        cudaMalloc(&d_w, N*sizeof(float));
        cudaMalloc(&d_g, N*sizeof(float));

        cudaMemcpy(
            d_w,
            flat_w.data(),
            N*sizeof(float),
            cudaMemcpyHostToDevice);

        cudaMemcpy(
            d_g,
            flat_g.data(),
            N*sizeof(float),
            cudaMemcpyHostToDevice);

        int threads = 256;
        int blocks = (N + threads - 1) / threads;

        sgd_kernel<<<blocks, threads>>>(
            d_w,
            d_g,
            lr,
            N);

        cudaMemcpy(
            flat_w.data(),
            d_w,
            N*sizeof(float),
            cudaMemcpyDeviceToHost);

        cudaFree(d_w);
        cudaFree(d_g);

        for(int i=0;i<rows;i++)
            for(int j=0;j<cols;j++)
                weight[i][j] =
                    flat_w[i * cols + j];
    }

    inline void SGD(
        std::vector<std::vector<std::vector<float>>>& weight,
        std::vector<std::vector<std::vector<float>>>& gradient,
        float lr)
    {
        int d1 = weight.size();
        int d2 = weight[0].size();
        int d3 = weight[0][0].size();

        int N = d1 * d2 * d3;

        std::vector<float> flat_w(N);
        std::vector<float> flat_g(N);

        for(int i=0;i<d1;i++)
        {
            for(int j=0;j<d2;j++)
            {
                for(int k=0;k<d3;k++)
                {
                    int idx =
                        (i * d2 + j) * d3 + k;

                    flat_w[idx] =
                        weight[i][j][k];

                    flat_g[idx] =
                        gradient[i][j][k];
                }
            }
        }

        float *d_w;
        float *d_g;

        cudaMalloc(&d_w, N*sizeof(float));
        cudaMalloc(&d_g, N*sizeof(float));

        cudaMemcpy(
            d_w,
            flat_w.data(),
            N*sizeof(float),
            cudaMemcpyHostToDevice);

        cudaMemcpy(
            d_g,
            flat_g.data(),
            N*sizeof(float),
            cudaMemcpyHostToDevice);

        int threads = 256;
        int blocks = (N + threads - 1) / threads;

        sgd_kernel<<<blocks, threads>>>(
            d_w,
            d_g,
            lr,
            N);

        cudaMemcpy(
            flat_w.data(),
            d_w,
            N*sizeof(float),
            cudaMemcpyDeviceToHost);

        cudaFree(d_w);
        cudaFree(d_g);

        for(int i=0;i<d1;i++)
        {
            for(int j=0;j<d2;j++)
            {
                for(int k=0;k<d3;k++)
                {
                    int idx =
                        (i * d2 + j) * d3 + k;

                    weight[i][j][k] =
                        flat_w[idx];
                }
            }
        }
    }
}
#endif