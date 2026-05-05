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
                                   int d1, int d2, int d3)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;

        int N = d1 * d2 * d3;

        if (i < N)
        {
            curand_init(1234, i, 0, &states[i]);

            // generate 0 or 1
            float r = curand_uniform(&states[i]);
            data[i] = (r > 0.5f) ? 1 : 0;
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

    inline std::vector<vector<float>> random(int N1, int N2)
    {
        int N = N1 * N2;
        float *d_data;
        curandState *d_states;

        cudaMalloc(&d_data, N * sizeof(float));
        cudaMalloc(&d_states, N * sizeof(curandState));

        int threads = 256;
        int blocks = (N + threads - 1) / threads;

        random_kernel<<<blocks, threads>>>(d_data, d_states, N);

        vector<float> h_data(N);
        cudaMemcpy(h_data.data(), d_data, N * sizeof(float), cudaMemcpyDeviceToHost);

        cudaFree(d_data);
        cudaFree(d_states);

        vector<vector<float>> weigths;
        weigths.reserve(N1);

        for(int i = 0; i < N1; ++i)
        {
            vector<float> temp;
            temp.reserve(N2);

            int index = N1 * N2;

            for(int j = index; j < index + N2; ++j) temp.push_back(h_data[j]);

            weigths.push_back(temp);
        }
        return weigths;
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
    dropout_mask(int d1, int d2, int d3)
    {
        int N = d1 * d2 * d3;

        int *d_data;
        curandState *d_states;

        cudaMalloc(&d_data, N * sizeof(int));
        cudaMalloc(&d_states, N * sizeof(curandState));

        int threads = 256;
        int blocks = (N + threads - 1) / threads;

        random3d_kernel<<<blocks, threads>>>(d_data, d_states, d1, d2, d3);

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
    
    inline std::vector<std::vector<std::vector<float>>> dropout_gpu(
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
}

#endif