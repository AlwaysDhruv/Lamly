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

    inline std::vector<float> add(const std::vector<float>& A,
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
}

#endif