#ifndef ADD_CUH
#define ADD_CUH

#include <iostream>
#include <vector>
#include "Tensor.cuh"

using namespace std;

namespace
{
    curandGenerator_t generator;

    __global__ void fill_kernel(
        float* data,
        float value,
        size_t n)
    {
        size_t i = blockIdx.x * blockDim.x + threadIdx.x;

        if (i < n)
            data[i] = value;
    }

    __global__ void embedding_kernel(
        const long long* token_ids,
        const float* embed_mat,
        const float* pos_mat,
        float* X,
        int seq_len,
        int embed_size,
        int total)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        if (idx >= total)
        return;

        int c = idx % embed_size;
        int tmp = idx / embed_size;

        int t = tmp % seq_len;
        int b = tmp / seq_len;

        long long token = token_ids[b * seq_len + t];

        X[idx] =
            embed_mat[token * embed_size + c]
            + pos_mat[t * embed_size + c];
    }    
}

namespace Tensor
{
    void init_rng()
    {
        curandCreateGenerator(
            &generator,
            CURAND_RNG_PSEUDO_DEFAULT);

        curandSetPseudoRandomGeneratorSeed(
            generator,
            1234ULL);
    }

    void destroy_rng()
    {
        curandDestroyGenerator(generator);
    }

    void fill(
        float* ptr,
        float value,
        size_t n)
    {
        int threads = 256;
        int blocks = (n + threads - 1) / threads;

        fill_kernel<<<blocks, threads>>>(
            ptr,
            value,
            n);
    }

    void random(
        float* ptr,
        size_t n,
        float mean,
        float std)
    {
        curandGenerateNormal(
            generator,
            ptr,
            n,
            mean,
            std);
    }

    void display(
        const float* d_ptr,
        size_t n)
    {
        vector<float> h(n);

        cudaMemcpy(
            h.data(),
            d_ptr,
            n * sizeof(float),
            cudaMemcpyDeviceToHost);
        
        cout << "Shape : " << n << endl;
        for (size_t i = 0; i < n; i++)
        {
            cout << h[i] << ' ';
        }
        cout << '\n';
    }

    void display(
        const long long* d_ptr,
        size_t n)
    {
        vector<long long> h(n);

        cudaMemcpy(
            h.data(),
            d_ptr,
            n * sizeof(long long),
            cudaMemcpyDeviceToHost);
        
        cout << "Shape : " << n << endl;

        for (size_t i = 0; i < n; i++)
        {
            cout << h[i] << ' ';
        }

        cout << '\n';
    }

    float* prepare_x(const long long* tokens, const float* embed_mat, const float* pos_mat,
                    int batch_size, int seq_len, int embed_size)
    {
        
        int total = batch_size * seq_len * embed_size;
        float* X;
        cudaMalloc(&X, total * sizeof(float));

        int threads = 256;
        int blocks = (total + threads - 1) / threads;

        embedding_kernel<<<blocks, threads>>>(tokens, embed_mat, pos_mat, X, seq_len, embed_size, total);

        return X;
    }

    long long* flatten(const vector<vector<long long>>& x, int i, int current_batch_size, int seq_len)
    {
        vector<long long> batch_x;
        batch_x.reserve(current_batch_size * seq_len);
        
        for(int j = i; j < current_batch_size + i; ++j) for(int k = 0; k < seq_len; ++k) batch_x.push_back(x[j][k]);
        
        size_t bytes = batch_x.size() * sizeof(long long);
        long long* token_X = nullptr;
        cudaMalloc(&token_X, bytes);
        cudaMemcpy(token_X, batch_x.data(), bytes, cudaMemcpyHostToDevice);
        
        return token_X;
    }
}
#endif