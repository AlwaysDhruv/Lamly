#ifndef ADD_CUH
#define ADD_CUH

#include <iostream>
#include <vector>
#include "Tensor.cuh"
#include <curand.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

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

    __global__ void dropout_mask_kernel(
        float* mask,
        int N,
        float dropout_rate)
    {
        int idx =
            blockIdx.x *
            blockDim.x +
            threadIdx.x;

        if(idx < N)
        {
            curandState state;

            curand_init(
                clock64(),
                idx,
                0,
                &state
            );

            float r =
                curand_uniform(
                    &state
                );

            mask[idx] =
                (r > dropout_rate)
                ? 1.0f
                : 0.0f;
        }
    }
    
    __global__ void dropout_kernel(float* x, const float* mask, float probs, size_t N)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        if(idx < N)
        {
            x[idx] *= mask[idx] / probs;
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
        int batch_size,
        int seq_len,
        int embed_size)
    {
        int token_idx =
            blockIdx.x * blockDim.x +
            threadIdx.x;

        int total_tokens =
            batch_size * seq_len;

        if(token_idx < total_tokens)
        {
            int offset =
                token_idx * embed_size;

            float mean = 0.0f;

            for(int k = 0; k < embed_size; k++)
                mean += value[offset + k];

            mean /= embed_size;

            mean_out[token_idx] = mean;

            float variance = 0.0f;

            for(int k = 0; k < embed_size; k++)
            {
                float diff =
                    value[offset + k] - mean;

                variance += diff * diff;
            }

            variance /= embed_size;

            var_out[token_idx] = variance;

            float std =
                sqrtf(variance + 1e-5f);

            std_out[token_idx] = std;

            for(int k = 0; k < embed_size; k++)
            {
                float norm =
                    (value[offset + k] - mean)
                    / std;

                xnorm_out[offset + k] =
                    norm;

                value[offset + k] =
                    gamma[k] * norm
                    + beta[k];
            }
        }
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

    float* dropout_mask(size_t n, float rate)
    {
        float* mask;
        size_t bytes = n * sizeof(float);
        cudaMalloc(&mask, bytes);
        dropout_mask_kernel<<< (n + 255) / 256, 256 >>>(mask, n, rate);

        return mask;
    }

    void dropout(float* x, const float* mask, float probs, size_t N)
    {
        int threads = 256;
        int blocks = (N + threads - 1) / threads;

        dropout_kernel<<<blocks, threads>>>(x, mask, probs, N);
    }

    void layer_norm(float* x, float* gamma, float* beta, float* m, float* v, float* s, float* X_norm,
                    int batch_size, int seq_len, int embed_size)
    {
        int N = batch_size * seq_len;
        layernorm_kernel<<< (N + 255)/256, 256>>>(x, gamma, beta, m, v, s, X_norm, batch_size, seq_len, embed_size);
    }
}
#endif