#ifndef TENSOR_CUH
#define TENSOR_CUH

#include <vector>
#include <cuda_runtime.h>
#include <curand.h>

using namespace std;

namespace Tensor
{
    void init_rng();

    void destroy_rng();

    void fill(
        float* ptr,
        float value,
        size_t n);

    void random(
        float* ptr,
        size_t n,
        float mean = 0.0f,
        float std = 0.02f);

    void display(
        const long long* d_ptr,
        size_t n);

    void display(
        const float* d_ptr,
        size_t n);

    float* prepare_x(const long long* tokens, const float* embed_mat, const float* pos_mat,
                    int batch_size, int seq_len, int embed_size);        
    
    long long* flatten(const vector<vector<long long>>& x, int i, int current_batch_size, int seq_len);

    float* dropout_mask(size_t n, float rate);

    void dropout(float* x, const float* mask, float probs, size_t N);

    void linear_projection(const float* x, const float* wq, const float* wk, const float* wv, float* q, float* k, float* v, int batch_size, int seq_len, int embed_size);

    void linear_projection(float* x, float* q, float* k, float* v,
                            int batch_size, int seq_len, int embed_size);
}

#endif