#ifndef TENSOR_CUH
#define TENSOR_CUH

#include <cuda_runtime.h>
#include <curand.h>

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

    void prepare_x(float* X, const long long* tokens, const float* embed_mat, const float* pos_mat,
                    int batch_size, int seq_len, int embed_size, int num_seq);        
}

#endif