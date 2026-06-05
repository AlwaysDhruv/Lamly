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
}

#endif