#ifndef ADD_CUH
#define ADD_CUH

#include "Tensor.cuh"

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
        std::vector<float> h(n);

        cudaMemcpy(
            h.data(),
            d_ptr,
            n * sizeof(float),
            cudaMemcpyDeviceToHost);

        for (size_t i = 0; i < n; i++)
        {
            std::cout << h[i] << ' ';
        }
        std::cout << '\n';
    }

    void display(
        const long long* d_ptr,
        size_t n)
    {
        std::vector<long long> h(n);

        cudaMemcpy(
            h.data(),
            d_ptr,
            n * sizeof(long long),
            cudaMemcpyDeviceToHost);

        for (size_t i = 0; i < n; i++)
        {
            std::cout << h[i] << ' ';
        }

        std::cout << '\n';
    }
}
#endif