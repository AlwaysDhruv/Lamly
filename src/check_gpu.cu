#include <iostream>
#include <cuda_runtime.h>

int main()
{
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    if (err != cudaSuccess)
    {
        std::cout << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }

    if (deviceCount == 0)
    {
        std::cout << "No GPU available ❌" << std::endl;
        return 0;
    }

    std::cout << "GPU available ✅" << std::endl;
    std::cout << "Number of CUDA devices: " << deviceCount << std::endl;

    for (int i = 0; i < deviceCount; i++)
    {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);

        std::cout << "\nDevice " << i << ":" << std::endl;
        std::cout << "Name: " << prop.name << std::endl;
        std::cout << "Compute Capability: " 
                  << prop.major << "." << prop.minor << std::endl;
        std::cout << "Global Memory: " 
                  << prop.totalGlobalMem / (1024 * 1024) << " MB" << std::endl;
    }

    return 0;
}