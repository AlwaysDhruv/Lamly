cmake_minimum_required(VERSION 3.20)

project(GPT CUDA CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)

find_package(CUDAToolkit REQUIRED)

add_executable(app
    main.cu
    src/Tensor.cu
)

target_include_directories(app PRIVATE
    include
)

target_link_libraries(app PRIVATE
    CUDA::curand
)