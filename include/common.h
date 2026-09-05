#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <cuda_runtime.h>

#define CUDA_CHECK(expr_to_check) do {            \
    cudaError_t result  = expr_to_check;          \
    if(result != cudaSuccess)                     \
    {                                             \
        fprintf(stderr,                           \
                "CUDA Runtime Error: %s:%i:%d = %s\n", \
                __FILE__,                         \
                __LINE__,                         \
                result,\
                cudaGetErrorString(result));      \
    }                                             \
} while(0)

#define CUDA_CHECK_KERNEL() do {                \
    CUDA_CHECK(cudaGetLastError());             \
    CUDA_CHECK(cudaDeviceSynchronize());        \
} while(0)
