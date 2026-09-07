#pragma once

#include <cstdio>
#include <cstdlib>
#include <random>
#include <cuda_runtime.h>

/* Kernel Launcher */

using GemmLauncher = void (*)(
    const float*, const float*, float*, int, int, int
)

/* Global Parameter */

constexpr WARM = 10;
constexpr REPEAT = 100;

/* Error Checking */

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

/* Timer */

struct GpuTimer
{
    cudaEvent_t start_, stop_;
    GpuTimer() {
        CUDA_CHECK(cudaEventCreate(&start_));
        CUDA_CHECK(cudaEventCreate(&stop_));
    }
    ~GpuTimer() {
        CUDA_CHECK(cudaEventDestroy(start_));
        CUDA_CHECK(cudaEventDestroy(stop_));
    }
    void start() {
        CUDA_CHECK(cudaEventRecord(start_));
    }
    float stop_ms() {
        CUDA_CHECK(cudaEventRecord(stop_));
        CUDA_CHECK(cudaEventSynchronize(stop_));
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start_, stop_));
        return ms;
    }
};

/* Data Reference */

void random_init(float* data, int size);

void gemm_cpu(
    const float *A, const float *B, float *C, int M, int N, int K
);

bool check_result(
    const float *gpu, const float *cpu, int size,
    float atol = 1e-3f, float rtol = 1e-3f
);

/* Test GEMM */

void test_gemm(char *prompt, GemmLauncher launcher, int M, int N, int K);
