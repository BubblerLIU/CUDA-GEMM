#include <cstdio>
#include <cstdlib>
#include <random>
#include <cuda_runtime.h>

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

void random_init(float* data, int size)
{
    static std::mt19937 gen(17);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (int i = 0; i < size; ++i) {
        data[i] = dist(gen);
    }
}

void gemm_cpu(
    const float *A, const float *B, float *C, int M, int N, int K
) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0;
            for (int k = 0; k < K; ++k) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * M + j] = sum;
        }
    }
}

bool check_result(
    const float *gpu, const float *cpu, int size,
    float atol = 1e-3f, float rtol = 1e-3f
) {
    for (int i = 0; i < size; ++i) {
        float diff = std::fabs(gpu[i] - cpu[i]);
        float tolerance = atol + rtol * std::fabs(cpu[i]);
        if (diff > tolerance) {
            printf("Mismatch at %d: got %f, std %f\n", i, cpu[i], gpu[i]);
            return false;
        }
    }

    printf("PASS\n");
    return true;
}