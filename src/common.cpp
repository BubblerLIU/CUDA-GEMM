#include "common.h"

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

void test_gemm(char *prompt, GemmLauncher launcher, int M, int N, int K) {
    printf("TEST GEMM: %s\n", prompt);

    size_t bytes_A = M * K * sizeof(float);
    size_t bytes_B = K * N * sizeof(float);
    size_t bytes_C = M * N * sizeof(float);

    float* h_A = new float[M * K];
    float* h_B = new float[K * N];
    float* h_C_cpu = new float[M * N];
    float* h_C_gpu = new float[M * N];

    random_init(h_A, M * K);
    random_init(h_B, K * N);

    float* d_A;
    float* d_B;
    float* d_C;

    CUDA_CHECK(cudaMalloc(&d_A, bytes_A));
    CUDA_CHECK(cudaMalloc(&d_B, bytes_B));
    CUDA_CHECK(cudaMalloc(&d_C, bytes_C));

    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes_A, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes_B, cudaMemcpyHostToDevice));

    /* Compute CPU reference */
    gemm_cpu(h_A, h_B, h_C_cpu, M, N, K);

    /* Initialize CUDA context */
    CUDA_CHECK(cudaSetDevice(0));

    /* Warm up GPU */
    for (int i = 0; i < WARMUP; ++i) {
        launcher(d_A, d_B, d_C, M, N, K);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK_KERNEL();

    // Real Test
    GpuTimer timer;
    timer.start();

    for (int i = 0; i < REPEAT; ++i) {
        launcher(d_A, d_B, d_C, M, N, K);
    }

    float total_ms = timer.stop_ms();
    float avg_ms = total_ms / REPEAT;
    double gflops = 2.0 * M * N * K / (avg_ms * 1e6);
    printf("Average Latency: %f ms, Performance: %lf GFLOPS\n", avg_ms, gflops);

    // Check result
    CUDA_CHECK(cudaMemcpy(h_C_gpu, d_C, bytes_C, cudaMemcpyDeviceToHost));
    check_result(h_C_gpu, h_C_cpu, M * N);

    // Release memory
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    delete[] h_A;
    delete[] h_B;
    delete[] h_C_cpu;
    delete[] h_C_gpu;
}