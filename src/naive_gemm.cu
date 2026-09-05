#include "common.h"

__global__ void gemm_naive(
    const float *A, const float *B, float *C, int M, int N, int K
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

int main()
{
    const int M = 4;
    const int K = 4;
    const int N = 4;

    size_t bytes_A = M * K * sizeof(float);
    size_t bytes_B = K * N * sizeof(float);
    size_t bytes_C = M * N * sizeof(float);

    float* h_A = new float[M * K];
    float* h_B = new float[K * N];
    float* h_C = new float[M * N];

    for (int i = 0; i < M * K; ++i) {
        h_A[i] = 1.0f;
    }

    for (int i = 0; i < K * N; ++i) {
        h_B[i] = 1.0f;
    }


    float* d_A;
    float* d_B;
    float* d_C;

    CUDA_CHECK(cudaMalloc(&d_A, bytes_A));
    CUDA_CHECK(cudaMalloc(&d_B, bytes_B));
    CUDA_CHECK(cudaMalloc(&d_C, bytes_C));


    CUDA_CHECK(cudaMemcpy(
        d_A,
        h_A,
        bytes_A,
        cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(
        d_B,
        h_B,
        bytes_B,
        cudaMemcpyHostToDevice));


    dim3 block(16, 16);

    dim3 grid(
        (N + block.x - 1) / block.x,
        (M + block.y - 1) / block.y
    );


    gemm_naive<<<grid, block>>>(
        d_A,
        d_B,
        d_C,
        M,
        N,
        K
    );

    CUDA_CHECK_KERNEL();

    CUDA_CHECK(cudaMemcpy(
        h_C,
        d_C,
        bytes_C,
        cudaMemcpyDeviceToHost));


    for (int i = 0; i < M; ++i) {

        for (int j = 0; j < N; ++j) {
            std::cout << h_C[i * N + j] << " ";
        }

        std::cout << "\n";
    }


    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    delete[] h_A;
    delete[] h_B;
    delete[] h_C;

    return 0;
}