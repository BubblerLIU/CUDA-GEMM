#include <cuda_runtime.h>

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

void launch_gemm_naive(
    const float *A, const float *B, float *C, int M, int N, int K
) {
    dim3 block(32, 8);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
    gemm_naive<<<grid, block>>>(A, B, C, M, N, K);
}