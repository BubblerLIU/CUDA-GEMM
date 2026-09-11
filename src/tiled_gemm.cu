#include <cuda_runtime.h>

constexpr int TILE = 16;

__global__ void gemm_tiled(
    const float *A, const float *B, float *C, int M, int N, int K
) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    
}

void launch_gemm_tiled(
    const float *A, const float *B, float *C, int M, int N, int K
) {

}