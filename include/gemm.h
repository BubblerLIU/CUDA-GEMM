#pragma once

void launch_gemm_naive(
    const float *A, const float *B, float *C, int M, int N, int K
);

void launch_gemm_tiled(
    const float *A, const float *B, float *C, int M, int N, int K
)