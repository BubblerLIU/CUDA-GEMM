#include "common.h"
#include "gemm.h"

int main() {
    float naive_latency, tiled_latency, speedup;
    int M, N, K;

    M = 1; N = 1; K = 1;
    printf("Matrix dimensions: M=%d, N=%d, K=%d\n", M, N, K);
    test_gemm("naive", launch_gemm_naive, M, N, K, &naive_latency);
    test_gemm("tiled", launch_gemm_tiled, M, N, K, &tiled_latency);
    speedup = naive_latency / tiled_latency;
    printf("Speed Up: %.2f\n", speedup);

    M = 17; N = 19; K = 23;
    printf("Matrix dimensions: M=%d, N=%d, K=%d\n", M, N, K);
    test_gemm("naive", launch_gemm_naive, M, N, K, &naive_latency);
    test_gemm("tiled", launch_gemm_tiled, M, N, K, &tiled_latency);
    speedup = naive_latency / tiled_latency;
    printf("Speed Up: %.2f\n", speedup);

    M = 127; N = 129; K = 131;
    printf("Matrix dimensions: M=%d, N=%d, K=%d\n", M, N, K);
    test_gemm("naive", launch_gemm_naive, M, N, K, &naive_latency);
    test_gemm("tiled", launch_gemm_tiled, M, N, K, &tiled_latency);
    speedup = naive_latency / tiled_latency;
    printf("Speed Up: %.2f\n", speedup);

    M = 1024; N =1024; K = 1024;
    printf("Matrix dimensions: M=%d, N=%d, K=%d\n", M, N, K);
    test_gemm("naive", launch_gemm_naive, M, N, K, &naive_latency);
    test_gemm("tiled", launch_gemm_tiled, M, N, K, &tiled_latency);
    speedup = naive_latency / tiled_latency;
    printf("Speed Up: %.2f\n", speedup);

    M = 1024; N = 1024; K = 2048;
    printf("Matrix dimensions: M=%d, N=%d, K=%d\n", M, N, K);
    test_gemm("naive", launch_gemm_naive, M, N, K, &naive_latency);
    test_gemm("tiled", launch_gemm_tiled, M, N, K, &tiled_latency);
    speedup = naive_latency / tiled_latency;
    printf("Speed Up: %.2f\n", speedup);

    M = 2048; N = 2048; K = 1024;
    printf("Matrix dimensions: M=%d, N=%d, K=%d\n", M, N, K);
    test_gemm("naive", launch_gemm_naive, M, N, K, &naive_latency);
    test_gemm("tiled", launch_gemm_tiled, M, N, K, &tiled_latency);
    speedup = naive_latency / tiled_latency;
    printf("Speed Up: %.2f\n", speedup);

    M = 1023; N =1025; K = 1027;
    printf("Matrix dimensions: M=%d, N=%d, K=%d\n", M, N, K);
    test_gemm("naive", launch_gemm_naive, M, N, K, &naive_latency);
    test_gemm("tiled", launch_gemm_tiled, M, N, K, &tiled_latency);
    speedup = naive_latency / tiled_latency;
    printf("Speed Up: %.2f\n", speedup);

    return 0;
}