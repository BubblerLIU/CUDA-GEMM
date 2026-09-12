# CUDA GEMM

A framework for learning and comparing CUDA GEMM optimizations. It implements row-major FP32 matrix multiplication `C = A × B`, where A, B, and C have shapes `M × K`, `K × N`, and `M × N`, respectively.

## Implementations

| Implementation | Approach |
| --- | --- |
| Naive | Each thread computes one output element by reading A and B directly from global memory, using `16 × 16` thread blocks. |
| Tiled | Reuses A and B data through `16 × 16` shared-memory tiles to reduce repeated global memory reads, with zero padding at boundaries. |

Both implementations include boundary checks and support matrix dimensions that are not multiples of 16.

## Project Structure

```text
include/           Kernel launch interfaces and utility declarations
src/main.cpp       Test entry point and parameter configuration
src/common.cpp     Data initialization, CPU reference checks, and test harness
src/naive_gemm.cu   Naive CUDA implementation
src/tiled_gemm.cu   Shared-memory tiled implementation
Makefile           Build and run
```

## Build and Run

```bash
make          # Build bin/gemm
make run      # Build and run
make clean    # Remove build artifacts
```

## Benchmark

Each implementation runs 10 warmup iterations followed by 100 timed iterations. Warmup and repetition counts can be changed in `include/common.h`.

CUDA events measure execution time, and the test reports average latency and GFLOPS. Timing excludes memory allocation, host-device transfers, and CPU reference computation. 

Results are compared elementwise against the CPU implementation with the tolerance 
$$
\lvert \text{GPU} - \text{CPU} \rvert \leq 10^{-3}+10^{-3}\lvert\text{CPU}\rvert
$$

Throughput is calculated as
$$
\frac{2MNK}{\text{average latency(ms)}\times 10^{6}}\ \text{GFLOPS}
$$

Test environment
```text
GPU: NVIDIA 5090
CPU:
OS:
```

For the same matrix dimensions:
$$
\text{Speedup} = \frac{\text{Naive average latency}}{\text{current implementation average latency  }}
$$

| M | N | K | Implementation | Average Latency (ms) | GFLOPS | Speedup vs. Naive | Correctness |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1024 | 1024 | 1024 | Naive | TBD | TBD | TBD | TBD |
| 1024 | 1024 | 1024 | Tiled | TBD | TBD | TBD | TBD |