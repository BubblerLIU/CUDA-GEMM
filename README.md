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

CUDA events measure execution time, and the test reports average latency and TFLOPS. Timing excludes memory allocation, host-device transfers, and CPU reference computation. 

Results are compared elementwise against the CPU implementation with the tolerance 
$$
\lvert \text{GPU} - \text{CPU} \rvert \leq 10^{-3}+10^{-3}\lvert\text{CPU}\rvert
$$

Throughput is calculated as
$$
\frac{2MNK}{\text{average latency(ms)}\times 10^{9}}\ \text{TFLOPS}
$$

For the same matrix dimensions:
$$
\text{Speedup} = \frac{\text{Naive average latency}}{\text{current implementation average latency  }}
$$

| M | N | K | Naive latency（ms） | Tiled latency（ms） | Tiled TFLOPS | Speedup |
|---:|---:|---:|---:|---:|---:|---:|
| 1024 | 1024 | 1024 | 0.305 | 0.226 | 9.52 | 1.35 |
| 1024 | 1024 | 2048 | 0.608 | 0.448 | 9.59 | 1.36 |
| 2048 | 2048 | 1024 | 1.149 | 0.853 | 10.07 | 1.35 |
| 1023 | 1025 | 1027 | 0.302 | 0.229 | 9.39 | 1.32 |

Test environment
```text
GPU: NVIDIA GeForce RTX 5090
CUDA Version: 13.0
Compute Capability: 12.0
CPU: INTEL(R) XEON(R) GOLD 6530
OS: Ubuntu 22.04.5 LTS
```