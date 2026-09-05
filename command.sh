#!/bin/bash
nvcc -O3 -I ./include src/naive_gemm.cu -o bin/naive_gemm
./bin/naive_gemm