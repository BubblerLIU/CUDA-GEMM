#!/bin/bash
nvcc -O3 ./src/test_gemm.cu -o ./bin/test
./bin/test