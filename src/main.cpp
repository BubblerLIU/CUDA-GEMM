#include "common.h"
#include "gemm.h"

int main() {
    test_gemm("naive", launch_gemm_naive, 1024, 1024, 1024);
    return 0;
}