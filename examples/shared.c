#include <stdio.h>
#include "vgpu.h"

KERNEL(init, (),
    __shared__ int a[8];)
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (threadIdx.x == 0) {
        for (int i = 0; i < 8; i++) {
            a[i] = 0;
        }
    }

    __syncthreads();

    if (idx < 8) {
        a[idx] = idx;
    }

    __syncthreads();

    if (threadIdx.x == 0) {
        for (int i = 0; i < 8; i++) {
            printf("Block %d: %d\n", blockIdx.x, a[i]);
        }
    }
END_KERNEL

int main(void) {
    init(/* <<< */ dim3(2), dim3(4) /* >>> */);

    cudaDeviceSynchronize();

    return 0;
}

// CHECK-DAG: Block 0: 0
// CHECK-DAG: Block 0: 1
// CHECK-DAG: Block 0: 2
// CHECK-DAG: Block 0: 3
// CHECK-DAG: Block 0: 0
// CHECK-DAG: Block 0: 0
// CHECK-DAG: Block 0: 0
// CHECK-DAG: Block 0: 0

// CHECK-DAG: Block 1: 0
// CHECK-DAG: Block 1: 0
// CHECK-DAG: Block 1: 0
// CHECK-DAG: Block 1: 0
// CHECK-DAG: Block 1: 4
// CHECK-DAG: Block 1: 5
// CHECK-DAG: Block 1: 6
// CHECK-DAG: Block 1: 7
