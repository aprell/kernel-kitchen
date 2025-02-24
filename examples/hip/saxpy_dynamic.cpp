#include "hip/hip_runtime.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include "vgpu.h"

#ifdef __HIPCC__
  #define restrict __restrict__
#endif

// Adapted from Guray Ozen's PhD thesis, Chapter 6
// https://upcommons.upc.edu/handle/2117/125844

// /!\ A scalar doesn't work here because of OpenMP's data mapping
__device__ static int block[1];

__global__ void saxpy_dynamic(int n, float a, float *restrict x, float *restrict y) {
    __shared__ int next;

    while (1) {
        if (threadIdx.x == 0) {
            next = atomicAdd(block, 1);
        }
        __syncthreads();
        int i = next * blockDim.x + threadIdx.x;
        if (i >= n) break;
        y[i] = a * x[i] + y[i];
        __syncthreads();
    }
}

int main(void) {
    float *x, *y, *d_x, *d_y;
    int n = 1 << 10;

    x = (float *)malloc(n * sizeof(float));
    y = (float *)malloc(n * sizeof(float));
    assert(x && y);

    CHECK(hipMalloc((void **)&d_x, n * sizeof(float)));
    CHECK(hipMalloc((void **)&d_y, n * sizeof(float)));
    assert(d_x && d_y);

    for (int i = 0; i < n; i++) {
        x[i] = i;
        y[i] = n - i;
    }

    CHECK(hipMemcpy(d_x, x, n * sizeof(float), hipMemcpyHostToDevice));
    CHECK(hipMemcpy(d_y, y, n * sizeof(float), hipMemcpyHostToDevice));

    saxpy_dynamic<<<dim3(8), dim3(8)>>>(n, 2.0, d_x, d_y);

    CHECK(hipMemcpy(y, d_y, n * sizeof(float), hipMemcpyDeviceToHost));

    for (int i = 0; i < 5; i++) {
        printf("y[%d] = %.0f\n", i, y[i]);
    }

    puts("...");

    for (int i = n - 5; i < n; i++) {
        printf("y[%d] = %.0f\n", i, y[i]);
    }

    CHECK(hipFree(d_x));
    CHECK(hipFree(d_y));

    free(x);
    free(y);

    return 0;
}

// CHECK: y[0] = 1024
// CHECK: y[1] = 1025
// CHECK: y[2] = 1026
// CHECK: y[3] = 1027
// CHECK: y[4] = 1028
// ...
// CHECK: y[1019] = 2043
// CHECK: y[1020] = 2044
// CHECK: y[1021] = 2045
// CHECK: y[1022] = 2046
// CHECK: y[1023] = 2047

