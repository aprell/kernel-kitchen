#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include "vgpu.h"

#ifdef __CUDACC__
  #define restrict __restrict__
#endif

KERNEL(saxpy, (int n, float a, float *restrict x, float *restrict y))
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    #pragma omp critical
    printf("Thread(%2d,%2d)\n", blockIdx.x, threadIdx.x);
    // Grid-stride loop
    for (int i = idx; i < n; i += stride) {
        y[i] = a * x[i] + y[i];
    }
END_KERNEL

int main(void) {
    float *x, *y, *d_x, *d_y;
    int n = 1 << 10;

    x = (float *)malloc(n * sizeof(float));
    y = (float *)malloc(n * sizeof(float));
    assert(x && y);

    cudaMalloc((void **)&d_x, n * sizeof(float));
    cudaMalloc((void **)&d_y, n * sizeof(float));
    assert(d_x && d_y);

    for (int i = 0; i < n; i++) {
        x[i] = i;
        y[i] = n - i;
    }

    cudaMemcpy(d_x, x, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, n * sizeof(float), cudaMemcpyHostToDevice);

    saxpy(/* <<< */ dim3(8), dim3(8) /* >>> */, n, 2.0, d_x, d_y);

    cudaMemcpy(y, d_y, n * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < 5; i++) {
        printf("y[%d] = %.0f\n", i, y[i]);
    }

    puts("...");

    for (int i = n - 5; i < n; i++) {
        printf("y[%d] = %.0f\n", i, y[i]);
    }

    cudaFree(d_x);
    cudaFree(d_y);

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
