#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include "vgpu.h"

#define BLOCK_SIZE 8

KERNEL(reduce, (int n, float *in, float *out),
    __shared__ float temp[BLOCK_SIZE];)
    int gidx = blockIdx.x * blockDim.x + threadIdx.x;
    int lidx = threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    temp[lidx] = 0;

    // Thread-local reduction
    for (int i = gidx; i < n; i += stride) {
        temp[lidx] += in[gidx];
    }

    __syncthreads();

    // Block-local tree reduction
    for (int i = blockDim.x / 2; i > 0; i /= 2) {
        if (lidx < i) {
            temp[lidx] += temp[lidx + i];
        }
        __syncthreads();
    }

#ifndef __CUDACC__
    if (lidx == 0) {
        #pragma omp critical
        {
            for (int i = 0; i < BLOCK_SIZE; i++) {
                printf("%5.0f ", temp[i]);
            }
            printf("\n");
        }
    }
#endif

    // Global reduction
    if (lidx == 0) {
        atomicAdd(out, temp[0]);
    }
END_KERNEL

int main(void) {
    float *a, *d_a;
    float sum = 0;
    int n = 1 << 16;

    a = (float *)malloc(n * sizeof(float));
    assert(a);

    cudaMalloc((void **)&d_a, (n + 1) * sizeof(float));
    assert(d_a);

    for (int i = 0; i < n; i++) {
        a[i] = 1;
    }

    cudaMemcpy(d_a, a, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_a + n, &sum, sizeof(float), cudaMemcpyHostToDevice);

    reduce(/* <<< */ dim3(10), dim3(BLOCK_SIZE) /* >>> */, n, d_a, d_a + n);

    cudaMemcpy(&sum, d_a + n, sizeof(float), cudaMemcpyDeviceToHost);

    printf("%0.f\n", sum);
    assert(sum == n);

    cudaFree(d_a);

    free(a);

    return 0;
}

// CHECK-DAG:  6560  3280  1640  1640   820   820   820   820
// CHECK-DAG:  6560  3280  1640  1640   820   820   820   820
// CHECK-DAG:  6552  3276  1638  1638   819   819   819   819
// CHECK-DAG:  6552  3276  1638  1638   819   819   819   819
// CHECK-DAG:  6552  3276  1638  1638   819   819   819   819
// CHECK-DAG:  6552  3276  1638  1638   819   819   819   819
// CHECK-DAG:  6552  3276  1638  1638   819   819   819   819
// CHECK-DAG:  6552  3276  1638  1638   819   819   819   819
// CHECK-DAG:  6552  3276  1638  1638   819   819   819   819
// CHECK-DAG:  6552  3276  1638  1638   819   819   819   819
//            +----
// CHECK:     65536
