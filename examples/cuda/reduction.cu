#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include "vgpu.h"

#define BLOCK_SIZE 8

__global__ void reduce(int n, const float *in, float *out) {
    __shared__ float sdata[BLOCK_SIZE];

    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    sdata[tid] = 0;

    // Thread-local reduction
    for (int i = gid; i < n; i += stride) {
        sdata[tid] += in[i];
    }

    __syncthreads();

    // Block-local tree reduction
    for (int i = blockDim.x / 2; i > 0; i /= 2) {
        if (tid < i) {
            sdata[tid] += sdata[tid + i];
        }
        __syncthreads();
    }



    // Global reduction
    if (tid == 0) {
        atomicAdd(out, sdata[0]);
    }
}

int main(void) {
    float *a, *d_a;
    float sum = 0;
    int n = 1 << 16;

    a = (float *)malloc(n * sizeof(float));
    assert(a);

    CHECK(cudaMalloc((void **)&d_a, (n + 1) * sizeof(float)));
    assert(d_a);

    for (int i = 0; i < n; i++) {
        a[i] = 1;
    }

    CHECK(cudaMemcpy(d_a, a, n * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_a + n, &sum, sizeof(float), cudaMemcpyHostToDevice));

    reduce<<<dim3(10), dim3(BLOCK_SIZE)>>>(n, d_a, d_a + n);

    CHECK(cudaMemcpy(&sum, d_a + n, sizeof(float), cudaMemcpyDeviceToHost));

    printf("%0.f\n", sum);
    assert(sum == n);

    CHECK(cudaFree(d_a));

    free(a);

    return 0;
}


//            +----
// CHECK:     65536

