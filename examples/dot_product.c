#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include "vgpu.h"

#define BLOCK_SIZE 8

KERNEL(dot, (const float *a, const float *b, float *c, int n),
    __shared__ float sdata[BLOCK_SIZE];)
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    float tmp = 0;
    for (int i = gid; i < n; i += stride) {
        tmp += a[i] * b[i];
    }

    sdata[tid] = tmp;
    __syncthreads();

    for (int i = blockDim.x / 2; i > 0; i /= 2) {
        if (tid < i) {
            sdata[tid] += sdata[tid + i];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(c, sdata[0]);
    }
END_KERNEL

int main(void) {
    float *a, *b, c = 0;
    float *d_a, *d_b, *d_c;
    int n = 1 << 10;

    a = (float *)malloc(n * sizeof(float));
    b = (float *)malloc(n * sizeof(float));

    CHECK(cudaMalloc((void **)&d_a, n * sizeof(float)));
    CHECK(cudaMalloc((void **)&d_b, n * sizeof(float)));
    CHECK(cudaMalloc((void **)&d_c, sizeof(float)));

    for (int i = 0; i < n; i++) {
        a[i] = 1;
        b[i] = i + 1;
    }

    CHECK(cudaMemcpy(d_a, a, n * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_b, b, n * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_c, &c, sizeof(float), cudaMemcpyHostToDevice));

    dot(/* <<< */ dim3(8), dim3(BLOCK_SIZE) /* >>> */, d_a, d_b, d_c, n);

    CHECK(cudaMemcpy(&c, d_c, sizeof(float), cudaMemcpyDeviceToHost));

    printf("%.0f\n", c);

    CHECK(cudaFree(d_a));
    CHECK(cudaFree(d_b));
    CHECK(cudaFree(d_c));

    free(a);
    free(b);

    return 0;
}

// CHECK: 524800
