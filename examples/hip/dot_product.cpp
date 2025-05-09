#include "hip/hip_runtime.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include "vgpu.h"

#define BLOCK_SIZE 8

__global__ void dot(const float *a, const float *b, float *c, int n) {
    __shared__ float temp[BLOCK_SIZE];

    int gidx = blockIdx.x * blockDim.x + threadIdx.x;
    int lidx = threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    float tmp = 0;

    for (int i = gidx; i < n; i += stride) {
        tmp += a[i] * b[i];
    }

    temp[lidx] = tmp;

    __syncthreads();

    for (int i = blockDim.x / 2; i > 0; i /= 2) {
        if (lidx < i) {
            temp[lidx] += temp[lidx + i];
        }
        __syncthreads();
    }

    if (lidx == 0) {
        atomicAdd(c, temp[0]);
    }
}

int main(void) {
    float *a, *b, c = 0;
    float *d_a, *d_b, *d_c;
    int n = 1 << 10;

    a = (float *)malloc(n * sizeof(float));
    b = (float *)malloc(n * sizeof(float));

    CHECK(hipMalloc((void **)&d_a, n * sizeof(float)));
    CHECK(hipMalloc((void **)&d_b, n * sizeof(float)));
    CHECK(hipMalloc((void **)&d_c, sizeof(float)));

    for (int i = 0; i < n; i++) {
        a[i] = 1;
        b[i] = i + 1;
    }

    CHECK(hipMemcpy(d_a, a, n * sizeof(float), hipMemcpyHostToDevice));
    CHECK(hipMemcpy(d_b, b, n * sizeof(float), hipMemcpyHostToDevice));
    CHECK(hipMemcpy(d_c, &c, sizeof(float), hipMemcpyHostToDevice));

    dot<<<dim3(8), dim3(BLOCK_SIZE)>>>(d_a, d_b, d_c, n);

    CHECK(hipMemcpy(&c, d_c, sizeof(float), hipMemcpyDeviceToHost));

    printf("%.0f\n", c);

    CHECK(hipFree(d_a));
    CHECK(hipFree(d_b));
    CHECK(hipFree(d_c));

    free(a);
    free(b);

    return 0;
}

// CHECK: 524800

