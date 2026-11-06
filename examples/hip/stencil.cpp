#include "hip/hip_runtime.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include "vgpu.h"

#define BLOCK_SIZE 8
#define RADIUS 3

__global__ void stencil_1D(const int *in, int *out) {
    __shared__ int sdata[BLOCK_SIZE + 2 * RADIUS];

    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = threadIdx.x + RADIUS;

    // Read data into shared memory
    sdata[idx] = in[gid];
    if (threadIdx.x < RADIUS) {
        sdata[idx - RADIUS] = in[gid - RADIUS];
        sdata[idx + BLOCK_SIZE] = in[gid + BLOCK_SIZE];
    }

    // if (idx >= BLOCK_SIZE) {
    //     sdata[idx + RADIUS] = in[gid + RADIUS];
    // }

    __syncthreads();

    // Apply stencil
    int res = 0;
    for (int i = -RADIUS; i <= RADIUS; i++) {
        res += sdata[idx + i];
    }

    // Write back result
    out[gid] = res;
}

int main(void) {
    int *a, *b, *d_a, *d_b;
    int n = 1 << 5;

    a = (int *)malloc((n + 2 * RADIUS)  * sizeof(int));
    b = (int *)malloc((n + 2 * RADIUS) * sizeof(int));
    assert(a && b);

    CHECK(hipMalloc((void **)&d_a, (n + 2 * RADIUS) * sizeof(int)));
    CHECK(hipMalloc((void **)&d_b, (n + 2 * RADIUS) * sizeof(int)));
    assert(d_a && d_b);

    for (int i = 0; i < n + 2 * RADIUS; i++) {
        a[i] = 1;
        b[i] = -1;
    }

    CHECK(hipMemcpy(d_a, a, (n + 2 * RADIUS) * sizeof(int), hipMemcpyHostToDevice));
    CHECK(hipMemcpy(d_b, a, (n + 2 * RADIUS) * sizeof(int), hipMemcpyHostToDevice));

    stencil_1D<<<dim3(n / BLOCK_SIZE), dim3(BLOCK_SIZE)>>>(d_a + RADIUS, d_b + RADIUS);

    CHECK(hipMemcpy(b, d_b, (n + 2 * RADIUS) * sizeof(int), hipMemcpyDeviceToHost));

    for (int i = RADIUS; i < RADIUS + 5; i++) {
        printf("b[%d] = %d\n", i, b[i]);
    }

    puts("...");

    for (int i = n + RADIUS - 5; i < n + RADIUS; i++) {
        printf("b[%d] = %d\n", i, b[i]);
    }

    CHECK(hipFree(d_a));
    CHECK(hipFree(d_b));

    free(a);
    free(b);

    return 0;
}

// CHECK: b[3] = 7
// CHECK: b[4] = 7
// CHECK: b[5] = 7
// CHECK: b[6] = 7
// CHECK: b[7] = 7
// ...
// CHECK: b[30] = 7
// CHECK: b[31] = 7
// CHECK: b[32] = 7
// CHECK: b[33] = 7
// CHECK: b[34] = 7

