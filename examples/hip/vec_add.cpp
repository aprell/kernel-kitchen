#include "hip/hip_runtime.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include "vgpu.h"

__global__ void vec_add(const float *a, const float *b, float *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
#ifdef DEBUG
    #pragma omp critical
    printf("Thread(%2d,%2d)\n", blockIdx.x, threadIdx.x);
#endif
    // Grid-stride loop
    for (int i = idx; i < n; i += stride) {
        c[i] = a[i] + b[i];
    }
}

int main(void) {
    float *a, *b, *c;
    float *d_a, *d_b, *d_c;
    int n = 1 << 10;

    a = (float *)malloc(n * sizeof(float));
    b = (float *)malloc(n * sizeof(float));
    c = (float *)malloc(n * sizeof(float));

    CHECK(hipMalloc((void **)&d_a, n * sizeof(float)));
    CHECK(hipMalloc((void **)&d_b, n * sizeof(float)));
    CHECK(hipMalloc((void **)&d_c, n * sizeof(float)));

    for (int i = 0; i < n; i++) {
        a[i] = i;
        b[i] = n - i;
    }

    CHECK(hipMemcpy(d_a, a, n * sizeof(float), hipMemcpyHostToDevice));
    CHECK(hipMemcpy(d_b, b, n * sizeof(float), hipMemcpyHostToDevice));

    vec_add<<<dim3(8), dim3(8)>>>(d_a, d_b, d_c, n);

    CHECK(hipMemcpy(c, d_c, n * sizeof(float), hipMemcpyDeviceToHost));

    for (int i = 0; i < 5; i++) {
        printf("c[%d] = %.0f\n", i, c[i]);
    }

    puts("...");

    for (int i = n - 5; i < n; i++) {
        printf("c[%d] = %.0f\n", i, c[i]);
    }

    CHECK(hipFree(d_a));
    CHECK(hipFree(d_b));
    CHECK(hipFree(d_c));

    free(a);
    free(b);
    free(c);

    return 0;
}

// CHECK: c[0] = 1024
// CHECK: c[1] = 1024
// CHECK: c[2] = 1024
// CHECK: c[3] = 1024
// CHECK: c[4] = 1024
// ...
// CHECK: c[1019] = 1024
// CHECK: c[1020] = 1024
// CHECK: c[1021] = 1024
// CHECK: c[1022] = 1024
// CHECK: c[1023] = 1024

