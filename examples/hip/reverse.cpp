#include "hip/hip_runtime.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include "vgpu.h"

__global__ void reverse(float *a, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // Grid-stride loop
    for (int i = idx; i < n/2; i += stride) {
        float t = a[i];
        a[i] = a[n - 1 - i];
        a[n - 1 - i] = t;
    }
}

int main(void) {
    float *a, *d_a;
    int n = 1 << 10;

    a = (float *)malloc(n * sizeof(float));
    assert(a);

    CHECK(hipMalloc((void **)&d_a, n * sizeof(float)));
    assert(d_a);

    for (int i = 0; i < n; i++) {
        a[i] = i + 1;
    }

    CHECK(hipMemcpy(d_a, a, n * sizeof(float), hipMemcpyHostToDevice));

    reverse<<<dim3(8), dim3(4)>>>(d_a, n);

    CHECK(hipMemcpy(a, d_a, n * sizeof(float), hipMemcpyDeviceToHost));

    for (int i = 0; i < 5; i++) {
        printf("a[%d] = %.0f\n", i, a[i]);
    }

    puts("...");

    for (int i = n - 5; i < n; i++) {
        printf("a[%d] = %.0f\n", i, a[i]);
    }

    CHECK(hipFree(d_a));

    free(a);

    return 0;
}

// CHECK: a[0] = 1024
// CHECK: a[1] = 1023
// CHECK: a[2] = 1022
// CHECK: a[3] = 1021
// CHECK: a[4] = 1020
// ...
// CHECK: a[1019] = 5
// CHECK: a[1020] = 4
// CHECK: a[1021] = 3
// CHECK: a[1022] = 2
// CHECK: a[1023] = 1

