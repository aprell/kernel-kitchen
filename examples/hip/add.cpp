#include "hip/hip_runtime.h"
#include <assert.h>
#include <stdio.h>
#include "vgpu.h"

__global__ void add(int *c, int a, int b) {
    *c = a + b;
}

int main(void) {
    int c, *d_c;

    hipMalloc((void **)&d_c, sizeof(int));
    assert(d_c);

    add<<<dim3(1), dim3(1)>>>(d_c, 2, 3);

    hipMemcpy(&c, d_c, sizeof(int), hipMemcpyDeviceToHost);

    printf("2 + 3 = %d\n", c);

    hipFree(d_c);

    return 0;
}

// CHECK: 2 + 3 = 5

