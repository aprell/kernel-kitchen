#include <stdio.h>
#include "vgpu.h"

#define N 8

__constant__ float numbers[N];

KERNEL(sum, (float *results))
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float res = 0;

    for (int i = 0; i < N; i++) {
        res += numbers[i];
    }

    results[idx] = res / (idx + 1);
END_KERNEL

int main(void) {
    float results[N];
    float *d_results;

    for (int i = 0; i < N; i++) {
        results[i] = i + 1;
    }

    CHECK(cudaMalloc((void **)&d_results, N * sizeof(float)));

    CHECK(cudaMemcpyToSymbol(numbers, results, N * sizeof(float)));

    sum(/* <<< */ dim3(2), dim3(4) /* >>> */, d_results);

    CHECK(cudaMemcpy(results, d_results, N * sizeof(float), cudaMemcpyDeviceToHost));

    for (int i = 0; i < N; i++) {
        printf("%6.2f\n", results[i]);
    }

    CHECK(cudaFree(d_results));

    return 0;
}

// CHECK: 36.00
// CHECK: 18.00
// CHECK: 12.00
// CHECK:  9.00
// CHECK:  7.20
// CHECK:  6.00
// CHECK:  5.14
// CHECK:  4.50
