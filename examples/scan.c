#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include "vgpu.h"

#define HILLIS_STEELE_TILE_SIZE 10
#define BLELLOCH_TILE_SIZE 10
#define BLELLOCH_TILE_SIZE_PADDED 16  // Smallest power of two >= BLELLOCH_TILE_SIZE

// Hillis-Steele inclusive scan
KERNEL(scan_hillis_steele, (const float *in, float *out, int n),
    __shared__ float buf[2][HILLIS_STEELE_TILE_SIZE];)
    int tid = threadIdx.x;
    int x = 0;
    int y = x ^ 1;

    buf[x][tid] = (tid < n) ? in[tid] : 0;
    __syncthreads();

    for (int stride = 1; stride < n; stride *= 2) {
        if (tid >= stride) {
            buf[y][tid] = buf[x][tid] + buf[x][tid - stride];
        } else {
            buf[y][tid] = buf[x][tid];
        }
        __syncthreads();
        x = y;
        y = x ^ 1;
    }

    if (tid < n) {
        out[tid] = buf[x][tid];
    }
END_KERNEL

// Blelloch exclusive scan
KERNEL(scan_blelloch, (const float *in, float *out, int n),
    __shared__ float buf[BLELLOCH_TILE_SIZE_PADDED];)
    int tid = threadIdx.x;

    // Use smallest power of two >= n
    int m = 1;
    while (m < n) m *= 2;

    for (int i = tid; i < m; i += blockDim.x) {
        buf[i] = (i < n) ? in[i] : 0;
    }
    __syncthreads();

    // Up-sweep
    for (int stride = 1; stride < m; stride *= 2) {
        int i = 2 * stride * (tid + 1) - 1;
        if (i < m) {
            buf[i] += buf[i - stride];
        }
        __syncthreads();
    }

    // Clear the root
    if (tid == 0) {
        buf[m - 1] = 0;
    }
    __syncthreads();

    // Down-sweep
    for (int stride = m / 2; stride > 0; stride /= 2) {
        int i = 2 * stride * (tid + 1) - 1;
        if (i < m) {
            float tmp = buf[i];
            buf[i] += buf[i - stride];
            buf[i - stride] = tmp;
        }
        __syncthreads();
    }

    for (int i = tid; i < n; i += blockDim.x) {
        out[i] = buf[i];
        // Inclusive scan:
        // out[i] += in[i];
    }
END_KERNEL

int main(void) {
    float *a, *d_a;
    float *b, *d_b;
    int n = HILLIS_STEELE_TILE_SIZE;

    a = (float *)malloc(n * sizeof(float));
    b = (float *)malloc(n * sizeof(float));

    CHECK(cudaMalloc((void **)&d_a, n * sizeof(float)));
    CHECK(cudaMalloc((void **)&d_b, n * sizeof(float)));

    for (int i = 0; i < n; i++) {
        a[i] = 1;
    }

    CHECK(cudaMemcpy(d_a, a, n * sizeof(float), cudaMemcpyHostToDevice));

    LAUNCH(scan_hillis_steele, dim3(1), dim3(HILLIS_STEELE_TILE_SIZE), d_a, d_b, n);

    CHECK(cudaMemcpy(b, d_b, n * sizeof(float), cudaMemcpyDeviceToHost));

    for (int i = 0; i < n; i++) {
        printf("%.0f ", b[i]);
    }

    printf("\n");

    // CHECK: 1 2 3 4 5 6 7 8 9 10

    // Reuse arrays
    n = BLELLOCH_TILE_SIZE;

    LAUNCH(scan_blelloch, dim3(1), dim3(BLELLOCH_TILE_SIZE), d_a, d_b, n);

    CHECK(cudaMemcpy(b, d_b, n * sizeof(float), cudaMemcpyDeviceToHost));

    for (int i = 0; i < n; i++) {
        printf("%.0f ", b[i]);
    }

    printf("\n");

    // CHECK: 0 1 2 3 4 5 6 7 8 9

    CHECK(cudaFree(d_a));
    CHECK(cudaFree(d_b));

    free(a);
    free(b);

    return 0;
}
