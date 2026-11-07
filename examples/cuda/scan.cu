#include <assert.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include "vgpu.h"

#define HILLIS_STEELE_TILE_SIZE 10
#define BLELLOCH_TILE_SIZE 10
// Smallest power of two >= BLELLOCH_TILE_SIZE
#define BLELLOCH_TILE_SIZE_PADDED 16

// Hillis-Steele inclusive scan (of one tile)
__global__ void scan_hillis_steele(const float *in, float *out, float *tile_sums, int n) {
    __shared__ float buf[2][HILLIS_STEELE_TILE_SIZE];

    int tid = threadIdx.x;
    int gid = blockIdx.x * HILLIS_STEELE_TILE_SIZE + tid;
    int x = 0;
    int y = x ^ 1;

    buf[x][tid] = (gid < n) ? in[gid] : 0;
    __syncthreads();

    for (int stride = 1; stride < HILLIS_STEELE_TILE_SIZE; stride *= 2) {
        if (tid >= stride) {
            buf[y][tid] = buf[x][tid] + buf[x][tid - stride];
        } else {
            buf[y][tid] = buf[x][tid];
        }
        __syncthreads();
        x = y;
        y = x ^ 1;
    }

    if (gid < n) {
        out[gid] = buf[x][tid];
    }

    if (tid == HILLIS_STEELE_TILE_SIZE - 1 && tile_sums != NULL) {
        tile_sums[blockIdx.x] = buf[x][tid];
    }
}

void test_scan_hillis_steele(int n) {
    float *a, *d_a;
    float *b, *d_b;

    a = (float *)malloc(n * sizeof(float));
    b = (float *)malloc(n * sizeof(float));

    CHECK(cudaMalloc((void **)&d_a, n * sizeof(float)));
    CHECK(cudaMalloc((void **)&d_b, n * sizeof(float)));

    for (int i = 0; i < n; i++) {
        a[i] = 1;
    }

    CHECK(cudaMemcpy(d_a, a, n * sizeof(float), cudaMemcpyHostToDevice));

    LAUNCH(scan_hillis_steele, dim3(1), dim3(HILLIS_STEELE_TILE_SIZE), d_a, d_b, NULL, n);

    CHECK(cudaMemcpy(b, d_b, n * sizeof(float), cudaMemcpyDeviceToHost));

    for (int i = 0; i < n; i++) {
        printf("%.0f ", b[i]);
    }

    printf("\n");

    CHECK(cudaFree(d_a));
    CHECK(cudaFree(d_b));

    free(a);
    free(b);
}

// Blelloch exclusive scan (of one tile)
__global__ void scan_blelloch(const float *in, float *out, float *tile_sums, int n) {
    __shared__ float buf[BLELLOCH_TILE_SIZE_PADDED];

    const int m = BLELLOCH_TILE_SIZE_PADDED;
    int tid = threadIdx.x;

    for (int i = tid; i < m; i += blockDim.x) {
        int gid = blockIdx.x * BLELLOCH_TILE_SIZE + i;
        buf[i] = (i < BLELLOCH_TILE_SIZE && gid < n) ? in[gid] : 0;
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

    // Clear the root (after saving the total)
    if (tid == 0) {
        if (tile_sums != NULL) {
            tile_sums[blockIdx.x] = buf[m - 1];
        }
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

    for (int i = tid; i < BLELLOCH_TILE_SIZE; i += blockDim.x) {
        int gid = blockIdx.x * BLELLOCH_TILE_SIZE + i;
        if (gid < n) {
            out[gid] = buf[i];
            // Inclusive scan:
            // out[gid] += in[gid];
        }
    }
}

void test_scan_blelloch(int n) {
    float *a, *d_a;
    float *b, *d_b;

    a = (float *)malloc(n * sizeof(float));
    b = (float *)malloc(n * sizeof(float));

    CHECK(cudaMalloc((void **)&d_a, n * sizeof(float)));
    CHECK(cudaMalloc((void **)&d_b, n * sizeof(float)));

    for (int i = 0; i < n; i++) {
        a[i] = 1;
    }

    CHECK(cudaMemcpy(d_a, a, n * sizeof(float), cudaMemcpyHostToDevice));

    LAUNCH(scan_blelloch, dim3(1), dim3(BLELLOCH_TILE_SIZE), d_a, d_b, NULL, n);

    CHECK(cudaMemcpy(b, d_b, n * sizeof(float), cudaMemcpyDeviceToHost));

    for (int i = 0; i < n; i++) {
        printf("%.0f ", b[i]);
    }

    printf("\n");

    CHECK(cudaFree(d_a));
    CHECK(cudaFree(d_b));

    free(a);
    free(b);
}

__global__ void uniform_add(const float *tile_offsets, float *out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] += tile_offsets[blockIdx.x];
    }
}

void scan_exclusive(const float *d_in, float *d_out, int n) {
    int num_tiles = ceil_div(n, BLELLOCH_TILE_SIZE);
    float *d_tile_sums, *d_tile_offsets;

    if (num_tiles == 1) {
        LAUNCH(scan_blelloch, dim3(1), dim3(BLELLOCH_TILE_SIZE), d_in, d_out, NULL, n);
        return;
    }

    CHECK(cudaMalloc((void **)&d_tile_sums, num_tiles * sizeof(float)));
    CHECK(cudaMalloc((void **)&d_tile_offsets, num_tiles * sizeof(float)));

    // 1. Scan each tile and collect totals
    LAUNCH(scan_blelloch, dim3(num_tiles), dim3(BLELLOCH_TILE_SIZE), d_in, d_out, d_tile_sums, n);

    // 2. Scan totals to obtain offsets (recursive)
    scan_exclusive(d_tile_sums, d_tile_offsets, num_tiles);

    // 3. Add each tile's offset to all of its elements (result of 1)
    LAUNCH(uniform_add, dim3(num_tiles), dim3(BLELLOCH_TILE_SIZE), d_tile_offsets, d_out, n);

    CHECK(cudaFree(d_tile_sums));
    CHECK(cudaFree(d_tile_offsets));
}

void scan_inclusive(const float *d_in, float *d_out, int n) {
    int num_tiles = ceil_div(n, HILLIS_STEELE_TILE_SIZE);
    float *d_tile_sums, *d_tile_offsets;

    CHECK(cudaMalloc((void **)&d_tile_sums, num_tiles * sizeof(float)));
    CHECK(cudaMalloc((void **)&d_tile_offsets, num_tiles * sizeof(float)));

    // 1. Scan each tile and collect totals
    LAUNCH(scan_hillis_steele, dim3(num_tiles), dim3(HILLIS_STEELE_TILE_SIZE), d_in, d_out, d_tile_sums, n);

    // 2. Scan totals to obtain offsets
    scan_exclusive(d_tile_sums, d_tile_offsets, num_tiles);

    // 3. Add each tile's offset to all of its elements (result of 1)
    LAUNCH(uniform_add, dim3(num_tiles), dim3(HILLIS_STEELE_TILE_SIZE), d_tile_offsets, d_out, n);

    CHECK(cudaFree(d_tile_sums));
    CHECK(cudaFree(d_tile_offsets));
}

void test_scan(int n, bool inclusive) {
    float *a, *d_a;
    float *b, *d_b;

    a = (float *)malloc(n * sizeof(float));
    b = (float *)malloc(n * sizeof(float));

    CHECK(cudaMalloc((void **)&d_a, n * sizeof(float)));
    CHECK(cudaMalloc((void **)&d_b, n * sizeof(float)));

    for (int i = 0; i < n; i++) {
        a[i] = 1;
    }

    CHECK(cudaMemcpy(d_a, a, n * sizeof(float), cudaMemcpyHostToDevice));

    #define SCAN (inclusive ? scan_inclusive : scan_exclusive)
    SCAN(d_a, d_b, n);
    #undef SCAN

    CHECK(cudaMemcpy(b, d_b, n * sizeof(float), cudaMemcpyDeviceToHost));

    for (int i = 0; i < n; i++) {
        printf("%3.0f ", b[i]);
        if ((i + 1) % 10 == 0) {
            printf("\n");
        }
    }

    CHECK(cudaFree(d_a));
    CHECK(cudaFree(d_b));

    free(a);
    free(b);
}

int main(void) {
    test_scan_hillis_steele(HILLIS_STEELE_TILE_SIZE);
    // CHECK: 1 2 3 4 5 6 7 8 9 10

    test_scan_blelloch(BLELLOCH_TILE_SIZE);
    // CHECK: 0 1 2 3 4 5 6 7 8 9

    test_scan(120, /* inclusive = */ true);
    // CHECK:   1   2   3   4   5   6   7   8   9  10
    // CHECK:  11  12  13  14  15  16  17  18  19  20
    // CHECK:  21  22  23  24  25  26  27  28  29  30
    // CHECK:  31  32  33  34  35  36  37  38  39  40
    // CHECK:  41  42  43  44  45  46  47  48  49  50
    // CHECK:  51  52  53  54  55  56  57  58  59  60
    // CHECK:  61  62  63  64  65  66  67  68  69  70
    // CHECK:  71  72  73  74  75  76  77  78  79  80
    // CHECK:  81  82  83  84  85  86  87  88  89  90
    // CHECK:  91  92  93  94  95  96  97  98  99 100
    // CHECK: 101 102 103 104 105 106 107 108 109 110
    // CHECK: 111 112 113 114 115 116 117 118 119 120

    test_scan(120, /* inclusive = */ false);
    // CHECK:   0   1   2   3   4   5   6   7   8   9
    // CHECK:  10  11  12  13  14  15  16  17  18  19
    // CHECK:  20  21  22  23  24  25  26  27  28  29
    // CHECK:  30  31  32  33  34  35  36  37  38  39
    // CHECK:  40  41  42  43  44  45  46  47  48  49
    // CHECK:  50  51  52  53  54  55  56  57  58  59
    // CHECK:  60  61  62  63  64  65  66  67  68  69
    // CHECK:  70  71  72  73  74  75  76  77  78  79
    // CHECK:  80  81  82  83  84  85  86  87  88  89
    // CHECK:  90  91  92  93  94  95  96  97  98  99
    // CHECK: 100 101 102 103 104 105 106 107 108 109
    // CHECK: 110 111 112 113 114 115 116 117 118 119

    return 0;
}

