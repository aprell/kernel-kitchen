#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include "vgpu.h"

struct complex {
    // a + ib
    float a, b;
};

static inline __device__ float magnitude2(struct complex c) {
    return c.a * c.a + c.b * c.b;
}

static inline __device__ struct complex add(struct complex c, struct complex d) {
    return (struct complex){c.a + d.a, c.b + d.b};
}

static inline __device__ struct complex mul(struct complex c, struct complex d) {
    // (a + ib) * (c + id) = (ac - bd) + i(ad + bc)
    return (struct complex){c.a * d.a - c.b * d.b, c.a * d.b + c.b * d.a};
}

#define DIM 100

__device__ int julia(int x, int y) {
    const float scale = 1.5;
    float jx = scale * (float)(DIM/2 - x) / (DIM/2);
    float jy = scale * (float)(DIM/2 - y) / (DIM/2);

    struct complex c = {-0.8, 0.156};
    struct complex a = {jx, jy};

    for (int i = 0; i < 200; i++) {
        a = add(mul(a, a), c);
        if (magnitude2(a) > 1000) {
            return 0;
        }
    }

    return 1;
}

__global__ void kernel(unsigned char *img) {
    int yidx = blockIdx.y * blockDim.y + threadIdx.y;
    int xidx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int y = yidx; y < DIM; y += blockDim.y * gridDim.y) {
        for (int x = xidx; x < DIM; x += blockDim.x * gridDim.x) {
            int o = y * DIM + x;
            /* R */ img[o * 4 + 0] = 255 * julia(x, y);
            /* G */ img[o * 4 + 1] = 0;
            /* B */ img[o * 4 + 2] = 0;
            /* A */ img[o * 4 + 3] = 255;
        }
    }
}

void display(unsigned char *img) {
    for (int y = 0; y < DIM; y++) {
        for (int x = 0; x < DIM; x++) {
            int o = y * DIM + x;
            printf(img[o * 4 + 0] == 255 ? "." : " ");
        }
        printf("\n");
    }
}

int main(void) {
    unsigned char *img = (unsigned char *)malloc(DIM * DIM * sizeof(int));
    assert(img);

    unsigned char *d_img;
    CHECK(cudaMalloc((void **)&d_img, DIM * DIM * sizeof(int)));

    dim3 thread_blocks = dim3(DIM / 10, DIM / 10);
    dim3 threads_per_block = dim3(4, 2);
    kernel<<<thread_blocks, threads_per_block>>>(d_img);

    CHECK(cudaMemcpy(img, d_img, DIM * DIM * sizeof(int), cudaMemcpyDeviceToHost));

    display(img);

    CHECK(cudaFree(d_img));

    free(img);

    return 0;
}

// CHECK:                                                     .
// CHECK:                                                   .
// CHECK:                                                  .  .
// CHECK:                                                   ... .
// CHECK:                                                       .
// CHECK:                                                   .
// CHECK:                                               .
// CHECK:                                                .     .           .
// CHECK:                                           .   .          . .     .
// CHECK:                                              . ..       .    .
// CHECK:                                                          ...
// CHECK:                                                .       ...   .  ..
// CHECK:                                          .  . .. ..    .   .    ..
// CHECK:                                          .   ..         . . ..... .           . ..
// CHECK:                                            .  .  .             . .               .  .
// CHECK:                                           . .    . .....  .        .            .
// CHECK:                                           ...  . .   .         .                 .  ..
// CHECK:                                         . ..... .   .      .         .     . ....  ...
// CHECK:                      .  .                  ....                     . .      ... ... .
// CHECK:                  .     .. . .           .. ....   ..  . .             .     ... .    ..
// CHECK:                  ..    .... .               ....    .  . .                 .   . .
// CHECK:                   . .  .      .              .. .    .                        . .  .         ....
// CHECK:                   ..   .       . ..        .. . .     .          .             .        .    .
// CHECK:                  ..     ..        ..        .    ..   .          .     .        .  .            .
// CHECK:                 ....     .        ..      .  . .. ..            .      .    ..  .      .     .    .
// CHECK:       . .  .    ... .. . .           .                 .                          .         .
// CHECK:           ..     ... . .                          .                          . . ...     ..
// CHECK:         .         .                          .                 .           . . .. ...    .  . .
// CHECK:   .    .     .      .  ..    .      .            .. .. .  .      ..        .     ....
// CHECK:     .            .  .        .     .          .   ..    .        ..        ..     ..
// CHECK:        .    .        .             .          .     . . ..        .. .       .   ..
// CHECK:     ....         .  . .                        .    . ..              .      .  . .
// CHECK:                    . .   .                 . .  .    ....               . ....    ..
// CHECK:               ..    . ...     .             . .  ..   .... ..           . . ..     .
// CHECK:                . ... ...      . .                     ....                  .  .
// CHECK:                ...  .... .     .         .      .   . ..... .
// CHECK:                ..  .                 .         .   . .  ...
// CHECK:                     .            .        .  ..... .    . .
// CHECK:                 .  .               . .             .  .  .
// CHECK:                    .. .           . ..... . .         ..   .
// CHECK:                                    ..    .   .    .. .. .  .
// CHECK:                                    ..  .   ...       .
// CHECK:                                          ...
// CHECK:                                        .    .       .. .
// CHECK:                                    .     . .          .   .
// CHECK:                                    .           .     .
// CHECK:                                                       .
// CHECK:                                                   .
// CHECK:                                               .
// CHECK:                                               . ...
// CHECK:                                                 .  .
// CHECK:                                                   .
// CHECK:                                                 .

