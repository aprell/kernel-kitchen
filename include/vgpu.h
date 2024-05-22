#ifndef VGPU_H
#define VGPU_H

#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include "common.h"

typedef struct {
    int x, y, z;
} dim3;

// Number of thread blocks
static dim3 gridDim;

// Number of threads per block
static dim3 blockDim;

// Index of thread block in grid
static dim3 blockIdx;

// Index of thread in block
static dim3 threadIdx;

#define VA_NARGS(...) VA_NARGS_(__VA_ARGS__, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1)
#define VA_NARGS_(_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, N, ...) N

#define dim3(...) dim3_(VA_NARGS(__VA_ARGS__), __VA_ARGS__)
#define dim3_(n, ...) dim3__(n, __VA_ARGS__)
#define dim3__(n, ...) dim3__##n(__VA_ARGS__)

#define dim3__1(x_) (dim3){.x = (x_), .y = 1, .z = 1}
#define dim3__2(x_, y_) (dim3){.x = (x_), .y = (y_), .z = 1}
#define dim3__3(x_, y_, z_) (dim3){.x = (x_), .y = (y_), .z = (z_)}

#define ARGS(...) dim3 gridDim_, dim3 blockDim_, ##__VA_ARGS__

#ifdef DEBUG
#define IS_BLOCK_000 (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0)
#define IS_THREAD_000 (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0)
#define CHECK_KERNEL_LAUNCH(name) \
do { \
    if (IS_BLOCK_000 && IS_THREAD_000) { \
        printf("Launching kernel " #name "<<<dim3(%d, %d, %d), dim3(%d, %d, %d)>>>\n", \
               gridDim.x, gridDim.y, gridDim.z, blockDim.x, blockDim.y, blockDim.z); \
    } \
    assert(gridDim.x * gridDim.y == omp_get_num_teams() && gridDim.z == 1); \
    assert(blockDim.x * blockDim.y == omp_get_team_size(1) && blockDim.z == 1); \
    assert(0 <= blockIdx.x && blockIdx.x < gridDim.x); \
    assert(0 <= blockIdx.y && blockIdx.y < gridDim.y); \
    assert(0 <= threadIdx.x && threadIdx.x < blockDim.x); \
    assert(0 <= threadIdx.y && threadIdx.y < blockDim.y); \
} while (0)
#else
#define CHECK_KERNEL_LAUNCH(_)
#endif

#define KERNEL(...) KERNEL_(VA_NARGS(__VA_ARGS__), __VA_ARGS__)
#define KERNEL_(n, ...) KERNEL__(n, __VA_ARGS__)
#define KERNEL__(n, ...) KERNEL__##n(__VA_ARGS__)

#define KERNEL__2(name, args) KERNEL__2_(name, ARGS args)
#define KERNEL__2_(name, ...) KERNEL__2__(name, __VA_ARGS__)
#define KERNEL__2__(name, ...) \
void name(__VA_ARGS__) { \
    gridDim = gridDim_; \
    blockDim = blockDim_; \
    unsigned int thread_blocks = gridDim.x * gridDim.y; \
    unsigned int threads_per_block = blockDim.x * blockDim.y; \
    _Pragma("omp target teams num_teams(thread_blocks) thread_limit(threads_per_block) private(blockIdx)") { \
    blockIdx = dim3(omp_get_team_num() % gridDim.x, omp_get_team_num() / gridDim.x, 0); \
    _Pragma("omp parallel num_threads(threads_per_block) private(threadIdx)") { \
    threadIdx = dim3(omp_get_thread_num() % blockDim.x, omp_get_thread_num() / blockDim.x , 0); \
    CHECK_KERNEL_LAUNCH(name); \
    __syncthreads();

#define KERNEL__3(name, args, decls) KERNEL__3_(name, decls, ARGS args)
#define KERNEL__3_(name, decls, ...) KERNEL__3__(name, decls, __VA_ARGS__)
#define KERNEL__3__(name, decls, ...) \
void name(__VA_ARGS__) { \
    gridDim = gridDim_; \
    blockDim = blockDim_; \
    unsigned int thread_blocks = gridDim.x * gridDim.y; \
    unsigned int threads_per_block = blockDim.x * blockDim.y; \
    _Pragma("omp target teams num_teams(thread_blocks) thread_limit(threads_per_block) private(blockIdx)") { \
    blockIdx = dim3(omp_get_team_num() % gridDim.x, omp_get_team_num() / gridDim.x, 0); \
    decls \
    _Pragma("omp parallel num_threads(threads_per_block) private(threadIdx)") { \
    threadIdx = dim3(omp_get_thread_num() % blockDim.x, omp_get_thread_num() / blockDim.x , 0); \
    CHECK_KERNEL_LAUNCH(name); \
    __syncthreads();

#define END_KERNEL }}}

#define __device__

#define __host__

#define __shared__

#define __constant__

#define __syncthreads() _Pragma("omp barrier")

#define atomicAdd(addr, value) ({ \
  typeof(*(addr)) $t0; \
  _Pragma("omp atomic capture") \
  { $t0 = *(addr); *(addr) += value; } \
  $t0; \
})

#define atomicSub(addr, value) atomicAdd(addr, -(value))
#define atomicInc(addr)        atomicAdd(addr, 1)
#define atomicDec(addr)        atomicSub(addr, 1)

typedef enum cudaMemcpyKind {
    cudaMemcpyHostToHost = 0,
    cudaMemcpyHostToDevice = 1,
    cudaMemcpyDeviceToHost = 2,
    cudaMemcpyDeviceToDevice = 3,
    cudaMemcpyDefault = 4
} cudaMemcpyKind;

//
// Ignore return values of type `cudaError_t`
//

static inline void cudaMalloc(void **devPtr, size_t size) {
    *devPtr = malloc(size);
}

static inline void cudaFree(void *devPtr) {
    free(devPtr);
}

static inline void cudaMemcpy(void *dst, const void *src, size_t count, cudaMemcpyKind kind) {
    assert(dst != NULL);
    assert(src != NULL);
    (void)kind;
    memcpy(dst, src, count);
}

#define cudaMemcpyToSymbol(dst, src, count) \
    cudaMemcpy((void *)(dst), src, count, cudaMemcpyHostToDevice)

static inline void cudaGetDeviceCount(int *count) {
    if (count != NULL) {
        *count = 1;
    }
}

typedef struct {
    char name[256];
    // TODO
} cudaDeviceProp;

static inline void cudaGetDeviceProperties(cudaDeviceProp *prop, int device) {
    if (prop != NULL && device == 0) {
        strncpy(prop->name, "VGPU", strlen("VGPU") + 1);
    }
}

static inline void cudaDeviceSynchronize(void) {
}

__attribute__((constructor))
static inline void vgpu_init(void) {
    setenv("KMP_TEAMS_THREAD_LIMIT", "1024", /* overwrite = */ 1);
}

#endif // VGPU_H
