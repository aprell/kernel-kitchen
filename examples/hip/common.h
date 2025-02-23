#ifndef COMMON_H
#define COMMON_H

#ifdef __VGPU__
  #define CHECK(call) call
#else
  #define CHECK(call) { \
    hipError_t error = call; \
    if (error != hipSuccess) { \
        fprintf(stderr, "Error: %s: '%s' at %s:%d\n", \
                #call, hipGetErrorString(call), __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
}
#endif

#define ceil_div(x, y) (((x) + (y) - 1) / (y))

#endif // COMMON_H
