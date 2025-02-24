#include "hip/hip_runtime.h"
#include <assert.h>
#include <stdio.h>
#include <string.h>
#include "vgpu.h"

__global__ void hello_1D() {
    printf("Hello from Thread %d\n", threadIdx.x);
}

__global__ void hello_2D() {
    printf("Hello from Thread (%d, %d)\n", threadIdx.x, threadIdx.y);
}

__global__ void hello_3D() {
    printf("Hello from Thread (%d, %d, %d)\n", threadIdx.x, threadIdx.y, threadIdx.z);
}

int main(void) {
    int count;

    CHECK(hipGetDeviceCount(&count));
    printf("Found %d device%s\n", count, count != 1 ? "s" : "");

    for (int i = 0; i < count; i++) {
        hipDeviceProp_t prop;
        CHECK(hipGetDeviceProperties(&prop, i));
        printf("Device %d name:\t%s\n", i, prop.name);
    }

    hello_1D<<<dim3(1), dim3(8)>>>();
    CHECK(hipGetLastError());

    hello_2D<<<dim3(1), dim3(4, 2)>>>();
    CHECK(hipGetLastError());

    hello_3D<<<dim3(1), dim3(4, 2, 1)>>>();
    CHECK(hipGetLastError());

    CHECK(hipDeviceSynchronize());

    return 0;
}

// CHECK-DAG: Hello from Thread 0
// CHECK-DAG: Hello from Thread 1
// CHECK-DAG: Hello from Thread 2
// CHECK-DAG: Hello from Thread 3
// CHECK-DAG: Hello from Thread 4
// CHECK-DAG: Hello from Thread 5
// CHECK-DAG: Hello from Thread 6
// CHECK-DAG: Hello from Thread 7

// CHECK-DAG: Hello from Thread (0, 0)
// CHECK-DAG: Hello from Thread (1, 0)
// CHECK-DAG: Hello from Thread (2, 0)
// CHECK-DAG: Hello from Thread (3, 0)
// CHECK-DAG: Hello from Thread (0, 1)
// CHECK-DAG: Hello from Thread (1, 1)
// CHECK-DAG: Hello from Thread (2, 1)
// CHECK-DAG: Hello from Thread (3, 1)

// CHECK-DAG: Hello from Thread (0, 0, 0)
// CHECK-DAG: Hello from Thread (1, 0, 0)
// CHECK-DAG: Hello from Thread (2, 0, 0)
// CHECK-DAG: Hello from Thread (3, 0, 0)
// CHECK-DAG: Hello from Thread (0, 1, 0)
// CHECK-DAG: Hello from Thread (1, 1, 0)
// CHECK-DAG: Hello from Thread (2, 1, 0)
// CHECK-DAG: Hello from Thread (3, 1, 0)

