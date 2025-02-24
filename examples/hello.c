#include <assert.h>
#include <stdio.h>
#include <string.h>
#include "vgpu.h"

KERNEL(hello_1D, ())
    printf("Hello from Thread %d\n", threadIdx.x);
END_KERNEL

KERNEL(hello_2D, ())
    printf("Hello from Thread (%d, %d)\n", threadIdx.x, threadIdx.y);
END_KERNEL

KERNEL(hello_3D, ())
    printf("Hello from Thread (%d, %d, %d)\n", threadIdx.x, threadIdx.y, threadIdx.z);
END_KERNEL

int main(void) {
    int count;

    CHECK(cudaGetDeviceCount(&count));
    printf("Found %d device%s\n", count, count != 1 ? "s" : "");

    for (int i = 0; i < count; i++) {
        cudaDeviceProp prop;
        CHECK(cudaGetDeviceProperties(&prop, i));
        printf("Device %d name:\t%s\n", i, prop.name);
    }

    hello_1D(/* <<< */ dim3(1), dim3(8) /* >>> */);
    CHECK(cudaGetLastError());

    hello_2D(/* <<< */ dim3(1), dim3(4, 2) /* >>> */);
    CHECK(cudaGetLastError());

    hello_3D(/* <<< */ dim3(1), dim3(4, 2, 1) /* >>> */);
    CHECK(cudaGetLastError());

    CHECK(cudaDeviceSynchronize());

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
