#include "hip/hip_runtime.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include "vgpu.h"

#define DATA_SIZE (100 * 1024 * 1024)
static unsigned char data[DATA_SIZE];

#define HIST_SIZE 256
static unsigned int hist[HIST_SIZE];

__global__ void histogram(unsigned char *data, unsigned int *hist) {
    __shared__ unsigned int temp[HIST_SIZE];

    int idx = threadIdx.x;
    int stride = blockDim.x;
    for (int i = idx; i < HIST_SIZE; i += stride) {
        temp[i] = 0;
    }

    __syncthreads();

    idx += blockIdx.x * blockDim.x;
    stride *= gridDim.x;
    for (int i = idx; i < DATA_SIZE; i += stride) {
        atomicAdd(&temp[data[i]], 1);
    }

    __syncthreads();

    idx = threadIdx.x;
    stride = blockDim.x;
    for (int i = idx; i < HIST_SIZE; i += stride) {
        atomicAdd(&hist[i], temp[i]);
    }
}

int main(void) {
    unsigned char *d_data;
    unsigned int *d_hist;

    for (int i = 0; i < DATA_SIZE; i++) {
        data[i] = rand() % 256;
    }

    CHECK(hipMalloc((void **)&d_data, DATA_SIZE * sizeof(unsigned char)));
    CHECK(hipMalloc((void **)&d_hist, HIST_SIZE * sizeof(unsigned int)));

    CHECK(hipMemcpy(d_data, data, DATA_SIZE * sizeof(unsigned char), hipMemcpyHostToDevice));

    histogram<<<dim3(10), dim3(8)>>>(d_data, d_hist);

    CHECK(hipMemcpy(hist, d_hist, HIST_SIZE * sizeof(unsigned int), hipMemcpyDeviceToHost));

    for (int i = 0; i < HIST_SIZE; i += 8) {
        for (int j = 0; j < 8; j++) {
            printf("%8d", hist[i + j]);
        }
        printf("\n");
    }

    // Validate results

    for (int i = 0; i < DATA_SIZE; i++) {
        hist[data[i]]--;
    }

    for (int i = 0; i < HIST_SIZE; i++) {
        assert(hist[i] == 0);
    }

    CHECK(hipFree(d_data));
    CHECK(hipFree(d_hist));

    return 0;
}

// CHECK:  409256  409078  409228  410256  409156  408915  409159  409307
// CHECK:  408413  409049  408838  409789  409254  409558  410144  408698
// CHECK:  409229  409805  408984  410069  408055  410042  409824  410016
// CHECK:  409698  409180  410609  410285  408951  409623  409227  410218
// CHECK:  409964  409382  408983  409772  409826  409244  408857  409044
// CHECK:  409685  410214  410102  409575  410509  410366  410429  409070
// CHECK:  409428  408004  408861  409487  410222  409780  408505  409677
// CHECK:  409269  409200  409934  409402  409499  409211  409744  409092
// CHECK:  409927  408305  409792  409896  409140  410217  409929  409554
// CHECK:  409117  410498  408538  409532  408419  409973  409899  409954
// CHECK:  410616  408462  411184  410056  409692  410818  411064  409105
// CHECK:  409923  410444  408966  410053  409864  409615  409261  409182
// CHECK:  409661  409518  409876  409888  409005  409701  410134  409151
// CHECK:  408286  409937  409040  410015  410309  409748  409759  409521
// CHECK:  409854  409177  409159  408992  410454  409485  408736  409048
// CHECK:  408581  409207  409812  409014  409138  408645  409605  410593
// CHECK:  410524  409549  411518  409337  409687  409548  409129  409329
// CHECK:  409148  408475  408687  410140  410034  410716  409046  409415
// CHECK:  410063  410762  410245  408908  409182  409712  410257  410007
// CHECK:  409455  409815  409584  410028  410455  409666  411170  410744
// CHECK:  409174  409933  409442  409154  410575  410085  409106  410318
// CHECK:  409903  408968  409838  410103  409782  410647  410427  409476
// CHECK:  409753  409922  408872  409307  409979  409620  409712  409761
// CHECK:  409490  408994  409499  408853  409138  409493  409842  409836
// CHECK:  409437  410168  409844  409638  410042  409622  409734  409283
// CHECK:  410407  410418  408582  409808  410121  411402  409820  409750
// CHECK:  408614  408432  409943  409827  410262  410048  408863  409894
// CHECK:  409208  409388  409665  408771  410503  409194  409191  410190
// CHECK:  408821  409460  410190  409759  409282  408612  408339  409574
// CHECK:  409484  409454  409988  409905  409651  409168  409921  409236
// CHECK:  408719  409356  408243  409413  410093  409160  409770  410231
// CHECK:  408661  410413  409868  409116  410851  409046  409191  410925

