#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include "vgpu.h"

#define DATA_SIZE (100 * 1024 * 1024)
static unsigned char data[DATA_SIZE];

#define HIST_SIZE 256
static unsigned int hist[HIST_SIZE];

KERNEL(histogram, (unsigned char *data, unsigned int *hist),
    __shared__ unsigned int temp[HIST_SIZE];)
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
END_KERNEL

int main(void) {
    unsigned char *d_data;
    unsigned int *d_hist;

    srand(100);

    for (int i = 0; i < DATA_SIZE; i++) {
        data[i] = rand() % 256;
    }

    CHECK(cudaMalloc((void **)&d_data, DATA_SIZE * sizeof(unsigned char)));
    CHECK(cudaMalloc((void **)&d_hist, HIST_SIZE * sizeof(unsigned int)));

    CHECK(cudaMemcpy(d_data, data, DATA_SIZE * sizeof(unsigned char), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_hist, hist, HIST_SIZE * sizeof(unsigned int), cudaMemcpyHostToDevice));

    histogram(/* <<< */ dim3(10), dim3(8) /* >>> */, d_data, d_hist);

    CHECK(cudaMemcpy(hist, d_hist, HIST_SIZE * sizeof(unsigned int), cudaMemcpyDeviceToHost));

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

    CHECK(cudaFree(d_data));
    CHECK(cudaFree(d_hist));

    return 0;
}

// CHECK:  410202  409933  408767  410462  408828  409636  410029  408736
// CHECK:  409795  410503  409806  407977  409836  408978  410028  409611
// CHECK:  410297  410089  410288  409098  409641  409880  408857  409163
// CHECK:  409728  409976  409981  409821  409152  408906  409212  409884
// CHECK:  409843  409901  410103  409068  410012  410131  409146  409463
// CHECK:  410118  409086  409374  409739  409646  408896  410058  409383
// CHECK:  410134  409640  409743  409584  409088  410301  409098  409208
// CHECK:  409616  409787  410344  410015  409333  409908  410598  410476
// CHECK:  409709  409587  409750  409524  409540  410209  409595  408553
// CHECK:  410252  408609  409483  409016  410252  410092  410466  409554
// CHECK:  409076  409715  408293  409504  409086  409773  409704  409635
// CHECK:  409824  409768  410130  409232  410293  410013  409651  409678
// CHECK:  409305  409495  409270  410378  409412  410054  408493  409164
// CHECK:  409507  409694  408460  409389  411153  410809  409862  407864
// CHECK:  409426  409614  408936  410262  409440  410049  409304  409337
// CHECK:  410688  410576  409372  409349  409265  410519  409177  409388
// CHECK:  409362  411143  408495  410651  408979  410137  410102  409428
// CHECK:  409123  410263  409810  408598  409046  409388  410304  409235
// CHECK:  408656  409428  410332  410266  409584  410292  409191  410130
// CHECK:  408687  409834  409287  409284  409805  409659  409033  410617
// CHECK:  409901  410497  409006  410099  410094  409724  410483  409297
// CHECK:  409853  410010  410346  408856  409155  408994  409474  409638
// CHECK:  409147  410492  409356  409802  408595  410461  409206  409916
// CHECK:  408609  409648  408706  408582  408600  408413  409972  410274
// CHECK:  410040  410008  408100  409427  409654  409369  409165  409217
// CHECK:  410220  410222  409059  410281  409019  409716  409017  409642
// CHECK:  409463  409608  409885  409665  409777  409965  409337  409836
// CHECK:  410256  409332  409128  409388  410193  410057  409283  408455
// CHECK:  409949  409582  408744  409265  409501  409259  409582  409541
// CHECK:  410366  409959  409909  410698  409433  409967  408701  409291
// CHECK:  409443  409799  409939  408707  409753  410096  409522  408502
// CHECK:  409671  408349  409859  409321  409510  410094  409708  409924
