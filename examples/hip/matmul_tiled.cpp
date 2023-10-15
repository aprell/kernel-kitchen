#include "hip/hip_runtime.h"
#include "matrix.h"
#include "vgpu.h"

#define TILE_WIDTH 2

__global__ void matmul(float *A, float *B, float *C, int m, int n, int p) {
    __shared__ float A_tile[TILE_WIDTH][TILE_WIDTH];
    __shared__ float B_tile[TILE_WIDTH][TILE_WIDTH];


    #define A(i, j) A[(i) * n + (j)] // m x n
    #define B(i, j) B[(i) * p + (j)] // n x p
    #define C(i, j) C[(i) * p + (j)] // m x p

    int i = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int j = blockIdx.x * TILE_WIDTH + threadIdx.x;

    if (i < m && j < p) {
        int ii = threadIdx.y;
        int jj = threadIdx.x;
        float t = 0;
        for (int x = 0; x < n / TILE_WIDTH; x++) {
            // Load tiles of A and B
            A_tile[ii][jj] = A(i, x * TILE_WIDTH + jj);
            B_tile[ii][jj] = B(x * TILE_WIDTH + ii, j);
            __syncthreads();
            // Compute partial dot product
            for (int k = 0; k < TILE_WIDTH; k++) {
                t += A_tile[ii][k] * B_tile[k][jj];
            }
            __syncthreads();
        }
        C(i, j) = t;
    }

    #undef A
    #undef B
    #undef C
}

int main(void) {
    // Matrix dimensions must be a multiple of TILE_WIDTH
    int m = 40, n = 10, p = n;
    float **A = (float **)malloc_matrix(m, n);
    float **B = (float **)malloc_matrix(n, p);
    float **C = (float **)malloc_matrix(m, p);

    float *d_A, *d_B, *d_C;
    hipMalloc((void **)&d_A, m * n * sizeof(float));
    hipMalloc((void **)&d_B, n * p * sizeof(float));
    hipMalloc((void **)&d_C, m * p * sizeof(float));

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            A[i][j] = i * n + j;
        }
    }

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < p; j++) {
            B[i][j] = (j >= i) ? 1 : 0;
        }
    }

    hipMemcpy(d_A, A[0], m * n * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_B, B[0], n * p * sizeof(float), hipMemcpyHostToDevice);

    dim3 thread_blocks = dim3(ceil_div(n, TILE_WIDTH), ceil_div(m, TILE_WIDTH));
    dim3 threads_per_block = dim3(TILE_WIDTH, TILE_WIDTH);
    matmul<<<thread_blocks, threads_per_block>>>(d_A, d_B, d_C, m, n, p);

    hipMemcpy(C[0], d_C, m * p * sizeof(float), hipMemcpyDeviceToHost);

    print_matrix(C, m, p);

    hipFree(d_A);
    hipFree(d_B);
    hipFree(d_C);

    free_matrix(A);
    free_matrix(B);
    free_matrix(C);

    return 0;
}

// CHECK:    0.0     1.0     3.0     6.0    10.0    15.0    21.0    28.0    36.0    45.0
// CHECK:   10.0    21.0    33.0    46.0    60.0    75.0    91.0   108.0   126.0   145.0
// CHECK:   20.0    41.0    63.0    86.0   110.0   135.0   161.0   188.0   216.0   245.0
// CHECK:   30.0    61.0    93.0   126.0   160.0   195.0   231.0   268.0   306.0   345.0
// CHECK:   40.0    81.0   123.0   166.0   210.0   255.0   301.0   348.0   396.0   445.0
// CHECK:   50.0   101.0   153.0   206.0   260.0   315.0   371.0   428.0   486.0   545.0
// CHECK:   60.0   121.0   183.0   246.0   310.0   375.0   441.0   508.0   576.0   645.0
// CHECK:   70.0   141.0   213.0   286.0   360.0   435.0   511.0   588.0   666.0   745.0
// CHECK:   80.0   161.0   243.0   326.0   410.0   495.0   581.0   668.0   756.0   845.0
// CHECK:   90.0   181.0   273.0   366.0   460.0   555.0   651.0   748.0   846.0   945.0
// CHECK:  100.0   201.0   303.0   406.0   510.0   615.0   721.0   828.0   936.0  1045.0
// CHECK:  110.0   221.0   333.0   446.0   560.0   675.0   791.0   908.0  1026.0  1145.0
// CHECK:  120.0   241.0   363.0   486.0   610.0   735.0   861.0   988.0  1116.0  1245.0
// CHECK:  130.0   261.0   393.0   526.0   660.0   795.0   931.0  1068.0  1206.0  1345.0
// CHECK:  140.0   281.0   423.0   566.0   710.0   855.0  1001.0  1148.0  1296.0  1445.0
// CHECK:  150.0   301.0   453.0   606.0   760.0   915.0  1071.0  1228.0  1386.0  1545.0
// CHECK:  160.0   321.0   483.0   646.0   810.0   975.0  1141.0  1308.0  1476.0  1645.0
// CHECK:  170.0   341.0   513.0   686.0   860.0  1035.0  1211.0  1388.0  1566.0  1745.0
// CHECK:  180.0   361.0   543.0   726.0   910.0  1095.0  1281.0  1468.0  1656.0  1845.0
// CHECK:  190.0   381.0   573.0   766.0   960.0  1155.0  1351.0  1548.0  1746.0  1945.0
// CHECK:  200.0   401.0   603.0   806.0  1010.0  1215.0  1421.0  1628.0  1836.0  2045.0
// CHECK:  210.0   421.0   633.0   846.0  1060.0  1275.0  1491.0  1708.0  1926.0  2145.0
// CHECK:  220.0   441.0   663.0   886.0  1110.0  1335.0  1561.0  1788.0  2016.0  2245.0
// CHECK:  230.0   461.0   693.0   926.0  1160.0  1395.0  1631.0  1868.0  2106.0  2345.0
// CHECK:  240.0   481.0   723.0   966.0  1210.0  1455.0  1701.0  1948.0  2196.0  2445.0
// CHECK:  250.0   501.0   753.0  1006.0  1260.0  1515.0  1771.0  2028.0  2286.0  2545.0
// CHECK:  260.0   521.0   783.0  1046.0  1310.0  1575.0  1841.0  2108.0  2376.0  2645.0
// CHECK:  270.0   541.0   813.0  1086.0  1360.0  1635.0  1911.0  2188.0  2466.0  2745.0
// CHECK:  280.0   561.0   843.0  1126.0  1410.0  1695.0  1981.0  2268.0  2556.0  2845.0
// CHECK:  290.0   581.0   873.0  1166.0  1460.0  1755.0  2051.0  2348.0  2646.0  2945.0
// CHECK:  300.0   601.0   903.0  1206.0  1510.0  1815.0  2121.0  2428.0  2736.0  3045.0
// CHECK:  310.0   621.0   933.0  1246.0  1560.0  1875.0  2191.0  2508.0  2826.0  3145.0
// CHECK:  320.0   641.0   963.0  1286.0  1610.0  1935.0  2261.0  2588.0  2916.0  3245.0
// CHECK:  330.0   661.0   993.0  1326.0  1660.0  1995.0  2331.0  2668.0  3006.0  3345.0
// CHECK:  340.0   681.0  1023.0  1366.0  1710.0  2055.0  2401.0  2748.0  3096.0  3445.0
// CHECK:  350.0   701.0  1053.0  1406.0  1760.0  2115.0  2471.0  2828.0  3186.0  3545.0
// CHECK:  360.0   721.0  1083.0  1446.0  1810.0  2175.0  2541.0  2908.0  3276.0  3645.0
// CHECK:  370.0   741.0  1113.0  1486.0  1860.0  2235.0  2611.0  2988.0  3366.0  3745.0
// CHECK:  380.0   761.0  1143.0  1526.0  1910.0  2295.0  2681.0  3068.0  3456.0  3845.0
// CHECK:  390.0   781.0  1173.0  1566.0  1960.0  2355.0  2751.0  3148.0  3546.0  3945.0

