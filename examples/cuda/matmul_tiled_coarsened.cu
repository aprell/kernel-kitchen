#include "matrix.h"
#include "vgpu.h"

#define TILE_WIDTH 2

// Thread coarsening
#define COARSENING_FACTOR 4

__global__ void matmul(float *A, float *B, float *C, int m, int n, int p) {
    __shared__ float A_tile[TILE_WIDTH][TILE_WIDTH];
    __shared__ float B_tile[TILE_WIDTH][TILE_WIDTH];


    #define A(i, j) A[(i) * n + (j)] // m x n
    #define B(i, j) B[(i) * p + (j)] // n x p
    #define C(i, j) C[(i) * p + (j)] // m x p

    int i = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int j = blockIdx.x * TILE_WIDTH * COARSENING_FACTOR + threadIdx.x;

    if (i < m && j < p) {
        int ii = threadIdx.y;
        int jj = threadIdx.x;
        float t[COARSENING_FACTOR] = {0};
        for (int x = 0; x < n / TILE_WIDTH; x++) {
            // Load tiles of A and B
            A_tile[ii][jj] = A(i, x * TILE_WIDTH + jj);
            for (int c = 0; c < COARSENING_FACTOR; c++) {
                B_tile[ii][jj] = B(x * TILE_WIDTH + ii, j + c * TILE_WIDTH);
                __syncthreads();
                // Compute partial dot product
                for (int k = 0; k < TILE_WIDTH; k++) {
                    t[c] += A_tile[ii][k] * B_tile[k][jj];
                }
                __syncthreads();
            }
        }
        for (int c = 0; c < COARSENING_FACTOR; c++) {
            C(i, j + c * TILE_WIDTH) = t[c];
        }
    }

    #undef A
    #undef B
    #undef C
}

int main(void) {
    // Matrix dimensions must be a multiple of TILE_WIDTH
    int m = 40, n = 16, p = n;
    float **A = (float **)malloc_matrix(m, n);
    float **B = (float **)malloc_matrix(n, p);
    float **C = (float **)malloc_matrix(m, p);

    float *d_A, *d_B, *d_C;
    cudaMalloc((void **)&d_A, m * n * sizeof(float));
    cudaMalloc((void **)&d_B, n * p * sizeof(float));
    cudaMalloc((void **)&d_C, m * p * sizeof(float));

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

    cudaMemcpy(d_A, A[0], m * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B[0], n * p * sizeof(float), cudaMemcpyHostToDevice);

    dim3 thread_blocks = dim3(ceil_div(n, TILE_WIDTH), ceil_div(m, TILE_WIDTH));
    dim3 threads_per_block = dim3(TILE_WIDTH, TILE_WIDTH);
    matmul<<<thread_blocks, threads_per_block>>>(d_A, d_B, d_C, m, n, p);

    cudaMemcpy(C[0], d_C, m * p * sizeof(float), cudaMemcpyDeviceToHost);

    print_matrix(C, m, p);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    free_matrix(A);
    free_matrix(B);
    free_matrix(C);

    return 0;
}

// CHECK:    0.0     1.0     3.0     6.0    10.0    15.0    21.0    28.0    36.0    45.0    55.0    66.0    78.0    91.0   105.0   120.0
// CHECK:   16.0    33.0    51.0    70.0    90.0   111.0   133.0   156.0   180.0   205.0   231.0   258.0   286.0   315.0   345.0   376.0
// CHECK:   32.0    65.0    99.0   134.0   170.0   207.0   245.0   284.0   324.0   365.0   407.0   450.0   494.0   539.0   585.0   632.0
// CHECK:   48.0    97.0   147.0   198.0   250.0   303.0   357.0   412.0   468.0   525.0   583.0   642.0   702.0   763.0   825.0   888.0
// CHECK:   64.0   129.0   195.0   262.0   330.0   399.0   469.0   540.0   612.0   685.0   759.0   834.0   910.0   987.0  1065.0  1144.0
// CHECK:   80.0   161.0   243.0   326.0   410.0   495.0   581.0   668.0   756.0   845.0   935.0  1026.0  1118.0  1211.0  1305.0  1400.0
// CHECK:   96.0   193.0   291.0   390.0   490.0   591.0   693.0   796.0   900.0  1005.0  1111.0  1218.0  1326.0  1435.0  1545.0  1656.0
// CHECK:  112.0   225.0   339.0   454.0   570.0   687.0   805.0   924.0  1044.0  1165.0  1287.0  1410.0  1534.0  1659.0  1785.0  1912.0
// CHECK:  128.0   257.0   387.0   518.0   650.0   783.0   917.0  1052.0  1188.0  1325.0  1463.0  1602.0  1742.0  1883.0  2025.0  2168.0
// CHECK:  144.0   289.0   435.0   582.0   730.0   879.0  1029.0  1180.0  1332.0  1485.0  1639.0  1794.0  1950.0  2107.0  2265.0  2424.0
// CHECK:  160.0   321.0   483.0   646.0   810.0   975.0  1141.0  1308.0  1476.0  1645.0  1815.0  1986.0  2158.0  2331.0  2505.0  2680.0
// CHECK:  176.0   353.0   531.0   710.0   890.0  1071.0  1253.0  1436.0  1620.0  1805.0  1991.0  2178.0  2366.0  2555.0  2745.0  2936.0
// CHECK:  192.0   385.0   579.0   774.0   970.0  1167.0  1365.0  1564.0  1764.0  1965.0  2167.0  2370.0  2574.0  2779.0  2985.0  3192.0
// CHECK:  208.0   417.0   627.0   838.0  1050.0  1263.0  1477.0  1692.0  1908.0  2125.0  2343.0  2562.0  2782.0  3003.0  3225.0  3448.0
// CHECK:  224.0   449.0   675.0   902.0  1130.0  1359.0  1589.0  1820.0  2052.0  2285.0  2519.0  2754.0  2990.0  3227.0  3465.0  3704.0
// CHECK:  240.0   481.0   723.0   966.0  1210.0  1455.0  1701.0  1948.0  2196.0  2445.0  2695.0  2946.0  3198.0  3451.0  3705.0  3960.0
// CHECK:  256.0   513.0   771.0  1030.0  1290.0  1551.0  1813.0  2076.0  2340.0  2605.0  2871.0  3138.0  3406.0  3675.0  3945.0  4216.0
// CHECK:  272.0   545.0   819.0  1094.0  1370.0  1647.0  1925.0  2204.0  2484.0  2765.0  3047.0  3330.0  3614.0  3899.0  4185.0  4472.0
// CHECK:  288.0   577.0   867.0  1158.0  1450.0  1743.0  2037.0  2332.0  2628.0  2925.0  3223.0  3522.0  3822.0  4123.0  4425.0  4728.0
// CHECK:  304.0   609.0   915.0  1222.0  1530.0  1839.0  2149.0  2460.0  2772.0  3085.0  3399.0  3714.0  4030.0  4347.0  4665.0  4984.0
// CHECK:  320.0   641.0   963.0  1286.0  1610.0  1935.0  2261.0  2588.0  2916.0  3245.0  3575.0  3906.0  4238.0  4571.0  4905.0  5240.0
// CHECK:  336.0   673.0  1011.0  1350.0  1690.0  2031.0  2373.0  2716.0  3060.0  3405.0  3751.0  4098.0  4446.0  4795.0  5145.0  5496.0
// CHECK:  352.0   705.0  1059.0  1414.0  1770.0  2127.0  2485.0  2844.0  3204.0  3565.0  3927.0  4290.0  4654.0  5019.0  5385.0  5752.0
// CHECK:  368.0   737.0  1107.0  1478.0  1850.0  2223.0  2597.0  2972.0  3348.0  3725.0  4103.0  4482.0  4862.0  5243.0  5625.0  6008.0
// CHECK:  384.0   769.0  1155.0  1542.0  1930.0  2319.0  2709.0  3100.0  3492.0  3885.0  4279.0  4674.0  5070.0  5467.0  5865.0  6264.0
// CHECK:  400.0   801.0  1203.0  1606.0  2010.0  2415.0  2821.0  3228.0  3636.0  4045.0  4455.0  4866.0  5278.0  5691.0  6105.0  6520.0
// CHECK:  416.0   833.0  1251.0  1670.0  2090.0  2511.0  2933.0  3356.0  3780.0  4205.0  4631.0  5058.0  5486.0  5915.0  6345.0  6776.0
// CHECK:  432.0   865.0  1299.0  1734.0  2170.0  2607.0  3045.0  3484.0  3924.0  4365.0  4807.0  5250.0  5694.0  6139.0  6585.0  7032.0
// CHECK:  448.0   897.0  1347.0  1798.0  2250.0  2703.0  3157.0  3612.0  4068.0  4525.0  4983.0  5442.0  5902.0  6363.0  6825.0  7288.0
// CHECK:  464.0   929.0  1395.0  1862.0  2330.0  2799.0  3269.0  3740.0  4212.0  4685.0  5159.0  5634.0  6110.0  6587.0  7065.0  7544.0
// CHECK:  480.0   961.0  1443.0  1926.0  2410.0  2895.0  3381.0  3868.0  4356.0  4845.0  5335.0  5826.0  6318.0  6811.0  7305.0  7800.0
// CHECK:  496.0   993.0  1491.0  1990.0  2490.0  2991.0  3493.0  3996.0  4500.0  5005.0  5511.0  6018.0  6526.0  7035.0  7545.0  8056.0
// CHECK:  512.0  1025.0  1539.0  2054.0  2570.0  3087.0  3605.0  4124.0  4644.0  5165.0  5687.0  6210.0  6734.0  7259.0  7785.0  8312.0
// CHECK:  528.0  1057.0  1587.0  2118.0  2650.0  3183.0  3717.0  4252.0  4788.0  5325.0  5863.0  6402.0  6942.0  7483.0  8025.0  8568.0
// CHECK:  544.0  1089.0  1635.0  2182.0  2730.0  3279.0  3829.0  4380.0  4932.0  5485.0  6039.0  6594.0  7150.0  7707.0  8265.0  8824.0
// CHECK:  560.0  1121.0  1683.0  2246.0  2810.0  3375.0  3941.0  4508.0  5076.0  5645.0  6215.0  6786.0  7358.0  7931.0  8505.0  9080.0
// CHECK:  576.0  1153.0  1731.0  2310.0  2890.0  3471.0  4053.0  4636.0  5220.0  5805.0  6391.0  6978.0  7566.0  8155.0  8745.0  9336.0
// CHECK:  592.0  1185.0  1779.0  2374.0  2970.0  3567.0  4165.0  4764.0  5364.0  5965.0  6567.0  7170.0  7774.0  8379.0  8985.0  9592.0
// CHECK:  608.0  1217.0  1827.0  2438.0  3050.0  3663.0  4277.0  4892.0  5508.0  6125.0  6743.0  7362.0  7982.0  8603.0  9225.0  9848.0
// CHECK:  624.0  1249.0  1875.0  2502.0  3130.0  3759.0  4389.0  5020.0  5652.0  6285.0  6919.0  7554.0  8190.0  8827.0  9465.0 10104.0

