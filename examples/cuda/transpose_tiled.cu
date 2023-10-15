#include "matrix.h"
#include "vgpu.h"

#define TILE_WIDTH 2

__global__ void transpose_read_write_coalesced(float *A, float *AT, int m, int n) {
    __shared__ float A_tile[TILE_WIDTH][TILE_WIDTH];


    #define A(i, j)  A [(i) * n + (j)] // m x n
    #define AT(i, j) AT[(i) * m + (j)] // n x m

    int i = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int j = blockIdx.x * TILE_WIDTH + threadIdx.x;

    if (i < m && j < n) {
        int ii = threadIdx.y;
        int jj = threadIdx.x;
        A_tile[jj][ii] = A(i, j);
        __syncthreads();
        i = blockIdx.x * TILE_WIDTH + threadIdx.y;
        j = blockIdx.y * TILE_WIDTH + threadIdx.x;
        if (i < n && j < m) {
            AT(i, j) = A_tile[ii][jj];
        }
    }

    #undef A
    #undef AT
}

int main(void) {
    int m = 10, n = 40;
    float **A = (float **)malloc_matrix(m, n);
    float **AT = (float **)malloc_matrix(n, m);

    float *d_A, *d_AT;
    cudaMalloc((void **)&d_A, m * n * sizeof(float));
    cudaMalloc((void **)&d_AT, n * m * sizeof(float));

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            A[i][j] = i * n + j;
        }
    }

    cudaMemcpy(d_A, A[0], m * n * sizeof(float), cudaMemcpyHostToDevice);

    dim3 thread_blocks = dim3(ceil_div(n, TILE_WIDTH), ceil_div(m, TILE_WIDTH));
    dim3 threads_per_block = dim3(TILE_WIDTH, TILE_WIDTH);
    transpose_read_write_coalesced<<<thread_blocks, threads_per_block>>>(d_A, d_AT, m, n);

    cudaMemcpy(AT[0], d_AT, n * m * sizeof(float), cudaMemcpyDeviceToHost);

    print_matrix(AT, n, m);

    cudaFree(d_A);
    cudaFree(d_AT);

    free_matrix(A);
    free_matrix(AT);

    return 0;
}

// CHECK:   0.0    40.0    80.0   120.0   160.0   200.0   240.0   280.0   320.0   360.0
// CHECK:   1.0    41.0    81.0   121.0   161.0   201.0   241.0   281.0   321.0   361.0
// CHECK:   2.0    42.0    82.0   122.0   162.0   202.0   242.0   282.0   322.0   362.0
// CHECK:   3.0    43.0    83.0   123.0   163.0   203.0   243.0   283.0   323.0   363.0
// CHECK:   4.0    44.0    84.0   124.0   164.0   204.0   244.0   284.0   324.0   364.0
// CHECK:   5.0    45.0    85.0   125.0   165.0   205.0   245.0   285.0   325.0   365.0
// CHECK:   6.0    46.0    86.0   126.0   166.0   206.0   246.0   286.0   326.0   366.0
// CHECK:   7.0    47.0    87.0   127.0   167.0   207.0   247.0   287.0   327.0   367.0
// CHECK:   8.0    48.0    88.0   128.0   168.0   208.0   248.0   288.0   328.0   368.0
// CHECK:   9.0    49.0    89.0   129.0   169.0   209.0   249.0   289.0   329.0   369.0
// CHECK:  10.0    50.0    90.0   130.0   170.0   210.0   250.0   290.0   330.0   370.0
// CHECK:  11.0    51.0    91.0   131.0   171.0   211.0   251.0   291.0   331.0   371.0
// CHECK:  12.0    52.0    92.0   132.0   172.0   212.0   252.0   292.0   332.0   372.0
// CHECK:  13.0    53.0    93.0   133.0   173.0   213.0   253.0   293.0   333.0   373.0
// CHECK:  14.0    54.0    94.0   134.0   174.0   214.0   254.0   294.0   334.0   374.0
// CHECK:  15.0    55.0    95.0   135.0   175.0   215.0   255.0   295.0   335.0   375.0
// CHECK:  16.0    56.0    96.0   136.0   176.0   216.0   256.0   296.0   336.0   376.0
// CHECK:  17.0    57.0    97.0   137.0   177.0   217.0   257.0   297.0   337.0   377.0
// CHECK:  18.0    58.0    98.0   138.0   178.0   218.0   258.0   298.0   338.0   378.0
// CHECK:  19.0    59.0    99.0   139.0   179.0   219.0   259.0   299.0   339.0   379.0
// CHECK:  20.0    60.0   100.0   140.0   180.0   220.0   260.0   300.0   340.0   380.0
// CHECK:  21.0    61.0   101.0   141.0   181.0   221.0   261.0   301.0   341.0   381.0
// CHECK:  22.0    62.0   102.0   142.0   182.0   222.0   262.0   302.0   342.0   382.0
// CHECK:  23.0    63.0   103.0   143.0   183.0   223.0   263.0   303.0   343.0   383.0
// CHECK:  24.0    64.0   104.0   144.0   184.0   224.0   264.0   304.0   344.0   384.0
// CHECK:  25.0    65.0   105.0   145.0   185.0   225.0   265.0   305.0   345.0   385.0
// CHECK:  26.0    66.0   106.0   146.0   186.0   226.0   266.0   306.0   346.0   386.0
// CHECK:  27.0    67.0   107.0   147.0   187.0   227.0   267.0   307.0   347.0   387.0
// CHECK:  28.0    68.0   108.0   148.0   188.0   228.0   268.0   308.0   348.0   388.0
// CHECK:  29.0    69.0   109.0   149.0   189.0   229.0   269.0   309.0   349.0   389.0
// CHECK:  30.0    70.0   110.0   150.0   190.0   230.0   270.0   310.0   350.0   390.0
// CHECK:  31.0    71.0   111.0   151.0   191.0   231.0   271.0   311.0   351.0   391.0
// CHECK:  32.0    72.0   112.0   152.0   192.0   232.0   272.0   312.0   352.0   392.0
// CHECK:  33.0    73.0   113.0   153.0   193.0   233.0   273.0   313.0   353.0   393.0
// CHECK:  34.0    74.0   114.0   154.0   194.0   234.0   274.0   314.0   354.0   394.0
// CHECK:  35.0    75.0   115.0   155.0   195.0   235.0   275.0   315.0   355.0   395.0
// CHECK:  36.0    76.0   116.0   156.0   196.0   236.0   276.0   316.0   356.0   396.0
// CHECK:  37.0    77.0   117.0   157.0   197.0   237.0   277.0   317.0   357.0   397.0
// CHECK:  38.0    78.0   118.0   158.0   198.0   238.0   278.0   318.0   358.0   398.0
// CHECK:  39.0    79.0   119.0   159.0   199.0   239.0   279.0   319.0   359.0   399.0

