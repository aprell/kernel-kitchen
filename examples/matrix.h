#ifndef MATRIX_H
#define MATRIX_H

#include <assert.h>
#include <stdlib.h>
#include <stdio.h>

static inline float **malloc_matrix(int nrows, int ncols) {
    float **M = (float **)malloc(nrows * sizeof(float *) + nrows * ncols * sizeof(float));
    assert(M);

    // Beginning of first row
    M[0] = (float *)(M + nrows);

    for (int i = 1; i < nrows; i++) {
        // Beginning of ith row
        M[i] = M[i-1] + ncols;
    }

    return M;
}

#define free_matrix(M) free(M)

static inline void print_matrix(float **M, int m, int n) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            printf("%8.1f", M[i][j]);
        }
        printf("\n");
    }
}

#endif // MATRIX_H
