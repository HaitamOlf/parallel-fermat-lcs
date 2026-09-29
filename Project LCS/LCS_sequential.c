#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "lcs_common.h"

/* Wall-clock monotonic timer (matches MPI_Wtime granularity in the MPI
   variant, and clock_gettime in UPC/CUDA). clock() from <time.h> measures
   CPU time, not elapsed time, and gives misleading numbers under threading. */
static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec / 1e9;
}

int main(int argc, char **argv) {

    /* Defaults are the correctness fixture; @file paths load benchmark inputs. */
    int X_alloc = 0, Y_alloc = 0;
    char *X = load_arg_or_file(argc >= 3 ? argv[1] : "ASTRONOMY", &X_alloc);
    char *Y = load_arg_or_file(argc >= 3 ? argv[2] : "AMSTERDAM", &Y_alloc);

    int m = strlen(X);
    int n = strlen(Y);

    // Allocating matrix (m+1)x(n+1)
    int **L = malloc((m + 1) * sizeof(int *));
    for (int i = 0; i <= m; i++)
        L[i] = calloc(n + 1, sizeof(int));

    // Timing the DP fill only (matches MPI / UPC / CUDA timed regions:
    // no setup, no allocation, no reconstruction, no I/O).
    double t_start = now_sec();

    // Phase 1 : Construction of LCS matrix
    for (int i = 1; i <= m; i++) {
        for (int j = 1; j <= n; j++) {
            if (X[i - 1] == Y[j - 1])
                L[i][j] = L[i - 1][j - 1] + 1;
            else
                L[i][j] = (L[i - 1][j] > L[i][j - 1])
                          ? L[i - 1][j]
                          : L[i][j - 1];
        }
    }

    double t_end = now_sec();
    double elapsed = t_end - t_start;

    int lcs_len = L[m][n];

    // Phase 2 : Reconstruction of LCS matrix (untimed, matches other variants)
    char *LCS = malloc(lcs_len + 1);
    LCS[lcs_len] = '\0';

    int i = m, j = n, k = lcs_len - 1;

    while (i > 0 && j > 0) {
        if (X[i - 1] == Y[j - 1]) {
            LCS[k--] = X[i - 1];
            i--;
            j--;
        } else if (L[i - 1][j] >= L[i][j - 1]) {
            i--;
        } else {
            j--;
        }
    }

    // Printing the results
    printf("LCS length = %d\n", lcs_len);
    printf("LCS = %s\n", LCS);
    printf("DP fill time = %.6f seconds\n", elapsed);

    // Free allocated memory
    for (int i2 = 0; i2 <= m; i2++) free(L[i2]);
    free(L);
    free(LCS);
    if (X_alloc) free(X);
    if (Y_alloc) free(Y);

    return 0;
}
