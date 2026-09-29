#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "lcs_common.h"

int max_int(int a, int b) { return (a > b) ? a : b; }

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    /* Defaults for correctness demo; @file paths for benchmark inputs. */
    int X_alloc = 0, Y_alloc = 0;
    char *X_in = load_arg_or_file(argc >= 3 ? argv[1] : "ASTRONOMY", &X_alloc);
    char *Y_in = load_arg_or_file(argc >= 3 ? argv[2] : "AMSTERDAM", &Y_alloc);

    int m = (int) strlen(X_in);
    int n = (int) strlen(Y_in);

    /* Broadcast lengths and strings (rank 0 already has them) */
    MPI_Bcast(&m, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

    /* replicate strings on all ranks */
    char *X = malloc(m + 1);
    char *Y = malloc(n + 1);
    if (!X || !Y) { fprintf(stderr, "malloc failed\n"); MPI_Abort(MPI_COMM_WORLD, 1); }
    if (rank == 0) {
        memcpy(X, X_in, m + 1);
        memcpy(Y, Y_in, n + 1);
    }
    MPI_Bcast(X, m + 1, MPI_CHAR, 0, MPI_COMM_WORLD);
    MPI_Bcast(Y, n + 1, MPI_CHAR, 0, MPI_COMM_WORLD);

    /* allocate full matrix L locally: (m+1) x (n+1) */
    size_t rows = (size_t)(m + 1);
    size_t cols = (size_t)(n + 1);
    int *L = calloc(rows * cols, sizeof(int));
    if (!L) { fprintf(stderr, "calloc L failed\n"); MPI_Abort(MPI_COMM_WORLD, 1); }

    /* initialize first row and column already zero by calloc */

    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();

    /* Buffers for diagonal communication:
       diagBuf: size n+1, init = -1 means "I didn't compute this j".
       diagAll: result after MPI_Allreduce (MPI_MAX), values >= 0 are valid.
    */
    int *diagBuf = malloc((n + 1) * sizeof(int));
    int *diagAll = malloc((n + 1) * sizeof(int));
    if (!diagBuf || !diagAll) { fprintf(stderr, "malloc diag failed\n"); MPI_Abort(MPI_COMM_WORLD, 1); }

    /* Process anti-diagonals s = i + j, with i in [1..m], j in [1..n] */
    for (int s = 2; s <= m + n; ++s) {
        /* prepare diagBuf */
        for (int j = 0; j <= n; ++j) diagBuf[j] = -1;

        /* i ranges */
        int i_start = (s - n > 1) ? (s - n) : 1;
        int i_end   = (s - 1 < m) ? (s - 1) : m;

        /* compute owned elements: choose ownership policy i % size == rank */
        for (int i = i_start; i <= i_end; ++i) {
            if ((i % size) != rank) continue;
            int j = s - i;
            if (j < 1 || j > n) continue;

            /* indexes in L: idx(i,j) = i*(n+1) + j */
            int up   = L[(i - 1) * (n + 1) + j];     /* from diag s-1 */
            int left = L[i * (n + 1) + (j - 1)];     /* from diag s-1 */
            int diag = L[(i - 1) * (n + 1) + (j - 1)]; /* from diag s-2 */

            if (X[i - 1] == Y[j - 1]) {
                diagBuf[j] = diag + 1;
            } else {
                diagBuf[j] = max_int(up, left);
            }
        }

        /* allreduce diagonal with MPI_MAX (since uncomputed entries = -1) */
        MPI_Allreduce(diagBuf, diagAll, n + 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

        /* update local matrix L with diagAll for this diagonal */
        for (int j = 1; j <= n; ++j) {
            int val = diagAll[j];
            if (val >= 0) {
                int i = s - j;
                if (i >= 1 && i <= m) {
                    L[i * (n + 1) + j] = val;
                }
            }
        }

        /* next diagonal */
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double t1 = MPI_Wtime();

    /* After DP each rank has the full matrix L replicated; reconstruct on rank 0 */
    if (rank == 0) {
        int lcs_len = L[m * (n + 1) + n];
        char *lcs_seq = malloc((size_t)lcs_len + 1);
        if (!lcs_seq) { fprintf(stderr, "malloc lcs_seq failed\n"); MPI_Abort(MPI_COMM_WORLD, 1); }
        lcs_seq[lcs_len] = '\0';

        int ii = m, jj = n, k = lcs_len - 1;
        while (ii > 0 && jj > 0) {
            if (X[ii - 1] == Y[jj - 1]) {
                lcs_seq[k--] = X[ii - 1];
                ii--; jj--;
            } else {
                int up = L[(ii - 1) * (n + 1) + jj];
                int left = L[ii * (n + 1) + (jj - 1)];
                if (up >= left) ii--; else jj--;
            }
        }

        printf("MPI processes=%d\n m=%d n=%d\n LCS length=%d\n Sequence itself=%s\n time=%.6f seconds\n",
               size, m, n, lcs_len, lcs_seq, t1 - t0);

        free(lcs_seq);
    }

    free(diagBuf);
    free(diagAll);
    free(L);
    free(X);
    free(Y);
    if (X_alloc) free(X_in);
    if (Y_alloc) free(Y_in);

    MPI_Finalize();
    return 0;
}
