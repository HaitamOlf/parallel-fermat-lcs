#include <upc.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <stdlib.h>
#include "lcs_common.h"

/* Wall-clock monotonic timer (matches sequential / MPI_Wtime / CUDA event
   timers). clock() from <time.h> measures CPU time; under UPC it double-
   counts across threads and hides the wall-clock benefit of parallelism. */
static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec / 1e9;
}

int main(int argc, char **argv) {
    int X_alloc = 0, Y_alloc = 0;
    char *X = load_arg_or_file(argc >= 3 ? argv[1] : "ASTRONOMY", &X_alloc);
    char *Y = load_arg_or_file(argc >= 3 ? argv[2] : "AMSTERDAM", &Y_alloc);

    int m = (int)strlen(X);
    int n = (int)strlen(Y);
    size_t total = (size_t)(m + 1) * (n + 1);
    shared int *L = upc_all_alloc(total, sizeof(int));
    if (!L) { if (MYTHREAD==0) fprintf(stderr,"upc_all_alloc NULL\n"); return 1; }

    upc_barrier;
    for (size_t idx = MYTHREAD; idx < total; idx += THREADS) L[idx] = 0;
    upc_barrier;

    /* Time the DP fill only; exclude allocation and reconstruction so the
       measurement is comparable to the other three variants. */
    double t_start = now_sec();

    for (int s = 2; s <= m + n; ++s) {
        int i_start = (s - n > 1) ? (s - n) : 1;
        int i_end   = (s - 1 < m) ? (s - 1) : m;

        upc_forall (int i = i_start; i <= i_end; ++i; &L[(size_t)i*(n+1) + (size_t)(s-i)]) {
            int j = s - i;
            if (j < 1 || j > n) continue;
            size_t idx  = (size_t)i * (n + 1) + (size_t)j;
            size_t up   = (size_t)(i - 1) * (n + 1) + (size_t)j;
            size_t left = (size_t)i * (n + 1) + (size_t)(j - 1);
            size_t diag = (size_t)(i - 1) * (n + 1) + (size_t)(j - 1);

            if (X[i - 1] == Y[j - 1]) {
                L[idx] = L[diag] + 1;
            } else {
                int a = L[up];
                int b = L[left];
                L[idx] = (a > b) ? a : b;
            }
        }

        upc_barrier;
    }

    upc_barrier;
    double t_end = now_sec();
    double exec_time = t_end - t_start;
    int lcs_len = L[(size_t)m * (n + 1) + (size_t)n];

    upc_barrier;
    int *localL = NULL;
    if (MYTHREAD == 0) {
        localL = malloc(sizeof(int) * (size_t)(m + 1) * (n + 1));
        if (!localL) { fprintf(stderr,"malloc localL failed\n"); upc_barrier; return 1; }
        for (int ii = 0; ii <= m; ++ii)
            for (int jj = 0; jj <= n; ++jj)
                localL[ii*(n+1)+jj] = L[(size_t)ii*(n+1)+(size_t)jj];
    }
    upc_barrier;

    if (MYTHREAD == 0) {
        char *lcs_seq = malloc((size_t)lcs_len + 1);
        lcs_seq[lcs_len] = '\0';
        int idx = lcs_len, ii = m, jj = n;
        while (ii > 0 && jj > 0) {
            int up = localL[(ii-1)*(n+1) + jj];
            int left = localL[ii*(n+1) + (jj-1)];
            int diag = localL[(ii-1)*(n+1) + (jj-1)];
            (void)diag;
            if (X[ii-1] == Y[jj-1]) { lcs_seq[idx-1] = X[ii-1]; idx--; ii--; jj--; }
            else if (up >= left) ii--; else jj--;
        }
        printf("Threads: %d, LCS: %d, Seq: %s, DP fill time: %.6f\n",
               THREADS, lcs_len, lcs_seq, exec_time);
        free(lcs_seq); free(localL);
    }
    upc_barrier;
    upc_all_free(L);
    upc_barrier;
    if (X_alloc) free(X);
    if (Y_alloc) free(Y);
    return 0;
}
