#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "lcs_common.h"

/* CUDA error-check macro. Wraps every runtime call and kernel launch so
   a failure prints file:line and aborts instead of silently returning
   wrong results. */
#define CUDA_CHECK(call) do {                                              \
    cudaError_t _err = (call);                                             \
    if (_err != cudaSuccess) {                                             \
        fprintf(stderr, "CUDA error at %s:%d: %s\n",                       \
                __FILE__, __LINE__, cudaGetErrorString(_err));             \
        exit(EXIT_FAILURE);                                                \
    }                                                                      \
} while (0)

/* Call after every kernel launch to catch launch-config and in-kernel
   errors. cudaGetLastError picks up launch errors; the sync + check
   catches async errors from the kernel itself. */
#define CUDA_CHECK_KERNEL() do {                                           \
    CUDA_CHECK(cudaGetLastError());                                        \
    CUDA_CHECK(cudaDeviceSynchronize());                                   \
} while (0)

__global__ void compute_diagonal_kernel(const char *d_X, const char *d_Y, int *d_L, int s, int m, int n) {
    /* i ranges for this diagonal (same as MPI/UPC implementations) */
    int i_start = (s - n > 1) ? (s - n) : 1;
    int i_end   = (s - 1 < m) ? (s - 1) : m;
    int count = i_end - i_start + 1;
    if (count <= 0) return;

    int cols = n + 1;

    /* Grid-stride: each thread walks multiple diagonal cells if launched
       with fewer threads than the diagonal length. Required for the
       block/thread sweep — otherwise (1, 256) on a 10,000-length
       diagonal would leave 9,744 cells unwritten. */
    int global_tid    = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = gridDim.x  * blockDim.x;

    for (int t = global_tid; t < count; t += total_threads) {
        int i = i_start + t;
        int j = s - i;
        if (j < 1 || j > n) continue;

        int idx    = i * cols + j;
        int idx_up = (i - 1) * cols + j;
        int idx_l  = i * cols + (j - 1);
        int idx_d  = (i - 1) * cols + (j - 1);

        char a = d_X[i - 1];
        char b = d_Y[j - 1];

        int val;
        if (a == b) {
            val = d_L[idx_d] + 1;
        } else {
            int up  = d_L[idx_up];
            int lef = d_L[idx_l];
            val = (up > lef) ? up : lef;
        }
        d_L[idx] = val;
    }
}

int main(int argc, char **argv) {
    /* Usage: ./lcs_cuda [X_arg [Y_arg [blocks [threads]]]]
       X_arg / Y_arg use the load_arg_or_file convention (@file or literal).
       blocks / threads default to a fixed grid geometry that covers any
       diagonal for our benchmark sizes; the sweep in step 5 varies them. */
    int X_alloc = 0, Y_alloc = 0;
    char *X_h = load_arg_or_file(argc >= 3 ? argv[1] : "ASTRONOMY", &X_alloc);
    char *Y_h = load_arg_or_file(argc >= 3 ? argv[2] : "AMSTERDAM", &Y_alloc);

    int blocks_cfg  = (argc >= 4) ? atoi(argv[3]) : 256;
    int threads_cfg = (argc >= 5) ? atoi(argv[4]) : 256;
    if (blocks_cfg < 1)  blocks_cfg = 1;
    if (threads_cfg < 1) threads_cfg = 1;

    int m = (int)strlen(X_h);
    int n = (int)strlen(Y_h);

    // host matrix L (for final copy-back and reconstruction)
    size_t rows = (size_t)(m + 1);
    size_t cols = (size_t)(n + 1);
    size_t total = rows * cols;

    // allocate host matrix and init to zero
    int *L_h = (int *)calloc(total, sizeof(int));
    if (!L_h) { fprintf(stderr, "calloc host L failed\n"); return 1; }

    // allocate device memory for strings and matrix
    char *d_X = NULL;
    char *d_Y = NULL;
    int  *d_L = NULL;

    CUDA_CHECK(cudaMalloc((void**)&d_X, m * sizeof(char)));
    CUDA_CHECK(cudaMalloc((void**)&d_Y, n * sizeof(char)));
    CUDA_CHECK(cudaMalloc((void**)&d_L, total * sizeof(int)));

    /* Three separate event pairs so H2D copies, kernel work, and D2H
       copies are reported independently. Total elapsed = h2d + kernel + d2h. */
    cudaEvent_t e_h2d_start, e_h2d_stop;
    cudaEvent_t e_kern_start, e_kern_stop;
    cudaEvent_t e_d2h_start, e_d2h_stop;
    CUDA_CHECK(cudaEventCreate(&e_h2d_start));
    CUDA_CHECK(cudaEventCreate(&e_h2d_stop));
    CUDA_CHECK(cudaEventCreate(&e_kern_start));
    CUDA_CHECK(cudaEventCreate(&e_kern_stop));
    CUDA_CHECK(cudaEventCreate(&e_d2h_start));
    CUDA_CHECK(cudaEventCreate(&e_d2h_stop));

    /* --- H2D: strings + zero-init of L --- */
    CUDA_CHECK(cudaEventRecord(e_h2d_start, 0));
    CUDA_CHECK(cudaMemcpy(d_X, X_h, m * sizeof(char), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_Y, Y_h, n * sizeof(char), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_L, 0, total * sizeof(int)));
    CUDA_CHECK(cudaEventRecord(e_h2d_stop, 0));
    CUDA_CHECK(cudaEventSynchronize(e_h2d_stop));

    /* --- Kernel: one launch per anti-diagonal ---
       Launch geometry is fixed by argv (blocks_cfg, threads_cfg), not
       adapted per diagonal. The kernel's grid-stride loop handles
       diagonals whose length exceeds blocks_cfg * threads_cfg. */
    CUDA_CHECK(cudaEventRecord(e_kern_start, 0));
    for (int s = 2; s <= m + n; ++s) {
        int i_start = (s - n > 1) ? (s - n) : 1;
        int i_end   = (s - 1 < m) ? (s - 1) : m;
        int count = i_end - i_start + 1;
        if (count <= 0) continue;

        compute_diagonal_kernel<<<blocks_cfg, threads_cfg>>>(d_X, d_Y, d_L, s, m, n);
        CUDA_CHECK_KERNEL();
    }
    CUDA_CHECK(cudaEventRecord(e_kern_stop, 0));
    CUDA_CHECK(cudaEventSynchronize(e_kern_stop));

    /* --- D2H: matrix back for reconstruction --- */
    CUDA_CHECK(cudaEventRecord(e_d2h_start, 0));
    CUDA_CHECK(cudaMemcpy(L_h, d_L, total * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventRecord(e_d2h_stop, 0));
    CUDA_CHECK(cudaEventSynchronize(e_d2h_stop));

    float ms_h2d = 0.0f, ms_kern = 0.0f, ms_d2h = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms_h2d, e_h2d_start, e_h2d_stop));
    CUDA_CHECK(cudaEventElapsedTime(&ms_kern, e_kern_start, e_kern_stop));
    CUDA_CHECK(cudaEventElapsedTime(&ms_d2h, e_d2h_start, e_d2h_stop));

    // reconstruct LCS on host (same logic as sequential)
    int lcs_len = L_h[m * cols + n];
    char *lcs_seq = (char *)malloc((size_t)lcs_len + 1);
    if (!lcs_seq && lcs_len > 0) { fprintf(stderr,"malloc lcs_seq failed\n"); return 1; }
    if (lcs_len > 0) lcs_seq[lcs_len] = '\0';

    int ii = m, jj = n, k = lcs_len - 1;
    while (ii > 0 && jj > 0) {
        if (X_h[ii - 1] == Y_h[jj - 1]) {
            lcs_seq[k--] = X_h[ii - 1];
            ii--; jj--;
        } else {
            int up = L_h[(ii - 1) * cols + jj];
            int left = L_h[ii * cols + (jj - 1)];
            if (up >= left) ii--; else jj--;
        }
    }

    printf("CUDA LCS length = %d\n", lcs_len);
    if (lcs_len > 0 && lcs_len <= 200) printf("CUDA LCS = %s\n", lcs_seq);
    printf("Launch geometry      : blocks=%d threads=%d (total=%d)\n",
           blocks_cfg, threads_cfg, blocks_cfg * threads_cfg);
    printf("H2D transfer time    : %.3f ms\n", ms_h2d);
    printf("Kernel (DP fill) time: %.3f ms\n", ms_kern);
    printf("D2H transfer time    : %.3f ms\n", ms_d2h);
    printf("Total GPU time       : %.3f ms\n", ms_h2d + ms_kern + ms_d2h);

    // cleanup
    free(L_h);
    if (lcs_len > 0) free(lcs_seq);
    CUDA_CHECK(cudaFree(d_X));
    CUDA_CHECK(cudaFree(d_Y));
    CUDA_CHECK(cudaFree(d_L));
    cudaEventDestroy(e_h2d_start);
    cudaEventDestroy(e_h2d_stop);
    cudaEventDestroy(e_kern_start);
    cudaEventDestroy(e_kern_stop);
    cudaEventDestroy(e_d2h_start);
    cudaEventDestroy(e_d2h_stop);
    if (X_alloc) free(X_h);
    if (Y_alloc) free(Y_h);
    return 0;
}
