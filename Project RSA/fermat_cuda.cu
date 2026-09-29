#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include <cuda_runtime.h>
#include "rsa_common.h"

/* ---------- CUDA kernel: grid-stride Fermat search ----------
 * Each thread walks a[i] = a_start + tid + k*stride for k = 0,1,2,...
 * exiting either when it finds a b^2 = a^2 - n or when another thread
 * has flipped *found. The old kernel launched one thread per candidate
 * within a fixed window; if the answer lay past that window the search
 * silently failed. Grid-stride makes the launch geometry independent
 * of where the answer is.
 */
__global__ void fermat_kernel(uint64_t n, uint64_t a_start,
                              uint64_t *p, uint64_t *q, int *found) {
    uint64_t tid    = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t stride = (uint64_t)gridDim.x  * blockDim.x;

    /* Turing (sm_75) and newer cache global reads aggressively; a naked
     * !(*found) can be hoisted or cached in L1, so once one thread flips
     * the flag the other threads never notice and spin forever. Reading
     * through a volatile pointer forces an uncached load each iteration. */
    volatile int *vfound = found;

    /* Bound the search so `a * a` never overflows uint64. a * a fits iff
     * a < 2^32. For RSA_P = 2^31-1 and RSA_Q = 2^32-5 the target is
     * (p+q)/2 ~= 3.2e9 < 2^32, so this bound loses no valid answers.
     * Without it, threads that race past the target before `*found`
     * propagates hit overflowed arithmetic and can lock in spurious
     * factor pairs whose "product" only matches n modulo 2^64. */
    const uint64_t A_MAX = 1ULL << 32;

    for (uint64_t a = a_start + tid; a < A_MAX && !(*vfound); a += stride) {
        uint64_t b2 = a * a - n;
        uint64_t b  = (uint64_t)sqrt((double)b2);

        /* Same r / r+1 guard as the CPU is_perfect_square, since sqrt on
         * a double may round down by one for large b2. */
        if (b * b == b2) {
            if (atomicCAS(found, 0, 1) == 0) {
                *p = a - b;
                *q = a + b;
                __threadfence();
            }
            return;
        }
        if ((b + 1) * (b + 1) == b2) {
            b = b + 1;
            if (atomicCAS(found, 0, 1) == 0) {
                *p = a - b;
                *q = a + b;
                __threadfence();
            }
            return;
        }
    }
}

int main(int argc, char **argv) {
    /* Shared RSA modulus (see rsa_common.h) */
    uint64_t p_real = RSA_P;
    uint64_t q_real = RSA_Q;
    uint64_t n = p_real * q_real;

    uint64_t a_start = (uint64_t)ceil(sqrt((double)n));

    /* Usage: ./fermat_cuda [blocks [threads]]
       Defaults kept from step 4 so a bare `./fermat_cuda` behaves the
       same as before; the step-5 sweep overrides them for {1xN, Nx1, NxN}. */
    int blocks  = (argc >= 2) ? atoi(argv[1]) : 1024;
    int threads = (argc >= 3) ? atoi(argv[2]) : 256;
    if (blocks < 1)  blocks = 1;
    if (threads < 1) threads = 1;

    uint64_t *d_p, *d_q;
    int *d_found;

    uint64_t h_p = 0, h_q = 0;
    int h_found = 0;

    cudaMalloc(&d_p, sizeof(uint64_t));
    cudaMalloc(&d_q, sizeof(uint64_t));
    cudaMalloc(&d_found, sizeof(int));

    cudaMemcpy(d_found, &h_found, sizeof(int), cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    fermat_kernel<<<blocks, threads>>>(n, a_start, d_p, d_q, d_found);
    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);

    cudaMemcpy(&h_p, d_p, sizeof(uint64_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_q, d_q, sizeof(uint64_t), cudaMemcpyDeviceToHost);

    printf("CUDA Fermat Factorization\n");
    printf("-------------------------\n");
    printf("n = %lu\n\n", n);

    printf("Recovered factors:\n");
    printf("p = %lu\n", h_p);
    printf("q = %lu\n", h_q);
    printf("p * q = %lu\n", h_p * h_q);
    printf("Matches n : %s\n\n", (h_p * h_q == n) ? "yes" : "NO");

    printf("Blocks: %d  Threads/block: %d\n", blocks, threads);
    printf("Kernel time: %.6f ms\n", ms);

    cudaFree(d_p);
    cudaFree(d_q);
    cudaFree(d_found);

    return 0;
}
