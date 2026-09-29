#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
#include <upc.h>
#include "rsa_common.h"

/* ---------- Timing utility ---------- */
double now_ms() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

/* ---------- Check perfect square ---------- */
int is_perfect_square(uint64_t x, uint64_t *root) {
    uint64_t r = (uint64_t)(sqrt((double)x));
    if (r * r == x) {
        *root = r;
        return 1;
    }
    if ((r + 1) * (r + 1) == x) {
        *root = r + 1;
        return 1;
    }
    return 0;
}

/* ---------- Shared termination flag ---------- */
shared int found = 0;
shared uint64_t p_shared, q_shared;

int main() {
    /* Shared RSA modulus (see rsa_common.h) */
    uint64_t p_real = RSA_P;
    uint64_t q_real = RSA_Q;
    uint64_t n = p_real * q_real;

    uint64_t a_start = (uint64_t)ceil(sqrt((double)n));
    uint64_t a, b2, b;

    upc_barrier;

    double t1 = 0.0;
    if (MYTHREAD == 0)
        t1 = now_ms();

    /* Each thread checks a different sequence of a values.
       Because b = (q-p)/2 is unique for this n, exactly one thread hits
       the target — so the writes to p_shared/q_shared are race-free in
       practice even though the `found` flag has no atomicity. */
    for (a = a_start + MYTHREAD; !found; a += THREADS) {
        b2 = a * a - n;

        if (is_perfect_square(b2, &b)) {
            p_shared = a - b;
            q_shared = a + b;
            found = 1;
        }
    }

    upc_barrier;

    if (MYTHREAD == 0) {
        double t2 = now_ms();

        printf("UPC Fermat Factorization\n");
        printf("------------------------\n");
        printf("n = %lu\n\n", n);

        uint64_t p_local = p_shared;
        uint64_t q_local = q_shared;

        printf("Recovered factors:\n");
        printf("p = %lu\n", p_local);
        printf("q = %lu\n", q_local);
        printf("p * q = %lu\n", p_local * q_local);
        printf("Matches n : %s\n\n", (p_local * q_local == n) ? "yes" : "NO");

        printf("Factorization time: %.6f ms\n", t2 - t1);
        printf("UPC threads used : %d\n", THREADS);
    }

    return 0;
}
