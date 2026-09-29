#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
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

/* ---------- Fermat Factorization ---------- */
int main() {
    /* Shared RSA modulus (see rsa_common.h) */
    uint64_t p_real = RSA_P;
    uint64_t q_real = RSA_Q;
    uint64_t n = p_real * q_real;

    uint64_t a, b2, b;
    uint64_t p, q;

    /* Start timing */
    double t1 = now_ms();

    /* Fermat initialization */
    a = (uint64_t)ceil(sqrt((double)n));

    while (1) {
        b2 = a * a - n;

        if (is_perfect_square(b2, &b)) {
            p = a - b;
            q = a + b;
            break;
        }
        a++;
    }

    /* Stop timing */
    double t2 = now_ms();

    /* Output */
    printf("Sequential Fermat Factorization\n");
    printf("-------------------------------\n");
    printf("n = %lu\n\n", n);

    printf("Recovered factors:\n");
    printf("p = %lu\n", p);
    printf("q = %lu\n", q);
    printf("p * q = %lu\n", p * q);
    printf("Matches n : %s\n\n", (p * q == n) ? "yes" : "NO");

    printf("Factorization time: %.6f ms\n", t2 - t1);

    return 0;
}
