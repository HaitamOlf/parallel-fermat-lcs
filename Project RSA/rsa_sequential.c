#include <stdio.h>
#include <stdint.h>
#include <time.h>
#include "rsa_common.h"

/* ---------- Timing utility ---------- */
double now_ms() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

/* ---------- 128-bit-assisted mulmod (prevents overflow when p,q ~ 2^32) ---------- */
static inline uint64_t mulmod_u64(uint64_t a, uint64_t b, uint64_t m) {
    return (uint64_t)(((__uint128_t)a * b) % m);
}

/* ---------- Euclidean Algorithm ---------- */
uint64_t gcd(uint64_t a, uint64_t b) {
    while (b != 0) {
        uint64_t t = b;
        b = a % b;
        a = t;
    }
    return a;
}

/* ---------- Modular Inverse (iterative extended Euclidean, __int128 to be safe) ---------- */
uint64_t mod_inverse(uint64_t e, uint64_t phi) {
    __int128 old_r = (__int128)e, r = (__int128)phi;
    __int128 old_s = 1, s = 0;

    while (r != 0) {
        __int128 q = old_r / r;
        __int128 tmp = r;
        r = old_r - q * r;
        old_r = tmp;

        tmp = s;
        s = old_s - q * s;
        old_s = tmp;
    }

    if (old_r != 1) return 0;

    __int128 result = old_s % (__int128)phi;
    if (result < 0) result += (__int128)phi;
    return (uint64_t)result;
}

/* ---------- Modular Exponentiation (uses mulmod_u64) ---------- */
uint64_t modexp(uint64_t base, uint64_t exp, uint64_t mod) {
    uint64_t result = 1;
    base %= mod;

    while (exp > 0) {
        if (exp & 1)
            result = mulmod_u64(result, base, mod);

        base = mulmod_u64(base, base, mod);
        exp >>= 1;
    }

    return result;
}

/* ---------- Main ---------- */
int main() {
    /* Shared RSA parameters (see rsa_common.h) */
    uint64_t p = RSA_P;
    uint64_t q = RSA_Q;

    uint64_t n   = p * q;
    /* phi = (p-1)*(q-1). With p ~ 2^31 and q ~ 2^32, phi < 2^63; uint64 is safe. */
    uint64_t phi = (p - 1) * (q - 1);
    uint64_t e   = RSA_E;

    if (gcd(e, phi) != 1) {
        printf("Invalid RSA parameters\n");
        return 1;
    }

    uint64_t d = mod_inverse(e, phi);

    uint64_t message = 123456;
    uint64_t cipher, decrypted;

    /* ---------- Encryption timing ---------- */
    double t1 = now_ms();
    cipher = modexp(message, e, n);
    double t2 = now_ms();

    /* ---------- Decryption timing ---------- */
    double t3 = now_ms();
    decrypted = modexp(cipher, d, n);
    double t4 = now_ms();

    /* ---------- Output ---------- */
    printf("RSA Sequential Implementation\n");
    printf("-----------------------------\n");
    printf("p = %lu\n", p);
    printf("q = %lu\n", q);
    printf("n = %lu\n", n);
    printf("e = %lu\n", e);
    printf("d = %lu\n\n", d);

    printf("Original message : %lu\n", message);
    printf("Encrypted message: %lu\n", cipher);
    printf("Decrypted message: %lu\n", decrypted);
    printf("Round-trip OK    : %s\n\n", (decrypted == message) ? "yes" : "NO");

    printf("Encryption time : %.6f ms\n", t2 - t1);
    printf("Decryption time : %.6f ms\n", t4 - t3);
    printf("Total RSA time  : %.6f ms\n", (t2 - t1) + (t4 - t3));

    return 0;
}
