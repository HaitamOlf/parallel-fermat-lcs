#ifndef RSA_COMMON_H
#define RSA_COMMON_H

#include <stdint.h>

/*
 * Shared RSA modulus for rsa_sequential and all four Fermat variants.
 *
 * WHAT THIS IS (and isn't):
 *
 * This project uses Fermat factorization as a BENCHMARK WORKLOAD to
 * demonstrate how an embarrassingly parallel search scales across four
 * paradigms. It is NOT an attack on real RSA. Real 2048-bit RSA is
 * immune to Fermat unless the primes are within ~n^(1/4) of each other,
 * which requires deliberately adversarial or bugged keygen; this
 * project's primes are ~2^31 apart, roughly 40,000x farther apart than
 * n^(1/4) would demand for even this toy 63-bit n. Fermat cracks the
 * modulus quickly here for one reason only: n is a deliberately
 * UNDERSIZED 63-BIT MODULUS, so the O(g^2/p) search fits in seconds
 * on a laptop core. Real RSA moduli are 2048-4096 bits and remain
 * effectively unreachable by any known factoring algorithm.
 *
 * The RSA encrypt/decrypt in rsa_sequential.c uses the same n as a
 * self-consistency check: we set up RSA, then break it, so the recovered
 * factors must equal the ones we generated the key from. If they don't,
 * either the arithmetic overflowed or the primality assumption on p or q
 * is wrong.
 *
 * TIER SELECTION:
 *
 *   -DTIER_SMALL   : q ~= p + 6e8,  ~2e7 fermat iters,  ~50ms   seq
 *   (default)      : q ~= p + 2e9,  ~2e8 fermat iters,  ~500ms  seq   (MEDIUM)
 *   -DTIER_LARGE   : q ~= p + 4e9,  ~6e8 fermat iters,  ~2-3s   seq
 *
 * All tiers keep (p+q)/2 < 2^32 so the CUDA a*a arithmetic never
 * overflows uint64. Times are rough; actual walls come out of step-5
 * benchmarking on the reference Lenovo. p is fixed at Mersenne M31
 * across tiers; only q varies. All four q values are prime (verified
 * by Miller-Rabin in bench/gen_primes.py; source of the constants).
 */

#define P_FIXED  2147483647ULL   /* 2^31 - 1, Mersenne prime M31 */
#define Q_SMALL  2747483647ULL   /* smallest prime >= P_FIXED + 6e8   */
#define Q_MEDIUM 4294967291ULL   /* 2^32 - 5, largest prime < 2^32    */
#define Q_LARGE  6147483653ULL   /* smallest prime >= P_FIXED + 4e9   */

#if defined(TIER_SMALL)
  #define RSA_P P_FIXED
  #define RSA_Q Q_SMALL
  #define RSA_TIER_NAME "SMALL"
#elif defined(TIER_LARGE)
  #define RSA_P P_FIXED
  #define RSA_Q Q_LARGE
  #define RSA_TIER_NAME "LARGE"
#else
  #define RSA_P P_FIXED
  #define RSA_Q Q_MEDIUM
  #define RSA_TIER_NAME "MEDIUM"
#endif

#define RSA_E 65537ULL

#endif
