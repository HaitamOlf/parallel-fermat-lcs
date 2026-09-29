#!/usr/bin/env python3
"""
Find the smallest prime >= each target, for the three Fermat benchmark tiers.

We keep p = 2^31 - 1 (Mersenne prime M31) fixed across all tiers and only vary
q, so the search space is a single parameter. The tier constant RSA_P and
RSA_Q are then defined in rsa_common.h.

Targets are chosen so sequential Fermat runtime lands roughly at:
    SMALL   ~= 50   ms
    MEDIUM  ~= 500  ms   (already the step-4 modulus)
    LARGE   ~= 3    s

Iterations ~ (q - p)^2 / (8 * p).

Uses deterministic Miller-Rabin with the witness set that covers n < 2^64:
    {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37}
No external dependencies.
"""

def is_prime(n: int) -> bool:
    if n < 2:
        return False
    small = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)
    for p in small:
        if n == p:
            return True
        if n % p == 0:
            return False

    # write n-1 as d * 2^r
    d = n - 1
    r = 0
    while d % 2 == 0:
        d //= 2
        r += 1

    for a in small:
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            continue
        for _ in range(r - 1):
            x = (x * x) % n
            if x == n - 1:
                break
        else:
            return False
    return True


def next_prime(start: int) -> int:
    n = start if start % 2 == 1 else start + 1
    while not is_prime(n):
        n += 2
    return n


def report(label: str, p: int, q: int):
    gap = q - p
    iters = (gap * gap) // (8 * p)
    a_target = (p + q) // 2
    within_bound = a_target < (1 << 32)
    print(f"--- {label} ---")
    print(f"  p        = {p}")
    print(f"  q        = {q}")
    print(f"  p * q    = {p * q}   ({(p*q).bit_length()} bits)")
    print(f"  gap      = {gap}")
    print(f"  iters ~  = {iters:,}")
    print(f"  a_target = {a_target}   (< 2^32? {within_bound})")
    print()


def main():
    P_FIXED = 2**31 - 1                     # Mersenne prime M31, kept across tiers
    Q_MEDIUM_KNOWN = 2**32 - 5              # already prime, largest prime < 2^32

    # SMALL: target ~50 ms sequential -> ~2e7 iters -> gap ~ 6e8
    q_small_target = P_FIXED + 600_000_000

    # LARGE: target ~3 s sequential -> ~9e8 iters -> gap ~ 4e9
    # Constraint: a_target = (p+q)/2 < 2^32, so q < 2^33 - p ~= 6.44e9
    q_large_target = P_FIXED + 4_000_000_000

    q_small = next_prime(q_small_target)
    q_large = next_prime(q_large_target)

    # Sanity: assert both are actually prime (Miller-Rabin is deterministic here)
    assert is_prime(P_FIXED)
    assert is_prime(Q_MEDIUM_KNOWN)
    assert is_prime(q_small)
    assert is_prime(q_large)

    print("Fermat benchmark tier moduli (all p = 2^31 - 1)")
    print()
    report("SMALL",  P_FIXED, q_small)
    report("MEDIUM", P_FIXED, Q_MEDIUM_KNOWN)
    report("LARGE",  P_FIXED, q_large)

    print("Paste into rsa_common.h:")
    print()
    print(f"#define P_FIXED  {P_FIXED}ULL")
    print(f"#define Q_SMALL  {q_small}ULL")
    print(f"#define Q_MEDIUM {Q_MEDIUM_KNOWN}ULL")
    print(f"#define Q_LARGE  {q_large}ULL")


if __name__ == "__main__":
    main()
