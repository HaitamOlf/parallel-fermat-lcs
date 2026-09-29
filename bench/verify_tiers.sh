#!/usr/bin/env bash
# Verify RSA round-trip and Fermat correctness across all three tiers.
set -euo pipefail

cd ~/parallel-prog/"Project RSA"

for tier in SMALL MEDIUM LARGE; do
    if [ "$tier" = "MEDIUM" ]; then
        def=""
    else
        def="-DTIER_${tier}"
    fi

    echo "=================================================="
    echo "=== TIER: ${tier}"
    echo "=================================================="

    gcc -O2 -Wall $def -o rsa_seq rsa_sequential.c
    gcc -O2 -Wall $def -o fermat_seq fermat_sequential.c -lm

    echo "-- rsa_seq --"
    ./rsa_seq | grep -E 'p =|q =|n =|Round-trip'

    echo "-- fermat_seq --"
    ./fermat_seq | grep -E 'n =|p =|q =|Matches|Factorization'
    echo
done
