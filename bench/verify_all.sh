#!/usr/bin/env bash
# Full correctness re-check for step-5 prep.
#
# - RSA round-trip across all three tiers
# - Fermat: all four variants recover the same (p,q) for each tier (CUDA skipped
#   here; verified separately on Colab)
# - LCS: all four variants (CUDA skipped locally) agree on length & sequence
#   for the correctness fixture (ASTRONOMY/AMSTERDAM) and agree on length
#   for the 2000-char benchmark input
#
# CUDA correctness is verified on Colab after this script prints OK locally.

set -euo pipefail
export PATH=$PATH:/usr/local/berkeley_upc/opt/bin

RSA=~/parallel-prog/"Project RSA"
LCS=~/parallel-prog/"Project LCS"

pass() { echo "  PASS: $1"; }
fail() { echo "  FAIL: $1"; exit 1; }

# ------------------------------------------------------------------
echo "=== RSA + Fermat, tier sweep ==="
cd "$RSA"

for tier in SMALL MEDIUM LARGE; do
    def=""
    [ "$tier" != "MEDIUM" ] && def="-DTIER_${tier}"
    echo
    echo "--- TIER: $tier ---"

    gcc -O2 -Wall $def -o rsa_seq rsa_sequential.c
    gcc -O2 -Wall $def -o fermat_seq fermat_sequential.c -lm
    mpicc -O2 -Wall $def -o fermat_mpi fermat_mpi.c -lm
    upcc $def fermat_upc.c -o fermat_upc >/dev/null 2>&1

    # RSA round-trip
    if ./rsa_seq | grep -q "Round-trip OK    : yes"; then
        pass "rsa_seq round-trip"
    else
        fail "rsa_seq round-trip"
    fi

    # Grab expected n from rsa_seq (which prints it)
    expected_n=$(./rsa_seq | grep '^n = ' | awk '{print $3}')

    for prog in "./fermat_seq" "mpirun --oversubscribe -np 2 ./fermat_mpi" "upcrun -np 2 ./fermat_upc"; do
        out=$($prog 2>/dev/null)
        # "p * q = NNN" -> fields: p(1) *(2) q(3) =(4) NNN(5)
        got_pq=$(echo "$out" | grep '^p \* q = ' | awk '{print $5}')
        if [ "$got_pq" = "$expected_n" ]; then
            pass "$prog agrees on n"
        else
            fail "$prog: got p*q=$got_pq, expected n=$expected_n"
        fi
    done
done

# ------------------------------------------------------------------
echo
echo "=== LCS, correctness fixture (ASTRONOMY / AMSTERDAM) ==="
cd "$LCS"

gcc -O2 -Wall -o lcs_seq  LCS_sequential.c
mpicc -O2 -Wall -o lcs_mpi LCS_mpi.c
upcc LCS_upc.c -o lcs_upc >/dev/null 2>&1

seq_out=$(./lcs_seq)
mpi_out=$(mpirun --oversubscribe -np 4 ./lcs_mpi)
upc_out=$(upcrun -np 4 ./lcs_upc 2>/dev/null)

# All three should mention length 5 and sequence ASTRM
for name in seq mpi upc; do
    var="${name}_out"
    if echo "${!var}" | grep -q 'ASTRM' && echo "${!var}" | grep -qE '(length|LCS)[^0-9]*5'; then
        pass "lcs_$name: length 5 / ASTRM"
    else
        fail "lcs_$name: unexpected output: ${!var}"
    fi
done

# ------------------------------------------------------------------
echo
echo "=== LCS, benchmark input (2000-char strings) ==="

# Ensure benchmark strings exist
if [ ! -f X_2000.txt ] || [ ! -f Y_2000.txt ]; then
    fail "X_2000.txt / Y_2000.txt not present in $LCS"
fi

seq_len=$(./lcs_seq @X_2000.txt @Y_2000.txt | grep -E 'LCS length' | awk '{print $4}')
mpi_len=$(mpirun --oversubscribe -np 4 ./lcs_mpi @X_2000.txt @Y_2000.txt | grep -oE 'LCS length=[0-9]+' | head -1 | awk -F= '{print $2}')
upc_len=$(upcrun -np 4 ./lcs_upc @X_2000.txt @Y_2000.txt 2>/dev/null | grep -oE 'LCS: [0-9]+' | awk '{print $2}')

echo "  seq=$seq_len  mpi=$mpi_len  upc=$upc_len"
if [ "$seq_len" = "$mpi_len" ] && [ "$mpi_len" = "$upc_len" ] && [ -n "$seq_len" ]; then
    pass "all three variants agree on length ($seq_len)"
else
    fail "length disagreement: seq=$seq_len mpi=$mpi_len upc=$upc_len"
fi

echo
echo "=== ALL LOCAL CHECKS PASS ==="
