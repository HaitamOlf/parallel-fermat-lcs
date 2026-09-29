#!/usr/bin/env bash
# Benchmark all four Fermat variants across three modulus tiers.
#
# CPU/MPI/UPC runs happen here on the Lenovo. CUDA runs happen on Colab
# (separate script: colab_cells.md).
#
# Output CSV columns: tier,variant,threads,run,time_ms
# Emitted to bench/fermat_results.csv (overwritten each invocation).

set -euo pipefail
export PATH=$PATH:/usr/local/berkeley_upc/opt/bin

REPO=~/parallel-prog
RSA="$REPO/Project RSA"
OUT="$REPO/bench/fermat_results.csv"
REPEATS=5
THREAD_COUNTS=(1 2 4 8 16)
TIERS=(SMALL MEDIUM LARGE)

echo "tier,variant,threads,run,time_ms" > "$OUT"

cd "$RSA"

extract_ms() {
    # Both fermat_seq and fermat_mpi/upc print "Factorization time: NNN ms"
    grep 'Factorization time' | awk '{print $3}'
}

for tier in "${TIERS[@]}"; do
    def=""
    [ "$tier" != "MEDIUM" ] && def="-DTIER_${tier}"
    echo "=== Building tier: $tier ($def) ==="

    gcc  -O2 -Wall $def -o fermat_seq fermat_sequential.c -lm
    mpicc -O2 -Wall $def -o fermat_mpi fermat_mpi.c        -lm
    upcc $def fermat_upc.c -o fermat_upc >/dev/null 2>&1

    echo "-- seq --"
    for r in $(seq 1 $REPEATS); do
        ms=$(./fermat_seq | extract_ms)
        echo "  run $r: $ms ms"
        echo "$tier,seq,1,$r,$ms" >> "$OUT"
    done

    for np in "${THREAD_COUNTS[@]}"; do
        echo "-- mpi np=$np --"
        for r in $(seq 1 $REPEATS); do
            ms=$(mpirun --oversubscribe -np $np ./fermat_mpi 2>/dev/null | extract_ms)
            echo "  run $r: $ms ms"
            echo "$tier,mpi,$np,$r,$ms" >> "$OUT"
        done

        echo "-- upc np=$np --"
        for r in $(seq 1 $REPEATS); do
            ms=$(upcrun -np $np ./fermat_upc 2>/dev/null | extract_ms)
            echo "  run $r: $ms ms"
            echo "$tier,upc,$np,$r,$ms" >> "$OUT"
        done
    done
    echo
done

echo "Wrote: $OUT"
wc -l "$OUT"
