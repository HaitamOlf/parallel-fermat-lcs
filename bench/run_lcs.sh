#!/usr/bin/env bash
# Benchmark LCS variants (seq, MPI, UPC) across three input sizes on the Lenovo.
# CUDA LCS runs on Colab (colab_cells.md).
#
# Output CSV columns: size,variant,threads,run,time_ms
# Emitted to bench/lcs_results.csv (overwritten each invocation).

set -euo pipefail
export PATH=$PATH:/usr/local/berkeley_upc/opt/bin

REPO=~/parallel-prog
LCS="$REPO/Project LCS"
OUT="$REPO/bench/lcs_results.csv"
REPEATS=5
THREAD_COUNTS=(1 2 4 8 16)
SIZES=(2000 5000 10000)

echo "size,variant,threads,run,time_ms" > "$OUT"

cd "$LCS"

gcc  -O2 -Wall -o lcs_seq  LCS_sequential.c
mpicc -O2 -Wall -o lcs_mpi LCS_mpi.c
upcc LCS_upc.c -o lcs_upc >/dev/null 2>&1

# lcs_seq output: "DP fill time = X.YYYY seconds"
# lcs_mpi output: "time=X.YYYY seconds"
# lcs_upc output: "..., DP fill time: X.YYYY"
# Anchor to the number after the label so field position doesn't matter.
extract_ms_seq() { grep -oE 'DP fill time = [0-9.]+' | awk '{print $NF * 1000}'; }
extract_ms_mpi() { grep -oE 'time=[0-9.]+'          | awk -F= '{print $2 * 1000}'; }
extract_ms_upc() { grep -oE 'DP fill time: [0-9.]+' | awk '{print $NF * 1000}'; }

for size in "${SIZES[@]}"; do
    X="@X_${size}.txt"
    Y="@Y_${size}.txt"
    echo "=== SIZE: $size ==="

    echo "-- seq --"
    for r in $(seq 1 $REPEATS); do
        ms=$(./lcs_seq "$X" "$Y" | extract_ms_seq)
        echo "  run $r: $ms ms"
        echo "$size,seq,1,$r,$ms" >> "$OUT"
    done

    for np in "${THREAD_COUNTS[@]}"; do
        echo "-- mpi np=$np --"
        for r in $(seq 1 $REPEATS); do
            ms=$(mpirun --oversubscribe -np $np ./lcs_mpi "$X" "$Y" 2>/dev/null | extract_ms_mpi)
            echo "  run $r: $ms ms"
            echo "$size,mpi,$np,$r,$ms" >> "$OUT"
        done

        echo "-- upc np=$np --"
        for r in $(seq 1 $REPEATS); do
            ms=$(upcrun -shared-heap 512M -np $np ./lcs_upc "$X" "$Y" 2>/dev/null | extract_ms_upc)
            echo "  run $r: $ms ms"
            echo "$size,upc,$np,$r,$ms" >> "$OUT"
        done
    done
    echo
done

echo "Wrote: $OUT"
wc -l "$OUT"
