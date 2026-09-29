#!/usr/bin/env bash
# Record hardware + toolchain versions for the results tables in the README.
# Run this on the Lenovo before the benchmark runs.
set -euo pipefail
export PATH=$PATH:/usr/local/berkeley_upc/opt/bin

OUT=~/parallel-prog/bench/env_lenovo.txt
{
    echo "=== date ==="
    date --iso-8601=seconds
    echo
    echo "=== uname -a ==="
    uname -a
    echo
    echo "=== /etc/os-release ==="
    cat /etc/os-release
    echo
    echo "=== lscpu ==="
    lscpu
    echo
    echo "=== free -h ==="
    free -h
    echo
    echo "=== gcc --version ==="
    gcc --version
    echo
    echo "=== mpicc --version ==="
    mpicc --show
    mpicc --version
    echo
    echo "=== mpirun --version ==="
    mpirun --version
    echo
    echo "=== upcc --version ==="
    upcc --version 2>&1 || true
} > "$OUT"
echo "wrote $OUT"
