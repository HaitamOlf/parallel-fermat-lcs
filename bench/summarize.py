#!/usr/bin/env python3
"""Quick tabular summary of median times + speedups from both CSVs."""

import csv
import statistics
import sys
from collections import defaultdict
from pathlib import Path


def summarize(csv_path: Path, key_col: str, label: str,
              key_order, variants=("seq", "mpi", "upc"),
              thread_counts=(1, 2, 4, 8, 16)):
    data = defaultdict(list)
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            t = float(row["time_ms"])
            data[(row[key_col], row["variant"], int(row["threads"]))].append(t)

    print(f"\n=== {label} medians (ms) & speedup vs seq baseline ===")
    print(f"{key_col:<8} {'variant':<7} {'np':>3}  {'median_ms':>10}  {'speedup':>7}")

    seq_med = {k: statistics.median(v)
               for (k, var, n), v in data.items() if var == "seq"}

    for k in key_order:
        for var in variants:
            for np in thread_counts:
                key = (k, var, np)
                if key not in data:
                    continue
                med = statistics.median(data[key])
                spd = seq_med[k] / med if k in seq_med else float("nan")
                print(f"{k:<8} {var:<7} {np:>3}  {med:>10.2f}  {spd:>7.2f}x")


def main():
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.home() / "parallel-prog" / "bench"
    summarize(root / "fermat_results.csv", "tier", "FERMAT",
              key_order=("SMALL", "MEDIUM", "LARGE"))
    summarize(root / "lcs_results.csv", "size", "LCS",
              key_order=("2000", "5000", "10000"))


if __name__ == "__main__":
    main()
