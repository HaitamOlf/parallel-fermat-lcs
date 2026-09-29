#!/usr/bin/env python3
"""
Generate the two speedup figures for the README:
  bench/fermat_speedup.png  — parallel scaling of the embarrassingly-parallel case
  bench/lcs_speedup.png     — anti-scaling of the dependency-bound case

Inputs:
  bench/fermat_results.csv       (Lenovo CPU: seq, MPI, UPC across 3 tiers)
  bench/fermat_cuda_results.csv  (Colab T4:   CUDA across 3 tiers × 3 configs)
  bench/lcs_results.csv          (Lenovo CPU: seq, MPI, UPC across 3 sizes)
  bench/lcs_cuda_results.csv     (Colab T4:   CUDA across 3 sizes × 3 configs)

Cross-machine caveat: CUDA numbers come from Colab T4; CPU numbers from
Lenovo i7-13620H. The GPU reference lines in each panel are drawn on the
same speedup axis for orientation, but the README makes the caveat
explicit — this is not a controlled machine-to-machine comparison.
"""

import csv
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BENCH = Path(__file__).resolve().parent
THREADS = [1, 2, 4, 8, 16]


def load_cpu(path: Path, key_col: str):
    """Returns {(key, variant, threads): [times_ms, ...]}."""
    d = defaultdict(list)
    with open(path) as f:
        for r in csv.DictReader(f):
            d[(r[key_col], r["variant"], int(r["threads"]))].append(float(r["time_ms"]))
    return d


def load_cuda(path: Path, key_col: str, time_col: str = "time_ms"):
    """Returns {(key, config): [times_ms, ...]}."""
    d = defaultdict(list)
    with open(path) as f:
        for r in csv.DictReader(f):
            d[(r[key_col], r["config"])].append(float(r[time_col]))
    return d


def med(d, key):
    return statistics.median(d[key]) if key in d and d[key] else float("nan")


# ---------------------------------------------------------------------
# Fermat figure
# ---------------------------------------------------------------------
def plot_fermat():
    cpu = load_cpu(BENCH / "fermat_results.csv", "tier")
    cuda = load_cuda(BENCH / "fermat_cuda_results.csv", "tier")
    tiers = ["SMALL", "MEDIUM", "LARGE"]
    titles = {
        "SMALL":  "SMALL tier\nn ~ 5.9e18 (~2e7 iters)",
        "MEDIUM": "MEDIUM tier\nn ~ 9.2e18 (~2e8 iters)",
        "LARGE":  "LARGE tier\nn ~ 1.3e19 (~9e8 iters)",
    }

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    for ax, tier in zip(axes, tiers):
        seq = med(cpu, (tier, "seq", 1))
        mpi = [seq / med(cpu, (tier, "mpi", n)) for n in THREADS]
        upc = [seq / med(cpu, (tier, "upc", n)) for n in THREADS]

        ax.plot(THREADS, mpi, "o-", label="MPI (Lenovo)", linewidth=2)
        ax.plot(THREADS, upc, "s-", label="UPC (Lenovo)", linewidth=2)
        ax.plot(THREADS, THREADS, "k:", alpha=0.35, label="ideal linear")
        ax.axhline(1, color="gray", linestyle="--", alpha=0.5)

        # CUDA best config as reference line
        cuda_best_ms = min(statistics.median(cuda[(tier, c)])
                           for c in ("1xN", "Nx1", "NxN"))
        cuda_spd = seq / cuda_best_ms
        ax.axhline(cuda_spd, color="crimson", linestyle=":", linewidth=1.5,
                   label=f"T4 N×N: {cuda_spd:.1f}× (kernel only)")

        ax.set_title(titles[tier], fontsize=11)
        ax.set_xlabel("thread / rank count")
        ax.set_xscale("log", base=2)
        ax.set_xticks(THREADS)
        ax.set_xticklabels([str(t) for t in THREADS])
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9, loc="upper left")

    axes[0].set_ylabel("Speedup vs sequential (higher = better)")
    fig.suptitle(
        "Fermat factorization — parallel scaling (embarrassingly parallel case)\n"
        "CPU: Intel i7-13620H (Lenovo, WSL2). GPU reference: Colab Tesla T4.",
        fontsize=12, y=1.02
    )
    fig.tight_layout()
    out = BENCH / "fermat_speedup.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")
    plt.close(fig)


# ---------------------------------------------------------------------
# LCS figure — log-scaled Y because "speedup" is < 1 everywhere
# ---------------------------------------------------------------------
def plot_lcs():
    cpu = load_cpu(BENCH / "lcs_results.csv", "size")
    cuda = load_cuda(BENCH / "lcs_cuda_results.csv", "size", time_col="kernel_ms")
    sizes = ["2000", "5000", "10000"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    for ax, size in zip(axes, sizes):
        seq = med(cpu, (size, "seq", 1))
        mpi = [seq / med(cpu, (size, "mpi", n)) for n in THREADS]
        upc = [seq / med(cpu, (size, "upc", n)) for n in THREADS]

        ax.plot(THREADS, mpi, "o-", label="MPI (Lenovo)", linewidth=2)
        ax.plot(THREADS, upc, "s-", label="UPC (Lenovo)", linewidth=2)
        ax.axhline(1, color="gray", linestyle="--", alpha=0.5,
                   label="sequential baseline")

        cuda_best_kern = min(statistics.median(cuda[(size, c)])
                             for c in ("1xN", "Nx1", "NxN"))
        cuda_spd = seq / cuda_best_kern
        ax.axhline(cuda_spd, color="crimson", linestyle=":", linewidth=1.5,
                   label=f"T4 N×N kernel: {cuda_spd:.2f}×")

        ax.set_title(f"Strings: {size} × {size} chars", fontsize=11)
        ax.set_xlabel("thread / rank count")
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xticks(THREADS)
        ax.set_xticklabels([str(t) for t in THREADS])
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=9, loc="lower left")

    axes[0].set_ylabel("Speedup vs sequential (log; below 1 = slowdown)")
    fig.suptitle(
        "LCS anti-diagonal DP — parallel scaling (dependency-bound case)\n"
        "Every variant slower than single-thread sequential. This is the expected shape.",
        fontsize=12, y=1.02
    )
    fig.tight_layout()
    out = BENCH / "lcs_speedup.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")
    plt.close(fig)


if __name__ == "__main__":
    plot_fermat()
    plot_lcs()
