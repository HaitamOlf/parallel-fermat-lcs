#!/usr/bin/env python3
"""
Generate benchmark input strings for the LCS variants.

Emits X_<size>.txt and Y_<size>.txt for each size in a fixed set. Both files
in a size pair are the same length. Alphabet is uppercase A..Z; that keeps
the strings shell-safe (no metacharacters) so they can also be passed on
argv without quoting drama, though the driver scripts use @file mode.

Seeded RNG so runs are reproducible.
"""

import argparse
import random
from pathlib import Path

SIZES = (2000, 5000, 10000)
ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
SEED = 20260901   # fixed seed so bench outputs are reproducible


def gen_string(rng: random.Random, length: int) -> str:
    return "".join(rng.choice(ALPHABET) for _ in range(length))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir", type=Path,
                    help="directory to write X_<n>.txt / Y_<n>.txt into")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for size in SIZES:
        # Two independent RNGs per size, both derived deterministically
        # from SEED and size so runs are reproducible.
        rng_x = random.Random(SEED * 2 + size)
        rng_y = random.Random(SEED * 2 + size + 1)
        X = gen_string(rng_x, size)
        Y = gen_string(rng_y, size)
        (args.out_dir / f"X_{size}.txt").write_text(X)
        (args.out_dir / f"Y_{size}.txt").write_text(Y)
        print(f"wrote X_{size}.txt, Y_{size}.txt ({size} chars each)")


if __name__ == "__main__":
    main()
