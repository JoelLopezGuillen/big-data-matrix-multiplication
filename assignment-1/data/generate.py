"""Generate the benchmark input matrices A and B for each size n.

Files are raw little-endian float64 in row-major order:
    data/A_n{n}.bin  (seed 42)
    data/B_n{n}.bin  (seed 43)
Values are drawn from numpy.random.default_rng(seed) (PCG64), uniform in [-1, 1).
Generation is deterministic, so the binaries are not versioned.
"""

import argparse
import os

import numpy as np

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
SEEDS = {"A": 42, "B": 43}
LOW, HIGH = -1.0, 1.0


def matrix_path(name, n, data_dir=DATA_DIR):
    return os.path.join(data_dir, f"{name}_n{n}.bin")


def generate_matrix(seed, n):
    rng = np.random.default_rng(seed)
    return rng.uniform(LOW, HIGH, size=(n, n)).astype("<f8")


def load_matrix(name, n, data_dir=DATA_DIR):
    return np.fromfile(matrix_path(name, n, data_dir), dtype="<f8").reshape(n, n)


def ensure_inputs(n, data_dir=DATA_DIR, force=False):
    """Create A_n{n}.bin and B_n{n}.bin unless files of the right size exist."""
    expected_bytes = n * n * 8
    for name, seed in SEEDS.items():
        path = matrix_path(name, n, data_dir)
        if not force and os.path.exists(path) and os.path.getsize(path) == expected_bytes:
            continue
        generate_matrix(seed, n).tofile(path)


def main():
    parser = argparse.ArgumentParser(description="Generate benchmark input matrices.")
    parser.add_argument("sizes", type=int, nargs="+", help="matrix sizes n")
    parser.add_argument("--force", action="store_true", help="regenerate existing files")
    args = parser.parse_args()
    for n in args.sizes:
        ensure_inputs(n, force=args.force)
        print(f"n={n}: {matrix_path('A', n)}, {matrix_path('B', n)}")


if __name__ == "__main__":
    main()
