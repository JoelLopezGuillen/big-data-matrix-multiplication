"""Benchmark / verification runner for the pure-Python kernel.

Usage:
    runner.py bench  <n> <data_dir> <warmups> <repetitions>
    runner.py verify <n> <data_dir> <output_file>

`bench` prints exactly one JSON line with per-repetition kernel times (ns) and
checksums, then blocks reading one line from stdin so that the harness can
sample the peak working set of this still-alive process.

Only the standard library is used so the script can run on the base
interpreter (on Windows the venv python.exe is a launcher that spawns a second
process, which would hide the real memory usage from the harness).
"""

import json
import os
import sys
import time
from array import array

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from matmul import checksum, matmul  # noqa: E402

EXIT_OOM = 3


def load_matrix(path, n):
    """Load an n*n little-endian float64 row-major file as a list of lists."""
    flat = array("d")
    with open(path, "rb") as handle:
        flat.fromfile(handle, n * n)
    if sys.byteorder != "little":
        flat.byteswap()
    rows = [flat[i * n:(i + 1) * n].tolist() for i in range(n)]
    del flat
    return rows


def save_matrix(path, c):
    """Write a list-of-lists matrix as little-endian float64 row-major."""
    flat = array("d")
    for row in c:
        flat.extend(row)
    if sys.byteorder != "little":
        flat.byteswap()
    with open(path, "wb") as handle:
        flat.tofile(handle)


def load_inputs(n, data_dir):
    a = load_matrix(os.path.join(data_dir, f"A_n{n}.bin"), n)
    b = load_matrix(os.path.join(data_dir, f"B_n{n}.bin"), n)
    return a, b


def bench(n, data_dir, warmups, repetitions):
    a, b = load_inputs(n, data_dir)
    warmup_sink = 0.0
    for _ in range(warmups):
        c = matmul(a, b)
        warmup_sink += checksum(c)
        del c
    times_ns = []
    checksums = []
    for _ in range(repetitions):
        start = time.perf_counter_ns()
        c = matmul(a, b)
        end = time.perf_counter_ns()
        times_ns.append(end - start)
        checksums.append(checksum(c))
        del c
    result = {
        "language": "python",
        "n": n,
        "status": "ok",
        "warmups": warmups,
        "times_ns": times_ns,
        "checksums": checksums,
        "warmup_sink": warmup_sink,
    }
    print(json.dumps(result), flush=True)
    sys.stdin.readline()


def verify(n, data_dir, output_file):
    a, b = load_inputs(n, data_dir)
    save_matrix(output_file, matmul(a, b))


def main(argv):
    try:
        if len(argv) == 6 and argv[1] == "bench":
            bench(int(argv[2]), argv[3], int(argv[4]), int(argv[5]))
        elif len(argv) == 5 and argv[1] == "verify":
            verify(int(argv[2]), argv[3], argv[4])
        else:
            print(__doc__, file=sys.stderr)
            return 2
    except MemoryError:
        print(json.dumps({"language": "python", "status": "oom"}), flush=True)
        return EXIT_OOM
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
