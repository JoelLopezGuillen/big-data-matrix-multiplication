"""Build and run every correctness check: C unit tests, Python tests, cross-check.

Usage:
    .venv\\Scripts\\python run_tests.py
Exits non-zero if any suite fails.
"""

import os
import subprocess
import sys

import bench_common


def run(label, cmd):
    print(f"=== {label}: {' '.join(cmd)}", flush=True)
    code = subprocess.run(cmd, cwd=bench_common.ROOT).returncode
    print(f"=== {label}: {'PASSED' if code == 0 else 'FAILED'}\n", flush=True)
    return code == 0


def main():
    bench_common.build_c()
    tmp_csr = os.path.join(bench_common.C_BUILD_DIR_REL, "test_tmp.csr")
    results = [
        run("c", [os.path.join(bench_common.ROOT, bench_common.C_TEST_REL), tmp_csr]),
        run("python", [sys.executable, "-m", "pytest", "tests", "-q"]),
        run("cross-check", [sys.executable, "cross_check.py", "--skip-build"]),
    ]
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
