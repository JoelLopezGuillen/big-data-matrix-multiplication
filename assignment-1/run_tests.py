"""Build and run the correctness test suites of all three implementations.

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
    config = bench_common.load_config()
    bench_common.build_c(config)
    bench_common.build_java(config)
    results = [
        run("python", [sys.executable, "-m", "pytest", "python", "-q"]),
        run("java", [bench_common.java_executable(), "-cp", bench_common.JAVA_OUT_REL,
                     "MatMulTest"]),
        run("c", [os.path.join(bench_common.ROOT, bench_common.C_TEST_REL)]),
    ]
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
