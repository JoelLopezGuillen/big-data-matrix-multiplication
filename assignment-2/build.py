"""Build the C runner, the C tests and the CPU topology / cache tools.

Usage:
    .venv\\Scripts\\python build.py
"""

import bench_common


def main():
    bench_common.build_all()
    print("gcc flags:", " ".join(bench_common.GCC_FLAGS))
    print("Built c/build/runner.exe, c/build/test_matmul.exe, "
          "tools/build/cpu_topology.exe, tools/build/cache_info.exe")


if __name__ == "__main__":
    main()
