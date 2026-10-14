"""Build the C runner/tests, the Java classes and the CPU topology tool.

Usage:
    .venv\\Scripts\\python build.py [--config bench_config.json]
"""

import argparse

import bench_common


def main():
    parser = argparse.ArgumentParser(description="Build all native artefacts.")
    parser.add_argument("--config", default=bench_common.DEFAULT_CONFIG)
    args = parser.parse_args()
    config = bench_common.load_config(args.config)
    bench_common.build_all(config)
    print("gcc flags:  ", " ".join(config["gcc_flags"]))
    print("javac flags:", " ".join(config["javac_flags"]))
    print("Built c/build/runner.exe, c/build/test_matmul.exe, java/out/*.class, "
          "tools/build/cpu_topology.exe")


if __name__ == "__main__":
    main()
