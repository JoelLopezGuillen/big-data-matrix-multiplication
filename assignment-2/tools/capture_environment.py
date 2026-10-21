"""Capture the hardware/software environment of a benchmark run as JSON.

Usage:
    .venv\\Scripts\\python tools\\capture_environment.py [--config bench_config_dense.json]
                                                     [--output results/environment.json]
"""

import argparse
import datetime
import json
import os
import platform
import subprocess
import sys

import numpy as np
import psutil
import scipy

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import bench_common  # noqa: E402


def cpu_model():
    if os.name == "nt":
        try:
            import winreg

            key = winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
            )
            return winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
        except OSError:
            pass
    return platform.processor()


def first_line(cmd):
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True)
    except OSError as error:
        return f"unavailable ({error})"
    return proc.stdout.strip().splitlines()[0] if proc.stdout.strip() else ""


def caches():
    """CPU caches reported by Windows (tools/cache_info.exe), grouped by level/type/size."""
    try:
        if not os.path.exists(os.path.join(bench_common.ROOT, bench_common.CACHE_EXE_REL)):
            bench_common.build_tools()
        entries = json.loads(bench_common.run_checked(
            [os.path.join(bench_common.ROOT, bench_common.CACHE_EXE_REL)]))
    except (RuntimeError, OSError, ValueError) as error:
        return {"error": str(error)}
    unique = {}
    for entry in entries:
        key = (entry["level"], entry["type"], entry["size_bytes"])
        unique.setdefault(key, {"level": entry["level"], "type": entry["type"],
                                "size_bytes": entry["size_bytes"],
                                "line_size": entry["line_size"], "instances": 0})
        unique[key]["instances"] += 1
    return sorted(unique.values(), key=lambda e: (e["level"], e["type"], e["size_bytes"]))


def capture(config):
    topology = bench_common.cpu_topology()
    benchmark = {key: value for key, value in config.items() if key != "smoke"}
    return {
        "timestamp": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "cpu": {
            "model": cpu_model(),
            "physical_cores": psutil.cpu_count(logical=False),
            "logical_processors": psutil.cpu_count(logical=True),
            "performance_core_logical_processors": bench_common.performance_cores(topology),
            "efficiency_core_logical_processors": bench_common.efficiency_cores(topology),
            "topology": topology["logical_processors"],
            "caches": caches(),
        },
        "memory_total_bytes": psutil.virtual_memory().total,
        "os": {
            "platform": platform.platform(),
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
        },
        "python": {
            "version": platform.python_version(),
            "numpy_version": np.__version__,
            "scipy_version": scipy.__version__,
            "psutil_version": psutil.__version__,
        },
        "c": {
            "compiler": first_line(["gcc", "--version"]),
            "gcc_flags": config.get("gcc_flags", bench_common.GCC_FLAGS),
        },
        "benchmark": benchmark,
    }


def main():
    parser = argparse.ArgumentParser(description="Capture the benchmark environment.")
    parser.add_argument("--config", default=bench_common.DENSE_CONFIG)
    parser.add_argument(
        "--output", default=os.path.join(bench_common.ROOT, "results", "environment.json")
    )
    args = parser.parse_args()
    environment = capture(bench_common.load_config(args.config))
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(environment, handle, indent=2)
        handle.write("\n")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
