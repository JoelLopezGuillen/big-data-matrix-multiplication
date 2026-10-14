"""Capture the hardware/software environment of a benchmark run as JSON.

Usage:
    .venv\\Scripts\\python tools\\capture_environment.py [--output results/environment.json]
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


def first_line(cmd, use_stderr=False):
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True)
    except OSError as error:
        return f"unavailable ({error})"
    text = proc.stderr if use_stderr else proc.stdout
    return text.strip().splitlines()[0] if text.strip() else ""


def capture(config):
    topology = bench_common.cpu_topology(config)
    java_exe = bench_common.java_executable()
    java_version = subprocess.run([java_exe, "-version"], capture_output=True, text=True)
    return {
        "timestamp": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "cpu": {
            "model": cpu_model(),
            "physical_cores": psutil.cpu_count(logical=False),
            "logical_processors": psutil.cpu_count(logical=True),
            "performance_core_logical_processors": bench_common.performance_cores(topology),
            "efficiency_core_logical_processors": bench_common.efficiency_cores(topology),
            "topology": topology["logical_processors"],
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
            "implementation": platform.python_implementation(),
            "benchmark_interpreter": bench_common.python_executable(),
            "numpy_version": np.__version__,
            "psutil_version": psutil.__version__,
        },
        "java": {
            "executable": java_exe,
            "version": java_version.stderr.strip().splitlines(),
            "jvm_flags": config["java_flags"],
            "javac_flags": config["javac_flags"],
        },
        "c": {
            "compiler": first_line(["gcc", "--version"]),
            "gcc_flags": config["gcc_flags"],
        },
        "benchmark": {
            "sizes": config["sizes"],
            "repetitions": config["repetitions"],
            "warmups": config["warmups"],
            "timeout_seconds": config["timeout_seconds"],
            "pin_to_performance_cores": config["pin_to_performance_cores"],
            "priority_class": config["priority_class"],
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Capture the benchmark environment.")
    parser.add_argument("--config", default=bench_common.DEFAULT_CONFIG)
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
